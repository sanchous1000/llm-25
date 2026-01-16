import faiss
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import ollama
import time
import uuid
from langfuse import Langfuse, propagate_attributes
from dotenv import load_dotenv

load_dotenv()

class RAGEngine:
    def __init__(
        self, 
        store_dir: str, 
        embedding_model_name: str, 
        llm_model_name: str = "qwen3:4b-instruct"
    ):
        self.store_dir = Path(store_dir)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.llm_model = llm_model_name
        
        self.langfuse = None
        try:
            self.langfuse = Langfuse()
            print(f"Langfuse подключен")
        except Exception as e:
            print(f"Не удалось подключить Langfuse: {e}. Логирование отключено")

        print(f"Initializing RAG: {store_dir}")
        print(f"Embedding Model: {embedding_model_name}")
        print(f"LLM Model: {llm_model_name}")

        self.index = None
        self.metadata_map = {} 

        self._load_store()

    def _load_store(self):
        index_path = self.store_dir / "faiss.index"
        meta_path = self.store_dir / "metadata.jsonl"
        
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError("Index files not found.")

        self.index = faiss.read_index(str(index_path))
        
        print("Loading metadata from JSONL")
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                self.metadata_map[record['faiss_id']] = record['payload']
        
        print(f"Index loaded. Total records: {self.index.ntotal}")

    def _retrieve(self, query: str, k: int = 4) -> List[Dict]:
        query_vec = self.embedding_model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for faiss_id, distance in zip(indices[0], distances[0]):
            if faiss_id != -1 and faiss_id in self.metadata_map:
                payload = self.metadata_map[faiss_id]
                results.append({
                    "id": faiss_id,
                    "text": payload['text'],
                    "score": float(distance),
                    "meta": payload['metadata']
                })
        return results

    def _build_prompt(self, query: str, contexts: List[Dict]) -> str:
        context_blocks = []
        for i, ctx in enumerate(contexts):
            source = ctx['meta'].get('source_path', 'Unknown').replace("\\", "/")
            header = ctx['meta'].get('Header 1', "")
            if header: 
                header = f" ({header})"
            text_snippet = ctx['text'] 
            
            context_blocks.append(f"[Doc {i+1}] {source}{header}:\n{text_snippet}")

        context_str = "\n\n".join(context_blocks)

        prompt = f"""
You are a helpful and intelligent assistant. Your task is to answer the user's question using ONLY the provided context below.

INSTRUCTIONS:
1. Provide a clear and concise answer based strictly on the context.
2. If the answer is not in the context, strictly say: "I could not find information about this in the provided documents."
3. Cite the sources used in your answer using the format [Doc X], where X is the document number from the list.
4. Do not use any outside knowledge or make up facts.

CONTEXT:
{context_str}

QUESTION:
{query}

ANSWER:
"""
        return prompt

    def _format_response(self, answer_text: str, contexts: List[Dict]) -> Dict[str, Any]:
        sources = []
        for i, ctx in enumerate(contexts):
            source_path = ctx['meta'].get('source_path', '').replace("\\", "/")
            sources.append({
                "id": i + 1,
                "source": source_path,
                "section": ctx['meta'].get('Header 1', 'Root'),
                "page": ctx['meta'].get('page', ''),
                "snippet": ctx['text'][:150] + "..." if len(ctx['text']) > 150 else ctx['text']
            })

        return {
            "answer": answer_text.strip(),
            "sources": sources
        }

    def ask(self, query: str, k: int = 4, user_id: str = "anonymous", session_id: str = None) -> Dict[str, Any]:
        if not self.langfuse:
            raise Exception("Langfuse not connected")

        with self.langfuse.start_as_current_observation(
            as_type="span",
            name="RAG-Pipeline",
            input=query
        ) as root_trace:
            
            with self.langfuse.start_as_current_observation(
                as_type="span",
                name="Document-Retrieval",
                metadata={"k": k}
            ) as retrieval_span:
                retrieved_docs = self._retrieve(query, k=k)
                retrieval_span.update(
                    output={"count": len(retrieved_docs)},
                    metadata={"k": k,
                        "retrieved_ids": [doc["id"] for doc in retrieved_docs],
                        "retrieved_scores": [doc.get("score") for doc in retrieved_docs],
                        "chunk_contents": [doc.get("text")[:100] for doc in retrieved_docs]
                    }
                )
            
            if not retrieved_docs:
                root_trace.update(
                    output="No documents found",
                    metadata={"status": "no_docs"}
                )
                return {"answer": "I could not find any relevant documents...", "sources": []}
            
            prompt = self._build_prompt(query, retrieved_docs)
            
            with self.langfuse.start_as_current_observation(
                as_type="span",
                name="RAG-Query",
                input=query,
                metadata={"k": k, "model": self.llm_model, "retrieved_count": len(retrieved_docs)}
            ) as trace:
                
                with propagate_attributes(user_id=user_id, session_id=session_id):
                    with self.langfuse.start_as_current_observation(
                        as_type="generation",
                        name="LLM-Answer",
                        model=self.llm_model,
                        input=prompt,
                        metadata={"retrieved_docs_count": len(retrieved_docs)}
                    ) as generation:
                        
                        try:
                            response = ollama.chat(
                                model=self.llm_model, 
                                messages=[
                                    {
                                        'role': 'system', 
                                        'content': 'You are a strict factual assistant. You answer only based on the provided context and cite your sources.'
                                    }, 
                                    {
                                        'role': 'user', 
                                        'content': prompt
                                    }
                                ]
                            )
                            llm_answer = response['message']['content']
                            
                        except Exception as e:
                            llm_answer = f"Error generating answer: {e}"
                        
                        root_trace.update(output=llm_answer)
                        trace.update(output=llm_answer)
                        generation.update(output=llm_answer)

            
                        final_result = self._format_response(llm_answer, retrieved_docs)
                        return final_result
        

def main():
    parser = argparse.ArgumentParser(description="RAG Engine with Ollama and Faiss")
    
    parser.add_argument("--store_dir", default="./store_small", help="Directory containing Faiss index")
    parser.add_argument("--embedding_model", default="BAAI/bge-m3", help="HuggingFace model for embeddings")
    parser.add_argument("--llm_model", default="qwen3:4b-instruct", help="Ollama model name")
    parser.add_argument("--k", type=int, default=4, help="Number of contexts to retrieve")
    parser.add_argument("--user_id", default="cli-user", help="User ID for tracking")
    
    args = parser.parse_args()
    
    try:
        rag = RAGEngine(
            store_dir=args.store_dir,
            embedding_model_name=args.embedding_model,
            llm_model_name=args.llm_model
        )
    except Exception as e:
        print(f"Startup Error: {e}")
        return

    print("RAG ENGINE READY. Type 'exit' to quit")

    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")

    while True:
        try:
            query = input("\nUser: ")
            if query.lower() in ['exit', 'quit']:
                break
            
            if not query.strip():
                continue

            result = rag.ask(
                query, 
                k=args.k, 
                user_id=args.user_id, 
                session_id=session_id
            )
            
            print(f"\nAssistant:\n{result['answer']}")
            
            if result['sources']:
                print("\nSources:")
                for src in result['sources']:
                    print(f"- [{src['id']}] {src['source']} | {src['section']}")
        except KeyboardInterrupt:
            print("\nExiting")
            break

if __name__ == "__main__":
    main()