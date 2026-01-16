import faiss
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import ollama

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
            raise FileNotFoundError(
                "Index files not found. Please run load_to_vector_store.py first."
            )

        self.index = faiss.read_index(str(index_path))
        
        print("Loading metadata from JSONL")
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                self.metadata_map[record['faiss_id']] = record['payload']
        
        print(f"Index loaded successfully. Total records: {self.index.ntotal}")

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
You are a helpful assistant designed to answer questions based on provided text snippets.

TASK:
Use the context below to answer the user's question.

GUIDELINES:
1. Construct a coherent answer from the provided snippets.
2. If the answer is found in the context, provide it and cite the source like [Doc 1].
3. If the context is related to the question but doesn't give a direct answer, summarize the relevant information available.
4. Only say "I could not find information..." if the context is completely irrelevant.

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

    def ask(self, query: str, k: int = 4) -> Dict[str, Any]:
        retrieved_docs = self._retrieve(query, k=k)
        
        if not retrieved_docs:
            return {
                "answer": "I could not find any relevant documents in the knowledge base.",
                "sources": []
            }

        prompt = self._build_prompt(query, retrieved_docs)
        
        try:
            print("Generating LLM response")
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
            print(f"LLM Error: {e}")
            llm_answer = "Error generating answer from LLM"

        final_result = self._format_response(llm_answer, retrieved_docs)
        return final_result

def main():
    parser = argparse.ArgumentParser(description="RAG Engine with Ollama and Faiss")
    parser.add_argument("--store_dir", default="./vector_store", help="Directory containing Faiss index")
    parser.add_argument("--embedding_model", default="BAAI/bge-m3", help="HuggingFace model for embeddings")
    parser.add_argument("--llm_model", default="qwen3:4b-instruct", help="Ollama model name")
    parser.add_argument("--k", type=int, default=4, help="Number of contexts to retrieve")
    
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

    while True:
        try:
            query = input("\nUser: ")
            if query.lower() in ['exit', 'quit']:
                break
            
            if not query.strip():
                continue

            result = rag.ask(query, k=args.k)
            
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