import argparse
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from openai import OpenAI

from config import load_config, Config


SYSTEM_PROMPT = """You are a helpful assistant that answers questions about JavaScript based on the "You Don't Know JS" book series.

Rules:
1. Answer only based on the provided context
2. If the context doesn't contain enough information, say so
3. Cite sources using [Book: section] format
4. Be concise but thorough
5. Use code examples when appropriate"""


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[dict]
    model: str


class RAGPipeline:
    def __init__(self, config: Config = None):
        if config is None:
            config = load_config()
        self.config = config
        
        self.embedder = SentenceTransformer(config.embedding_model)
        self.qdrant = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)
        self.llm = OpenAI(base_url=config.llm_base_url, api_key="ollama")
    
    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        if top_k is None:
            top_k = self.config.top_k
        
        query_embedding = self.embedder.encode(query).tolist()
        
        results = self.qdrant.query_points(
            collection_name=self.config.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        return [
            {
                "text": r.payload["text"],
                "book": r.payload["book"],
                "file": r.payload["file"],
                "section": r.payload["section"],
                "score": r.score
            }
            for r in results.points
        ]
    
    def build_context(self, chunks: list[dict]) -> str:
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source = f"[{chunk['book']}: {chunk['section']}]"
            context_parts.append(f"Source {i} {source}:\n{chunk['text']}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def generate_answer(self, question: str, context: str) -> str:
        user_prompt = f"""Context from "You Don't Know JS":

{context}

---

Question: {question}

Answer based on the context above. Cite sources."""
        
        response = self.llm.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.config.llm_temperature
        )
        
        return response.choices[0].message.content
    
    def query(self, question: str, top_k: int = None) -> RAGResponse:
        chunks = self.retrieve(question, top_k)
        context = self.build_context(chunks)
        answer = self.generate_answer(question, context)
        
        return RAGResponse(
            question=question,
            answer=answer,
            sources=[
                {
                    "book": c["book"],
                    "section": c["section"],
                    "score": round(c["score"], 3)
                }
                for c in chunks
            ],
            model=self.config.llm_model
        )
    
    def interactive_mode(self):
        print("\nRAG Pipeline - Interactive Mode")
        print("Type 'quit' to exit\n")
        
        while True:
            question = input("Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            print("\nSearching...")
            response = self.query(question)
            
            print(f"\nAnswer ({response.model}):")
            print(response.answer)
            
            print("\nSources:")
            for src in response.sources:
                print(f"  - [{src['book']}] {src['section']} (score: {src['score']})")
            
            print()


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline for YDKJS")
    parser.add_argument("--question", "-q", type=str, help="Single question to answer")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--model", default="qwen2.5:3b", help="LLM model to use")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    config = load_config(
        top_k=args.top_k,
        llm_model=args.model
    )
    
    pipeline = RAGPipeline(config)
    
    if args.interactive:
        pipeline.interactive_mode()
    elif args.question:
        response = pipeline.query(args.question)
        
        print(f"\nQuestion: {response.question}")
        print(f"\nAnswer ({response.model}):")
        print(response.answer)
        
        print("\nSources:")
        for src in response.sources:
            print(f"  - [{src['book']}] {src['section']} (score: {src['score']})")
    else:
        # demo
        demo_questions = [
            "What is the difference between var and let?",
            "How do closures work in JavaScript?",
            "What is prototypal inheritance?"
        ]
        
        print("Demo mode - answering sample questions:\n")
        
        for q in demo_questions:
            print(f"Q: {q}")
            response = pipeline.query(q)
            print(f"A: {response.answer[:500]}...")
            print(f"Sources: {[s['book'] for s in response.sources]}")
            print("-" * 50)


if __name__ == "__main__":
    main()

