import time
from dataclasses import dataclass
from typing import Optional

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from openai import OpenAI

from config import Config, load_config


SYSTEM_PROMPT = """You are a helpful assistant that answers questions about JavaScript based on the "You Don't Know JS" book series.

Rules:
1. Answer only based on the provided context
2. If the context doesn't contain enough information, say so
3. Cite sources using [Book: section] format
4. Be concise but thorough"""


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[dict]
    retrieval_time: float
    generation_time: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class TracedRAGPipeline:
    """RAG Pipeline without @observe decorators - tracing handled by experiment runner"""
    
    def __init__(self, config: Config = None):
        if config is None:
            config = load_config()
        self.config = config
        
        self.embedder = SentenceTransformer(config.embedding_model)
        self.qdrant = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)
        self.llm = OpenAI(base_url=config.llm_base_url, api_key="ollama")
    
    def embed_query(self, query: str) -> list[float]:
        return self.embedder.encode(query).tolist()
    
    def retrieve(self, query: str, top_k: int = None) -> tuple[list[dict], float]:
        if top_k is None:
            top_k = self.config.top_k
        
        start = time.time()
        query_embedding = self.embed_query(query)
        
        results = self.qdrant.query_points(
            collection_name=self.config.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        elapsed = time.time() - start
        
        chunks = [
            {
                "text": r.payload["text"],
                "book": r.payload["book"],
                "file": r.payload["file"],
                "section": r.payload["section"],
                "score": r.score
            }
            for r in results.points
        ]
        
        return chunks, elapsed
    
    def build_context(self, chunks: list[dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = f"[{chunk['book']}: {chunk['section']}]"
            context_parts.append(f"Source {i} {source}:\n{chunk['text']}")
        return "\n\n---\n\n".join(context_parts)
    
    def generate(self, question: str, context: str) -> tuple[str, float, dict]:
        user_prompt = f"""Context from "You Don't Know JS":

{context}

---

Question: {question}

Answer based on the context above. Cite sources."""
        
        start = time.time()
        response = self.llm.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.config.llm_temperature
        )
        elapsed = time.time() - start
        
        usage = response.usage
        answer = response.choices[0].message.content
        
        return answer, elapsed, {
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None
        }
    
    def query(self, question: str, user_id: str = None, session_id: str = None, top_k: int = None) -> RAGResult:
        chunks, retrieval_time = self.retrieve(question, top_k)
        context = self.build_context(chunks)
        answer, generation_time, usage = self.generate(question, context)
        
        result = RAGResult(
            question=question,
            answer=answer,
            sources=[
                {"book": c["book"], "section": c["section"], "score": round(c["score"], 3)}
                for c in chunks
            ],
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens")
        )
        
        return result


def demo():
    config = load_config()
    pipeline = TracedRAGPipeline(config)
    
    questions = [
        "What is the difference between var and let?",
        "How do closures work in JavaScript?",
        "What is prototypal inheritance?"
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        result = pipeline.query(q, user_id="demo-user", session_id="demo-session")
        print(f"A: {result.answer[:200]}...")
        print(f"Sources: {[s['book'] for s in result.sources]}")
        print(f"Times: retrieval={result.retrieval_time:.2f}s, generation={result.generation_time:.2f}s")
    
    print("\nDemo completed")


if __name__ == "__main__":
    demo()
