from typing import List, Dict, Any, Optional
from openai import OpenAI
from anthropic import Anthropic

from config import Config
from embeddings import EmbeddingGenerator
from vector_store import VectorStore


class RAGPipeline:
    def __init__(self, config: Config, embedding_generator: EmbeddingGenerator, vector_store: VectorStore):
        self.config = config
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.llm_client = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        import os
        
        if self.config.rag.llm_provider == "openai":
            api_key = self.config.embeddings.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key is required. Set it in config.yaml or environment variable OPENAI_API_KEY"
                )
            self.llm_client = OpenAI(api_key=api_key, base_url=self.config.rag.base_url)
        elif self.config.rag.llm_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "Anthropic API key is required. Set it in environment variable ANTHROPIC_API_KEY"
                )
            self.llm_client = Anthropic(api_key=api_key)
        elif self.config.rag.llm_provider == "local":
            base_url = self.config.rag.base_url or os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
            if not base_url:
                raise ValueError(
                    "Local LLM base URL is required. Set it in config.yaml or environment variable LOCAL_LLM_BASE_URL"
                )
            self.llm_client = OpenAI(base_url=base_url, api_key="ollama")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.rag.llm_provider}")
    
    def _vectorize_query(self, query: str) -> List[float]:
        if self.config.embeddings.type in ["dense", "hybrid"]:
            embeddings = self.embedding_generator.generate_dense_embeddings([query])
            return embeddings[0].tolist()
        else:
            raise ValueError("Dense embeddings required for query vectorization")
    
    def _get_corpus_texts(self) -> List[str]:
        collection_name = self.config.vector_store.collection_name
        all_points = self.vector_store.client.scroll(
            collection_name=collection_name,
            limit=10000
        )[0]
        
        all_points_sorted = sorted(all_points, key=lambda p: p.id)
        corpus_texts = [point.payload.get("text", "") for point in all_points_sorted]
        return corpus_texts
    
    def _retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        if self.config.embeddings.type == "dense":
            query_embedding = self._vectorize_query(query)
            results = self.vector_store.search(query_embedding, top_k=self.config.rag.top_k)
        elif self.config.embeddings.type == "sparse":
            query_embedding = self._vectorize_query(query)
            results = self.vector_store.search(query_embedding, top_k=self.config.rag.top_k)
        elif self.config.embeddings.type == "hybrid":
            query_embedding = self._vectorize_query(query)
            corpus_texts = self._get_corpus_texts()
            results = self.vector_store.hybrid_search(
                query, query_embedding, corpus_texts, top_k=self.config.rag.top_k
            )
        else:
            raise ValueError(f"Unsupported embedding type: {self.config.embeddings.type}")
        
        return results
    
    def _build_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        context_text = "\n\n".join([
            f"[Источник: {chunk['metadata'].get('source', 'unknown')}, "
            f"Страница/Раздел: {chunk['metadata'].get('page', chunk['metadata'].get('section', 'unknown'))}]\n"
            f"{chunk['text']}"
            for chunk in context_chunks
        ])
        
        prompt = f"""Ты - помощник, который отвечает на вопросы на основе предоставленной документации.

Используй только информацию из предоставленного контекста для ответа. Если ответа нет в контексте, скажи об этом.

Контекст:
{context_text}

Вопрос: {query}

Ответ (с указанием источников в виде цитат):"""
        
        return prompt
    
    def _generate_answer(self, prompt: str) -> str:
        if self.config.rag.llm_provider in ["openai", "local"]:
            response = self.llm_client.chat.completions.create(
                model=self.config.rag.llm_model,
                messages=[
                    {"role": "system", "content": "Ты - помощник, который отвечает на вопросы на основе предоставленной документации."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.rag.temperature,
                max_tokens=self.config.rag.max_tokens,
            )
            return response.choices[0].message.content
        elif self.config.rag.llm_provider == "anthropic":
            response = self.llm_client.messages.create(
                model=self.config.rag.llm_model,
                max_tokens=self.config.rag.max_tokens,
                temperature=self.config.rag.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.rag.llm_provider}")
    
    def answer(self, query: str) -> Dict[str, Any]:
        context_chunks = self._retrieve_context(query)
        prompt = self._build_prompt(query, context_chunks)
        answer = self._generate_answer(prompt)
        
        citations = []
        for chunk in context_chunks:
            citations.append({
                "source": chunk["metadata"].get("source", "unknown"),
                "page": chunk["metadata"].get("page", chunk["metadata"].get("section", "unknown")),
                "snippet": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "score": chunk.get("score", 0.0),
            })
        
        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "context_chunks": len(context_chunks),
        }

