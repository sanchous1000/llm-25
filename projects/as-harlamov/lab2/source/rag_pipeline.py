import os
from typing import Any, Dict, List

from anthropic import Anthropic
from dotenv import load_dotenv
from langfuse import Langfuse
from openai import OpenAI

from config import Config
from embeddings import EmbeddingGenerator
from vector_store import VectorStore

load_dotenv()


class RAGPipeline:
    def __init__(
        self,
        config: Config,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore,
        langfuse_public_key: str = None,
        langfuse_secret_key: str = None,
        langfuse_host: str = None,
    ):
        self.config = config
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.llm_client = None
        self._initialize_llm()

        langfuse_url = (
            langfuse_host
            or os.getenv("LANGFUSE_BASE_URL")
            or os.getenv("LANGFUSE_HOST")
            or "http://localhost:3000"
        )

        self.langfuse = Langfuse(
            public_key=langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=langfuse_url,
        )

    def _initialize_llm(self):
        import os

        if self.config.rag.llm_provider == "openai":
            api_key = self.config.embeddings.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key is required. Set it in config.yaml or environment variable OPENAI_API_KEY",
                )
            self.llm_client = OpenAI(api_key=api_key, base_url=self.config.rag.base_url)
        elif self.config.rag.llm_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "Anthropic API key is required. Set it in environment variable ANTHROPIC_API_KEY",
                )
            self.llm_client = Anthropic(api_key=api_key)
        elif self.config.rag.llm_provider == "local":
            base_url = self.config.rag.base_url or os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
            if not base_url:
                raise ValueError(
                    "Local LLM base URL is required. Set it in config.yaml or environment variable LOCAL_LLM_BASE_URL",
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
            limit=10000,
        )[0]

        all_points_sorted = sorted(all_points, key=lambda p: p.id)
        corpus_texts = [point.payload.get("text", "") for point in all_points_sorted]
        return corpus_texts

    def _retrieve_context(self, query: str, parent_span=None) -> List[Dict[str, Any]]:
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
                query, query_embedding, corpus_texts, top_k=self.config.rag.top_k,
            )
        else:
            raise ValueError(f"Unsupported embedding type: {self.config.embeddings.type}")

        if parent_span:
            retrieval_metrics = {}
            if results:
                scores = [r.get("score", 0.0) for r in results]
                if scores:
                    retrieval_metrics["avg_retrieval_score"] = sum(scores) / len(scores)
                    retrieval_metrics["max_retrieval_score"] = max(scores)
                    retrieval_metrics["min_retrieval_score"] = min(scores)
                
                dense_scores = [r.get("dense_score", 0.0) for r in results if "dense_score" in r]
                if dense_scores:
                    retrieval_metrics["avg_dense_score"] = sum(dense_scores) / len(dense_scores)
                    retrieval_metrics["max_dense_score"] = max(dense_scores)
                
                sparse_scores = [r.get("sparse_score", 0.0) for r in results if "sparse_score" in r]
                if sparse_scores:
                    retrieval_metrics["avg_sparse_score"] = sum(sparse_scores) / len(sparse_scores)
                    retrieval_metrics["max_sparse_score"] = max(sparse_scores)
            
            with parent_span.start_as_current_span(
                name="retrieve_context",
                input={"query": query, "embedding_type": self.config.embeddings.type, "top_k": self.config.rag.top_k},
                metadata={
                    "embedding_type": self.config.embeddings.type,
                    "top_k": self.config.rag.top_k,
                    **retrieval_metrics,
                },
            ) as span:
                span.update(
                    output={"num_chunks": len(results), "chunks": results},
                    metadata={
                        "num_chunks": len(results),
                        "status": "success",
                        **retrieval_metrics,
                    },
                )

        return results

    def _build_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        context_text = "\n\n".join(
            [
                f"[Страница {chunk['metadata'].get('page', chunk['metadata'].get('section', 'unknown'))}]\n"
                f"{chunk['text']}"
                for chunk in context_chunks
            ],
        )

        prompt = f"""Ты - помощник, который отвечает на вопросы ТОЛЬКО на основе предоставленной документации.

ВАЖНО:
- Используй ТОЛЬКО информацию из предоставленного контекста ниже
- НЕ используй информацию из интернета или свои знания
- Если ответа нет в предоставленном контексте, скажи "Информация не найдена в документации"
- НЕ придумывай факты, которых нет в контексте

Контекст из документации:
{context_text}

Вопрос: {query}

Ответ (используй только информацию из контекста выше):"""

        return prompt

    def _generate_answer(self, prompt: str, parent_span=None) -> str:
        model_params = {
            "model": self.config.rag.llm_model,
            "temperature": self.config.rag.temperature,
            "max_tokens": self.config.rag.max_tokens,
            "provider": self.config.rag.llm_provider,
        }

        result = None
        error_msg = None

        try:
            if self.config.rag.llm_provider in ["openai", "local"]:
                response = self.llm_client.chat.completions.create(
                    model=self.config.rag.llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Ты - помощник, который отвечает на вопросы ТОЛЬКО на основе предоставленной документации. НЕ используй информацию из интернета или свои знания вне документации.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.rag.temperature,
                    max_tokens=self.config.rag.max_tokens,
                )
                result = response.choices[0].message.content
            elif self.config.rag.llm_provider == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.config.rag.llm_model,
                    max_tokens=self.config.rag.max_tokens,
                    temperature=self.config.rag.temperature,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                )
                result = response.content[0].text
            else:
                error_msg = f"Unsupported LLM provider: {self.config.rag.llm_provider}"
        except Exception as e:
            error_msg = f"Ошибка при генерации ответа: {e}"

        if parent_span:
            with parent_span.start_as_current_generation(
                name="generate_answer",
                model=self.config.rag.llm_model,
                model_parameters=model_params,
                input=prompt,
                metadata={"provider": self.config.rag.llm_provider, "model": self.config.rag.llm_model},
            ) as generation:
                if result:
                    generation.update(output=result, metadata={"status": "success"})
                elif error_msg:
                    generation.update(output=error_msg, level="ERROR", metadata={"error": error_msg})

        if error_msg:
            raise ValueError(error_msg)

        return result

    def answer(
        self,
        query: str,
        session_id: str = None,
        user_id: str = None,
        return_trace_id: bool = False,
    ) -> Dict[str, Any]:
        tags = [self.config.rag.llm_provider, self.config.rag.llm_model]
        if self.config.embeddings.type:
            tags.append(self.config.embeddings.type)

        with self.langfuse.start_as_current_span(
            name="rag_answer",
            input={"query": query},
            metadata={
                "query": query,
                "operation": "rag_answer",
                "llm_provider": self.config.rag.llm_provider,
                "llm_model": self.config.rag.llm_model,
                "embedding_type": self.config.embeddings.type,
                "top_k": self.config.rag.top_k,
            },
        ) as span:
            span.update_trace(
                name=f"rag_answer_{self.config.rag.llm_provider}",
                session_id=session_id,
                user_id=user_id,
                tags=tags,
            )

            trace_id = span.trace_id

            try:
                context_chunks = self._retrieve_context(query, parent_span=span)
                prompt = self._build_prompt(query, context_chunks)
                answer = self._generate_answer(prompt, parent_span=span)

                citations = []
                scores = []
                dense_scores = []
                sparse_scores = []
                
                for chunk in context_chunks:
                    score = chunk.get("score", 0.0)
                    scores.append(score)
                    
                    if "dense_score" in chunk:
                        dense_scores.append(chunk.get("dense_score", 0.0))
                    if "sparse_score" in chunk:
                        sparse_scores.append(chunk.get("sparse_score", 0.0))
                    
                    citations.append(
                        {
                            "source": chunk["metadata"].get("source", "unknown"),
                            "page": chunk["metadata"].get("page", chunk["metadata"].get("section", "unknown")),
                            "snippet": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                            "score": score,
                        },
                    )

                vector_db_metrics = {}
                if scores:
                    vector_db_metrics["avg_retrieval_score"] = sum(scores) / len(scores)
                    vector_db_metrics["max_retrieval_score"] = max(scores)
                    vector_db_metrics["min_retrieval_score"] = min(scores)
                    vector_db_metrics["score_std"] = (
                        (sum((s - vector_db_metrics["avg_retrieval_score"]) ** 2 for s in scores) / len(scores)) ** 0.5
                        if len(scores) > 1 else 0.0
                    )
                
                if dense_scores:
                    vector_db_metrics["avg_dense_score"] = sum(dense_scores) / len(dense_scores)
                    vector_db_metrics["max_dense_score"] = max(dense_scores)
                
                if sparse_scores:
                    vector_db_metrics["avg_sparse_score"] = sum(sparse_scores) / len(sparse_scores)
                    vector_db_metrics["max_sparse_score"] = max(sparse_scores)

                result = {
                    "query": query,
                    "answer": answer,
                    "citations": citations,
                    "context_chunks": len(context_chunks),
                    "vector_db_metrics": vector_db_metrics,
                }

                for metric_name, metric_value in vector_db_metrics.items():
                    try:
                        self.langfuse.create_score(
                            name=metric_name,
                            value=float(metric_value),
                            trace_id=trace_id,
                            comment=f"Метрика из векторной базы данных: {metric_name}",
                        )
                    except Exception as e:
                        pass

                span.update(
                    output=result,
                    metadata={
                        "num_citations": len(citations),
                        "num_context_chunks": len(context_chunks),
                        "status": "success",
                        **vector_db_metrics,
                    },
                )

                self.langfuse.flush()

                if return_trace_id:
                    result["trace_id"] = trace_id

                return result
            except Exception as e:
                span.update(
                    output={"error": str(e)},
                    level="ERROR",
                    metadata={"error": str(e)},
                )
                self.langfuse.flush()
                if return_trace_id:
                    raise
                raise
