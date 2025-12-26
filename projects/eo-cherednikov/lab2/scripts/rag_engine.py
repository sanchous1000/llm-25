import argparse
import json
from typing import List, Dict, Any, Optional
import requests

from utils.ollama import embed_query
from utils.qdrant import QdrantCollection


class RAGEngine:
    def __init__(
        self,
        qdrant: QdrantCollection,
        ollama_host: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2",
        top_k: int = 5
    ):
        self.qdrant = qdrant
        self.ollama_host = ollama_host
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.top_k = top_k


    def search_relevant_chunks(self, query: str) -> List[Dict[str, Any]]:
        query_vector = embed_query(query, model=self.embed_model, host=self.ollama_host)
        results = self.qdrant.search(query_vector=query_vector, top_k=self.top_k)
        chunks = []
        for result in results:
            chunk = {
                "text": result.payload.get("text", ""),
                "file_path": result.payload.get("file_path", ""),
                "heading": result.payload.get("heading", ""),
                "score": result.score,
                "metadata": {
                    k: v for k, v in result.payload.items()
                    if k not in ["text", "file_path", "heading"]
                }
            }
            chunks.append(chunk)
        return chunks


    def build_prompt(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        system_instruction: Optional[str] = None
    ) -> str:
        if system_instruction is None:
            system_instruction = """Ты - полезный ассистент, который отвечает на вопросы на основе предоставленного контекста.
Используй только информацию из контекста. Если в контексте нет ответа, скажи об этом.
В конце ответа укажи источники информации."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source_info = f"Источник: {chunk['file_path']}"
            if chunk['heading']:
                source_info += f" | Раздел: {chunk['heading']}"
            context_parts.append(
                f"[Документ {i}]\n{source_info}\n{chunk['text']}\n"
            )
        context = "\n".join(context_parts)
        prompt = f"""{system_instruction}

            Контекст:
            {context}
            
            Вопрос: {question}
            
            Ответ:"""
        return prompt


    def generate_answer(self, prompt: str) -> str:
        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"]


    def ask(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        chunks = self.search_relevant_chunks(question)
        if not chunks:
            return {
                "answer": "Извините, не удалось найти релевантную информацию для вашего вопроса.",
                "sources": [],
                "chunks": []
            }
        prompt = self.build_prompt(question, chunks)
        answer = self.generate_answer(prompt)
        sources = []
        for chunk in chunks:
            source = {
                "file_path": chunk["file_path"],
                "heading": chunk["heading"],
                "snippet": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "relevance_score": chunk["score"]
            }
            sources.append(source)
        result = {
            "answer": answer,
            "sources": sources if include_sources else [],
            "chunks_used": len(chunks)
        }
        return result


def format_response(result: Dict[str, Any]) -> str:
    output = []
    output.append("=" * 60)
    output.append("ОТВЕТ:")
    output.append("=" * 60)
    output.append(result["answer"])
    output.append("")
    if result.get("sources"):
        output.append("=" * 60)
        output.append("ИСТОЧНИКИ:")
        output.append("=" * 60)
        for i, source in enumerate(result["sources"], 1):
            output.append(f"\n[{i}] {source['file_path']}")
            if source['heading']:
                output.append(f"   Раздел: {source['heading']}")
            output.append(f"   Релевантность: {source['relevance_score']:.4f}")
            output.append(f"   Сниппет: {source['snippet']}")
    return "\n".join(output)


def main():
    ap = argparse.ArgumentParser(description="RAG-пайплайн для общения с пользователем")
    ap.add_argument("--ollama_host", default="http://localhost:11434", help="URL Ollama сервера")
    ap.add_argument("--embed_model", default="nomic-embed-text", help="Модель для эмбеддингов")
    ap.add_argument("--llm_model", default="llama3.2", help="Модель LLM для генерации ответов")
    ap.add_argument("--qdrant_host", default="localhost", help="Хост Qdrant")
    ap.add_argument("--qdrant_port", type=int, default=6333, help="Порт Qdrant")
    ap.add_argument("--collection", default="vllm_docs", help="Название коллекции")
    ap.add_argument("--top_k", type=int, default=5, help="Количество релевантных чанков")
    ap.add_argument("--question", help="Вопрос для ответа (если не указан, интерактивный режим)")
    ap.add_argument("--output_json", help="Сохранить ответ в JSON файл")
    args = ap.parse_args()

    qdrant = QdrantCollection(name=args.collection, host=args.qdrant_host, port=args.qdrant_port)
    rag = RAGEngine(
        qdrant=qdrant,
        ollama_host=args.ollama_host,
        embed_model=args.embed_model,
        llm_model=args.llm_model,
        top_k=args.top_k
    )
    if args.question:
        result = rag.ask(args.question)
        print(format_response(result))

        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        print("RAG-пайплайн запущен. Введите вопрос (или 'quit' для выхода):")
        while True:
            try:
                question = input("\n> ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                result = rag.ask(question)
                print(format_response(result))
            except KeyboardInterrupt:
                print("\nВыход...")
                break
            except Exception as e:
                print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()

