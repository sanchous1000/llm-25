#!/usr/bin/env python3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

from config import Config
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from rag_pipeline import RAGPipeline


def main():
    try:
        config = Config()
        embedding_generator = EmbeddingGenerator(config)
        vector_store = VectorStore(config, embedding_generator)
        rag = RAGPipeline(config, embedding_generator, vector_store)
    except Exception as e:
        print(f"Ошибка инициализации: {e}")
        return

    session_id = f"chat_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Session ID: {session_id}")
    print("Результаты можно просмотреть в интерфейсе Langfuse по адресу http://localhost:3000")
    print("Введите 'quit', 'exit' или 'q' для выхода\n")

    while True:
        try:
            query = input("> ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                break

            if not query:
                continue

            result = rag.answer(query, session_id=session_id, return_trace_id=True)

            print(f"\n{result['answer']}\n")

            if result["citations"]:
                print("Источники:")
                for i, c in enumerate(result["citations"][:5], 1):
                    page = c["page"]
                    score = c.get("score", 0.0)
                    snippet = c.get("snippet", "")
                    snippet_preview = snippet[:80] + "..." if len(snippet) > 80 else snippet
                    print(f"  {i}. стр. {page} ({score:.2f}): {snippet_preview}")
                print()

            if result.get("trace_id"):
                print(f"Trace ID: {result['trace_id']}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Ошибка: {e}\n")


if __name__ == "__main__":
    main()
