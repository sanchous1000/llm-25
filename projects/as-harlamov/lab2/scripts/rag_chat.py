#!/usr/bin/env python3
import sys
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

    while True:
        try:
            query = input("> ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                break

            if not query:
                continue

            result = rag.answer(query)

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

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Ошибка: {e}\n")


if __name__ == "__main__":
    main()
