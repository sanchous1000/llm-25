#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

from config import Config
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
import json


def main():
    print("Initializing RAG system...")
    
    try:
        config = Config()
        embedding_generator = EmbeddingGenerator(config)
        vector_store = VectorStore(config, embedding_generator)
        rag = RAGPipeline(config, embedding_generator, vector_store)
        
        print("RAG system ready!")
        print(f"  LLM: {config.rag.llm_provider}/{config.rag.llm_model}")
        print(f"  Embeddings: {config.embeddings.type}/{config.embeddings.model}")
        print(f"  Top-K: {config.rag.top_k}")
        
        if config.rag.llm_provider == "local":
            print(f"  Local LLM URL: {config.rag.base_url or 'http://localhost:8000/v1'}")
            print("  Note: Make sure your local LLM is running!")
        elif config.rag.llm_provider == "openai":
            print("  Using OpenAI API")
        elif config.rag.llm_provider == "anthropic":
            print("  Using Anthropic API")
        
        print("\nType 'quit' or 'exit' to stop\n")
    except ValueError as e:
        print(f"\nConfiguration error: {e}")
        print("\nTo fix this:")
        if "API key" in str(e):
            if "OpenAI" in str(e):
                print("  1. Set OPENAI_API_KEY environment variable:")
                print("     export OPENAI_API_KEY='your-key-here'")
                print("  2. Or use local LLM by setting llm_provider: 'local' in config.yaml")
            elif "Anthropic" in str(e):
                print("  1. Set ANTHROPIC_API_KEY environment variable:")
                print("     export ANTHROPIC_API_KEY='your-key-here'")
        elif "base URL" in str(e):
            print("  1. Set LOCAL_LLM_BASE_URL environment variable:")
            print("     export LOCAL_LLM_BASE_URL='http://localhost:8000/v1'")
            print("  2. Or update config.yaml with the correct base_url")
        print("\nAlternatively, edit source/config.yaml to change llm_provider")
        return
    
    while True:
        try:
            query = input("Вопрос: ").strip()
            
            if query.lower() in ["quit", "exit", "q"]:
                print("До свидания!")
                break
            
            if not query:
                continue
            
            print("\nОбработка запроса...")
            result = rag.answer(query)
            
            print("\n" + "="*60)
            print("ОТВЕТ:")
            print("="*60)
            print(result["answer"])
            print("\n" + "="*60)
            print("ИСТОЧНИКИ:")
            print("="*60)
            for i, citation in enumerate(result["citations"], 1):
                print(f"\n[{i}] Источник: {citation['source']}")
                print(f"    Страница/Раздел: {citation['page']}")
                print(f"    Релевантность: {citation['score']:.4f}")
                print(f"    Сниппет: {citation['snippet']}")
            print("="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nДо свидания!")
            break
        except Exception as e:
            print(f"\nОшибка: {e}\n")


if __name__ == "__main__":
    main()

