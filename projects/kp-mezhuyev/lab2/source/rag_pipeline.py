"""
RAG пайплайн для ответов на вопросы по документации.

Этапы:
1. Векторизация запроса
2. Поиск релевантных чанков в Elasticsearch
3. Сборка промпта с контекстом
4. Вызов LLM для генерации ответа
5. Форматирование ответа с цитатами
"""
import argparse
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

from config_utils import load_config
from embeddings import DenseEmbedder, format_text_for_e5
from es_utils import get_es_client


def search_relevant_chunks(
    es_client: Elasticsearch,
    index_name: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Ищет релевантные чанки в Elasticsearch.
    
    Args:
        es_client: Клиент Elasticsearch.
        index_name: Имя индекса.
        query_embedding: Эмбеддинг запроса.
        top_k: Количество релевантных чанков.
    
    Returns:
        Список релевантных чанков с метаданными.
    """
    search_body = {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_embedding},
                },
            }
        },
        "_source": ["text", "source_path", "title", "header", "metadata", "chunk_id"],
        "size": top_k,
    }
    
    response = es_client.search(index=index_name, body=search_body)
    
    chunks = []
    for hit in response["hits"]["hits"]:
        chunks.append({
            "text": hit["_source"]["text"],
            "source_path": hit["_source"].get("source_path", ""),
            "title": hit["_source"].get("title", ""),
            "header": hit["_source"].get("header", ""),
            "score": hit["_score"],
            "metadata": hit["_source"].get("metadata", {}),
            "chunk_id": hit["_source"].get("chunk_id", ""),
        })
    
    return chunks


def build_prompt(
    question: str,
    context_chunks: list[dict[str, Any]],
) -> str:
    """Собирает промпт для LLM с контекстом.
    
    Args:
        question: Вопрос пользователя.
        context_chunks: Релевантные чанки из документации.
    
    Returns:
        Промпт для LLM.
    """
    # Формируем контекст из чанков
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source_info = chunk.get("source_path", "unknown")
        header = chunk.get("header", "")
        if header:
            source_info = f"{source_info}#{header}"
        
        context_parts.append(
            f"[Документ {i}] ({source_info})\n{chunk['text']}\n"
        )
    
    context = "\n".join(context_parts)
    
    # Собираем промпт
    prompt = f"""Ты - помощник, который отвечает на вопросы по документации FastAPI на основе предоставленного контекста.

Контекст из документации:
{context}

Вопрос: {question}

Инструкции:
1. Ответь на вопрос, используя информацию из контекста
2. Если информации недостаточно, скажи об этом
3. Укажи источники информации в формате [Документ N]
4. Используй русский язык для ответа
5. Будь точным и конкретным

Ответ:"""
    
    return prompt


def call_llm(
    prompt: str,
    config: dict[str, Any],
) -> str:
    """Вызывает LLM для генерации ответа.
    
    Args:
        prompt: Промпт для LLM.
        config: Конфигурация с настройками LLM.
    
    Returns:
        Ответ от LLM.
    """
    rag_config = config.get("rag", {})
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url=rag_config.get("llm_base_url", "http://localhost:11434/v1"),
            api_key=rag_config.get("llm_api_key", "ollama"),  # Для Ollama можно любой ключ
        )
        
        response = client.chat.completions.create(
            model=rag_config.get("llm_model", "qwen2.5:3b"),
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=rag_config.get("llm_temperature", 0.3),
            max_tokens=rag_config.get("llm_max_tokens", 512),
        )
        
        return response.choices[0].message.content
    except ImportError:
        raise ImportError(
            "openai package is required. Install it with: pip install openai"
        )
    except Exception as e:
        return f"Ошибка при вызове LLM: {e}"


def format_answer_with_citations(
    answer: str,
    context_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Форматирует ответ с цитатами.
    
    Args:
        answer: Ответ от LLM.
        context_chunks: Использованные чанки.
    
    Returns:
        Отформатированный ответ с метаданными.
    """
    # Извлекаем источники
    sources = []
    for i, chunk in enumerate(context_chunks, 1):
        sources.append({
            "index": i,
            "source_path": chunk.get("source_path", ""),
            "title": chunk.get("title", ""),
            "header": chunk.get("header", ""),
            "snippet": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
            "relevance_score": chunk.get("score", 0.0),
            "chunk_id": chunk.get("chunk_id", ""),
        })
    
    return {
        "answer": answer,
        "sources": sources,
        "num_sources": len(sources),
    }


def rag_query(
    question: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Выполняет RAG запрос.
    
    Args:
        question: Вопрос пользователя.
        config: Конфигурация.
    
    Returns:
        Ответ с цитатами.
    """
    es_config = config.get("elasticsearch", {})
    rag_config = config.get("rag", {})
    embeddings_config = config.get("embeddings", {})
    
    # Подключаемся к Elasticsearch
    es_client, es_url = get_es_client(es_config)
    print(f"Connected to Elasticsearch: {es_url}")
    
    if not es_client.ping():
        raise ConnectionError("Cannot connect to Elasticsearch")
    
    index_name = es_config.get("index_name", "fastapi_docs")
    
    # Создаем эмбеддинг запроса
    dense_config = embeddings_config.get("dense", {})
    model_name = dense_config.get("model", "intfloat/multilingual-e5-base")
    embedder = DenseEmbedder(
        model_name=model_name,
        device=dense_config.get("device", "cpu"),
    )
    
    is_e5_model = "e5" in model_name.lower()
    query_text = format_text_for_e5(question, prefix="query: ") if is_e5_model else question
    query_embedding = embedder.embed([query_text])[0].tolist()
    
    # Ищем релевантные чанки
    top_k = rag_config.get("top_k", 5)
    context_chunks = search_relevant_chunks(es_client, index_name, query_embedding, top_k)
    
    if not context_chunks:
        return {
            "answer": "Не найдено релевантной информации в документации.",
            "sources": [],
            "num_sources": 0,
        }
    
    # Собираем промпт
    prompt = build_prompt(question, context_chunks)
    
    # Вызываем LLM
    answer = call_llm(prompt, config)
    
    # Форматируем ответ с цитатами
    result = format_answer_with_citations(answer, context_chunks)
    
    return result


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(description="RAG pipeline for answering questions")
    parser.add_argument(
        "--query",
        type=str,
        help="Question to answer",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="source/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        help="Override Elasticsearch index name (e.g., fastapi_docs_exp_baseline_recursive_1024)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode",
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    lab2_dir = script_dir.parent
    
    config_path = lab2_dir / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_config(config_path)
    
    # Переопределяем индекс, если указан
    if args.index_name:
        if "elasticsearch" not in config:
            config["elasticsearch"] = {}
        config["elasticsearch"]["index_name"] = args.index_name
        print(f"Используется индекс: {args.index_name}")
    
    if args.interactive:
        print("RAG Pipeline - Interactive Mode")
        print("Введите 'exit' для выхода\n")
        
        while True:
            question = input("Вопрос: ").strip()
            if question.lower() in ["exit", "quit", "выход"]:
                break
            
            if not question:
                continue
            
            try:
                result = rag_query(question, config)
                
                print("\n" + "=" * 60)
                print("ОТВЕТ:")
                print("=" * 60)
                print(result["answer"])
                print("\nИсточники:")
                for source in result["sources"]:
                    print(f"  [{source['index']}] {source['source_path']}")
                    if source["header"]:
                        print(f"      Заголовок: {source['header']}")
                    print(f"      Релевантность: {source['relevance_score']:.4f}")
                print("=" * 60 + "\n")
            except Exception as e:
                print(f"Ошибка: {e}\n")
    elif args.query:
        result = rag_query(args.query, config)
        
        print("\n" + "=" * 60)
        print("ВОПРОС:", args.query)
        print("=" * 60)
        print("\nОТВЕТ:")
        print(result["answer"])
        print("\nИсточники:")
        for source in result["sources"]:
            print(f"  [{source['index']}] {source['source_path']}")
            if source["header"]:
                print(f"      Заголовок: {source['header']}")
        print("=" * 60)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
