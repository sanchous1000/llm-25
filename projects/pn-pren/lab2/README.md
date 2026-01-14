# RAG Agent Lab2

## Задание

Построение RAG-агента для документации: парсинг документов в Markdown, разбиение на чанки, векторизация, индексация в Qdrant, оценка метрик retrieval и реализация QA-пайплайна с LLM.

## Архитектура

Парсинг: конвертация DOCX/PDF/PPTX в Markdown
*Эмбеддинги: dense embeddings (sentence-transformers)
Хранилище: Qdrant с HNSW индексом
LLM: Ollama (llama3.2:3b) для генерации ответов

## Запуск

```bash
docker compose up -d
```

```bash
docker exec app python run_pipeline.py
```

Запрос к агенту:
```bash
docker-compose exec app python src/rag_agent.py --query "ваш вопрос"
# или
docker-compose exec app python src/rag_agent.py --interactive
```

## Метрики

**Параметры**: chunk_size=512, overlap=50, splitter=recursive, model=all-MiniLM-L6-v2

Recall@5: 0.26
Recall@10: 0.36
Precision@5: 0.15
Precision@10: 0.11
MRR: 0.40

Метрики рассчитаны на 20 тестовых запросах из `data/eval_queries.json`.