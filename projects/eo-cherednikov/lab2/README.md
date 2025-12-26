# Лабораторная работа №2: RAG-агент по документации

## Описание

Реализация RAG (Retrieval-Augmented Generation) агента для работы с корпусом документов. Система поддерживает парсинг различных форматов документов, построение векторного индекса и генерацию ответов на основе найденной информации.

## Структура проекта

```
lab2/
├── scripts/
│   ├── build_index.py          # Парсинг документов и разбиение на чанки
│   ├── build_embeddings.py     # Построение эмбеддингов и загрузка в Qdrant
│   ├── load_to_vector_store.py  # Управление векторным хранилищем
│   ├── evaluate.py              # Оценка качества retrieval
│   ├── rag_engine.py            # RAG-пайплайн для общения
│   └── utils/
│       ├── __init__.py
│       ├── ollama.py            # Утилиты для работы с Ollama
│       └── qdrant.py            # Утилиты для работы с Qdrant
├── data/
│   ├── raw/                     # Исходные документы
│   ├── md_parsed/               # Распарсенные Markdown файлы
│   └── chunks/                  # Чанки документов
├── configs/                     # Конфигурационные файлы
└── requirements.txt             # Зависимости проекта
```

## Установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Убедитесь, что запущены:
   - **Ollama** сервер (для эмбеддингов и LLM)
   - **Qdrant** сервер (для векторного хранилища)

## Использование

### 1. Подготовка данных

Поместите исходные документы в `data/raw/`. Поддерживаемые форматы:
- Markdown (`.md`)
- PDF (`.pdf`)
- Word (`.docx`, `.doc`)
- PowerPoint (`.pptx`)
- HTML (`.html`, `.htm`)

### 2. Парсинг и разбиение на чанки

```bash
python scripts/build_index.py \
    --raw data/raw \
    --md_out data/md_parsed \
    --chunks_out data/chunks \
    --chunk_size 500 \
    --overlap 100 \
    --splitter recursive \
    --include_headers \
    --rebuild
```

**Параметры:**
- `--raw`: Директория с исходными файлами
- `--md_out`: Директория для Markdown файлов
- `--chunks_out`: Директория для чанков
- `--chunk_size`: Размер чанка в токенах (100-1000)
- `--overlap`: Перекрытие между чанками
- `--splitter`: Стратегия разбиения (`simple`, `markdown`, `recursive`)
- `--include_headers`: Включать заголовки в чанки (для markdown splitter)
- `--rebuild`: Пересобрать индекс

**Стратегии разбиения:**
- `simple`: Простое разбиение по токенам
- `markdown`: Разбиение с учетом заголовков (h1-h3)
- `recursive`: Рекурсивное разбиение с учетом структуры документа

### 3. Построение эмбеддингов

```bash
python scripts/build_embeddings.py \
    --input_jsonl data/chunks/chunks.jsonl \
    --ollama_host http://localhost:11434 \
    --embed_model nomic-embed-text \
    --qdrant_host localhost \
    --qdrant_port 6333 \
    --collection vllm_docs \
    --distance cosine \
    --embedding_type dense \
    --recreate \
    --hnsw_m 16 \
    --hnsw_ef_construction 100 \
    --hnsw_ef_search 50
```

**Параметры:**
- `--input_jsonl`: Путь к JSONL файлу с чанками
- `--embed_model`: Модель для эмбеддингов (Ollama)
- `--collection`: Название коллекции в Qdrant
- `--embedding_type`: Тип эмбеддингов (`dense`, `sparse`, `hybrid`)
- `--recreate`: Пересоздать коллекцию
- `--hnsw_*`: Параметры HNSW индекса

### 4. Управление векторным хранилищем

```bash
python scripts/load_to_vector_store.py \
    --qdrant_host localhost \
    --qdrant_port 6333 \
    --collection vllm_docs \
    --drop-and-reindex \
    --vec_size 768 \
    --distance cosine \
    --hnsw_m 16
```

### 5. Оценка качества

Создайте файл с ground truth данными (`ground_truth.json`):
```json
[
  {
    "question": "Ваш вопрос",
    "relevant_file_paths": ["path/to/relevant/file1", "path/to/relevant/file2"],
    "description": "Описание того, что ищем"
  }
]
```

Запустите оценку:
```bash
python scripts/evaluate.py \
    --ollama_host http://localhost:11434 \
    --embed_model nomic-embed-text \
    --qdrant_host localhost \
    --qdrant_port 6333 \
    --collection vllm_docs \
    --top_k 10 \
    --ground_truth_file ground_truth.json
```

### 6. RAG-пайплайн для общения

**Интерактивный режим:**
```bash
python scripts/rag_engine.py \
    --ollama_host http://localhost:11434 \
    --embed_model nomic-embed-text \
    --llm_model llama3.2 \
    --qdrant_host localhost \
    --qdrant_port 6333 \
    --collection vllm_docs \
    --top_k 5
```

**Одиночный вопрос:**
```bash
python scripts/rag_engine.py \
    --question "Ваш вопрос" \
    --llm_model llama3.2 \
    --output_json answer.json
```

## Конфигурация

Все параметры можно настроить через CLI аргументы. Для удобства можно создать конфигурационные файлы в `configs/` и использовать их через скрипты-обертки.

## Особенности реализации

1. **Парсинг документов:**
   - Поддержка множества форматов (PDF, DOCX, PPTX, HTML, MD)
   - Сохранение метаданных (источник, страница, слайд, дата)
   - Нормализация в Markdown

2. **Разбиение на чанки:**
   - Три стратегии: simple, markdown, recursive
   - Настраиваемый размер и overlap
   - Сохранение структуры документа

3. **Эмбеддинги:**
   - Dense эмбеддинги через Ollama
   - Поддержка sparse (BM25) и hybrid подходов
   - Конфигурируемые параметры

4. **Векторное хранилище:**
   - Qdrant с настраиваемыми HNSW параметрами
   - Поддержка пересборки и переиндексации
   - Идемпотентные операции

5. **RAG-пайплайн:**
   - Векторизация запроса
   - Поиск релевантных чанков
   - Сборка промпта с контекстом
   - Генерация ответа с цитатами

## Метрики качества

Система оценивает качество retrieval по следующим метрикам:
- **Recall@k**: Доля релевантных документов среди найденных
- **Precision@k**: Доля релевантных среди первых k результатов
- **MRR**: Mean Reciprocal Rank

## Примечания

- Убедитесь, что Ollama сервер запущен и модели загружены
- Qdrant должен быть запущен локально или доступен по сети
- Для больших корпусов документов может потребоваться настройка параметров HNSW
- Рекомендуется экспериментировать с различными стратегиями разбиения и параметрами для оптимизации качества

