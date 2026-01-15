# Лабораторная работа №2: RAG-агент по документации

## Описание

Реализация RAG (Retrieval-Augmented Generation) агента для работы с корпусом документов. Система поддерживает парсинг различных форматов документов, построение векторного индекса и генерацию ответов на основе найденной информации.

## Структура проекта

```
lab2/
├── scripts/
│   ├── build_index.py          # Парсинг документов и разбиение на чанки
│   ├── build_embeddings.py     # Построение эмбеддингов и загрузка в Qdrant
│   ├── test_vector_store.py    # Тестирование векторного хранилища
│   ├── evaluate.py              # Оценка качества retrieval
│   ├── rag_engine.py            # RAG-пайплайн для общения
│   └── utils/
│       ├── __init__.py
│       ├── ollama.py            # Утилиты для работы с Ollama
│       └── qdrant.py            # Утилиты для работы с Qdrant
├── data/
│   ├── raw/                     # Исходные документы
│   │   └── docs/                # Markdown файлы документации
│   ├── chunks/                  # Чанки документов (JSONL)
│   └── ground_truth.json        # Ground truth данные для оценки
├── configs/                     # Конфигурационные файлы
└── README.md                    # Этот файл
```

## Использованные технологии

### LLM и модели эмбеддингов:
- **nomic-embed-text** - модель для создания эмбеддингов текстов (через Ollama)
- **qwen3:0.6b** - языковая модель для генерации ответов (через Ollama)

### Фреймворки и библиотеки:
- **Ollama** - локальный сервер для запуска LLM и моделей эмбеддингов
- **Qdrant** - векторная база данных для хранения и поиска эмбеддингов
- **qdrant-client** - Python-клиент для работы с Qdrant
- **langchain-text-splitters** - библиотека для разбиения текста на чанки с учетом структуры Markdown
- **numpy**, **tqdm**, **requests** - вспомогательные библиотеки

## Архитектура системы

Система реализует классический пайплайн RAG:

1. **Подготовка данных**: Markdown-файлы разбиваются на чанки с учетом иерархии заголовков (H1-H3). Каждый чанк содержит метаданные: путь к файлу, заголовок секции, уровень вложенности.

2. **Векторизация**: Текстовые чанки преобразуются в эмбеддинги с помощью модели `nomic-embed-text` и сохраняются в Qdrant с метаданными.

3. **Retrieval**: При запросе создается эмбеддинг запроса, выполняется поиск по косинусному расстоянию, возвращаются top-k наиболее релевантных документов.

4. **Generation**: Найденные документы используются как контекст для LLM, которая генерирует ответ на основе предоставленной информации.

### Особенности реализации

1. **Умное разбиение на чанки**: используется двухуровневое разбиение:
   - Сначала по заголовкам (MarkdownHeaderTextSplitter)
   - Затем на чанки фиксированного размера с перекрытием (MarkdownTextSplitter)

2. **Сохранение структуры**: каждый чанк содержит информацию о пути в иерархии документа (например, "Getting Started/Installation/GPU/CUDA").

3. **Нормализация путей**: пути файлов нормализуются (используются прямые слеши) для консистентности между операционными системами.

4. **Оценка качества**: метрики вычисляются на уровне файлов, а не отдельных чанков, что обеспечивает корректную оценку retrieval-системы.

## Инструкция по запуску

### Предварительные требования

1. Установить и запустить **Ollama**:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve

ollama pull nomic-embed-text
ollama pull qwen3:0.6b
```

2. Установить и запустить **Qdrant**:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

3. Установить зависимости Python:
```bash
pip install langchain-text-splitters qdrant-client requests numpy tqdm
```

### Пошаговая инструкция

#### Шаг 1: парсинг Markdown-файлов и преобразование документации в структурированный датасет:
```bash
python scripts/build_index.py
  --input_dir data/raw/docs/
  --output_jsonl data/chunks/chunks.jsonl
  --levels 1,2,3
  --chunk_size 512
  --overlap 64
```


#### Шаг 2: построение эмбеддингов и загрузка в Qdrant
```bash
python scripts/build_embeddings.py
  --input_jsonl data/chunks/chunks.jsonl
  --ollama_host http://localhost:11434
  --embed_model nomic-embed-text
  --qdrant_host localhost
  --qdrant_port 6333
  --collection vllm_docs
  --distance cosine
  --recreate
```

#### Шаг 3: тестирование поиска
```bash
python scripts/test_vector_store.py
  --query "How to install vLLM?"
  --ollama_host http://localhost:11434
  --embed_model nomic-embed-text
  --qdrant_host localhost
  --qdrant_port 6333
  --collection vllm_docs
  --top_k 5
```

**Результат**: выводятся top-5 наиболее релевантных документов с оценками схожести.

#### Шаг 4: RAG-запрос к LLM - генерация ответа на вопрос с использованием найденного контекста:

```bash
python scripts/rag_engine.py
  --question "How to use multiple GPUs to inference a model?"
  --ollama_host http://localhost:11434
  --embed_model nomic-embed-text
  --llm_model qwen3:0.6b
  --qdrant_host localhost
  --qdrant_port 6333
  --collection vllm_docs
  --top_k 5
```

**Результат**: LLM генерирует ответ на основе найденных документов и выводит список использованных источников.

#### Шаг 5: оценка качества retrieval
```bash
python scripts/evaluate.py
  --ground_truth_file data/ground_truth.json
  --ollama_host http://localhost:11434
  --embed_model nomic-embed-text
  --qdrant_host localhost
  --qdrant_port 6333
  --collection vllm_docs
  --top_k 10
```

**Результат**: выводятся средние значения метрик - Mean Recall@5, Recall@10; Mean Precision@5, Precision@10; Mean MRR