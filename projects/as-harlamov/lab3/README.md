# Лабораторная работа №3 — Langfuse

## Что было сделано

В рамках лабораторной работы №2 была реализована полнофункциональная RAG (Retrieval-Augmented Generation) система для работы с документацией.

В рамках данной работы сделана Интеграция Langfuse для мониторинга и экспериментов:
- Логирование трассировок всех запросов к RAG-пайплайну
- Инструментирование этапов: retrieval, генерация ответа
- Сохранение метрик качества поиска (retrieval scores, dense/sparse scores)
- Интеграция в `RAGPipeline` для автоматического логирования
- Скрипты для создания датасетов и запуска экспериментов

### Вспомогательные инструменты

- `scripts/create_dataset.py` - создание тестовых датасетов в Langfuse
- `scripts/run_experiment.py` - запуск экспериментов через Langfuse Experiment Runs
- `docker-compose.yml` - развертывание Langfuse (web, worker, PostgreSQL, ClickHouse, Redis, MinIO)

## Добавление Langfuse для мониторинга и экспериментов

### Интеграция в RAG-пайплайн

В `RAGPipeline` (`source/rag_pipeline.py`) добавлена полная интеграция с Langfuse:

- **Инициализация клиента**: Langfuse инициализируется в конструкторе `RAGPipeline` с поддержкой параметров через переменные окружения (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`)
- **Трассировка запросов**: Каждый вызов `answer()` создает trace в Langfuse с метаданными (LLM провайдер, модель, тип эмбеддингов, top_k)
- **Инструментирование этапов**:
  - `retrieve_context` - логирование процесса поиска с метриками (avg/max/min retrieval scores, dense/sparse scores)
  - `generate_answer` - логирование генерации ответа через LLM с параметрами модели
- **Метрики качества**: Автоматическое сохранение метрик из векторной базы данных в Langfuse через `create_score()`:
  - `avg_retrieval_score`, `max_retrieval_score`, `min_retrieval_score`
  - `avg_dense_score`, `max_dense_score` (для hybrid search)
  - `avg_sparse_score`, `max_sparse_score` (для hybrid search)
  - `score_std` (стандартное отклонение)

### Создание датасетов

`scripts/create_dataset.py` - скрипт для создания датасетов в Langfuse на основе тестовых вопросов из `data/test_questions.json`:
- Создание датасета через `langfuse.create_dataset()`
- Добавление элементов с input (запрос) и expected_output (релевантные chunk IDs)
- Сохранение метаданных для каждого элемента

### Запуск экспериментов

`scripts/run_experiment.py` - скрипт для проведения экспериментов через Langfuse Experiment Runs:
- Загрузка датасета из Langfuse
- Запуск эксперимента через `langfuse.run_experiment()` с функцией-задачей
- Автоматическая оценка результатов и сохранение метрик
- Сравнение различных конфигураций (разные модели, типы эмбеддингов, параметры)

### Использование

1. Запуск Langfuse: `docker-compose up -d`
2. Создание датасета: `python scripts/create_dataset.py`
3. Интерактивное использование с логированием: `python scripts/rag_chat.py` (результаты видны в Langfuse UI)
4. Запуск эксперимента: `python scripts/run_experiment.py --dataset-name rag_python_qa`

Все трассировки, метрики и результаты экспериментов доступны в веб-интерфейсе Langfuse по адресу `http://localhost:3000`.
