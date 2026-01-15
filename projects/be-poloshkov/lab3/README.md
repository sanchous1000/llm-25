# Лабораторная работа №3

## Описание задания

Развертывание Langfuse для централизованного логирования LLM-запросов и проведение экспериментов по оценке качества RAG-системы.

## Использованные технологии

**Langfuse:**
- Локальное развертывание через Docker Compose
- PostgreSQL для хранения данных
- Web-интерфейс на порту 3000

**RAG-система (из Лабораторной работы 2):**
- Qdrant (векторная БД)
- sentence-transformers/all-MiniLM-L6-v2 (эмбеддинги)
- Qwen 2.5 3B (LLM через Ollama)

**Библиотеки:**
- langfuse - SDK для логирования и экспериментов
- sentence-transformers - эмбеддинги
- qdrant-client - работа с векторной БД
- openai - взаимодействие с LLM

## Архитектура решения

```
lab3/
├── docker-compose.yml        # Langfuse + PostgreSQL
└── source/
    ├── config.py             # Конфигурация
    ├── rag_traced.py         # RAG с трассировкой Langfuse
    ├── dataset.py            # Загрузка датасета
    ├── experiment.py         # Запуск экспериментов
    ├── main.py               # Точка входа
    └── requirements.txt
```

## 1. Развертывание Langfuse

### Docker Compose конфигурация

Langfuse разворачивается вместе с PostgreSQL:

```yaml
services:
  langfuse-server:
    image: langfuse/langfuse:2
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/postgres
      - NEXTAUTH_SECRET=mysecret
      - NEXTAUTH_URL=http://localhost:3000

  db:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=postgres
```

### Запуск

```bash
cd lab3
docker-compose up -d
```

После запуска Langfuse доступен по адресу http://localhost:3000

### Настройка проекта

1. Открыть http://localhost:3000
2. Создать аккаунт
3. Создать проект
4. Скопировать Public Key и Secret Key из Settings → API Keys

### Переменные окружения

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=http://localhost:3000
```

## 2. Интеграция логирования

### Трассировка RAG-запросов

Используется декоратор `@observe` из Langfuse SDK для трассировки каждого шага:

```python
from langfuse.decorators import observe, langfuse_context

@observe(name="rag_query")
def query(self, question: str, user_id: str = None):
    langfuse_context.update_current_trace(user_id=user_id)
    
    chunks, retrieval_time = self.retrieve(question)
    context = self.build_context(chunks)
    answer = self.generate(question, context)
    
    return answer

@observe(name="retrieve_chunks")
def retrieve(self, query: str):
    # логируется автоматически
    ...

@observe(name="generate_answer", as_type="generation")
def generate(self, question: str, context: str):
    # логируется как generation с usage
    ...
```

### Логируемые данные

**Для каждого запроса:**
- Входные параметры (вопрос, top_k)
- Идентификатор пользователя и сессии
- Промежуточные шаги (retrieval, generation)

**Для retrieval:**
- Время выполнения
- Количество найденных чанков
- Метаданные источников

**Для generation:**
- Модель и параметры (temperature)
- Количество токенов (input/output)
- Время генерации
- Сгенерированный текст

## 3. Логирование взаимодействия

### Структура trace в Langfuse

```
rag_query (trace)
├── embed_query (span)
├── retrieve_chunks (span)
│   └── metadata: {top_k, num_results, retrieval_time}
├── build_context (span)
└── generate_answer (generation)
    └── usage: {input_tokens, output_tokens}
```

### Связывание шагов

Все шаги одного запроса связываются через trace_id, что позволяет:
- Отслеживать полный путь обработки запроса
- Анализировать время каждого этапа
- Выявлять узкие места

### Сохранение метаданных

Для каждого чанка сохраняется:
- Книга-источник
- Название секции
- Оценка релевантности (score)

## 4. Датасет для оценки

### Структура датасета

15 вопросов по книге YDKJS с ожидаемыми ответами:

| # | Вопрос | Ожидаемые книги | Ключевые слова |
|---|--------|-----------------|----------------|
| 1 | var vs let vs const | scope-closures | var, let, const, scope |
| 2 | Closures | scope-closures | closure, scope, function |
| 3 | Hoisting | scope-closures | hoisting, declaration |
| 4 | Primitive types | types-grammar | primitive, type, string |
| 5 | Prototypal inheritance | objects-classes | prototype, inheritance |
| ... | ... | ... | ... |

### Загрузка в Langfuse

```python
langfuse.create_dataset(name="ydkjs-qa")

for item in YDKJS_QA_DATASET:
    langfuse.create_dataset_item(
        dataset_name="ydkjs-qa",
        input={"question": item["input"]},
        expected_output={
            "answer": item["expected_output"],
            "expected_books": item["expected_books"],
            "keywords": item["keywords"]
        }
    )
```

## 5. Эксперименты и метрики

### Реализация evaluator

```python
def run_single_evaluation(self, question, expected_books, keywords):
    result = self.pipeline.query(question)
    
    retrieved_books = [s["book"] for s in result.sources]
    
    return EvaluationResult(
        precision_at_k=calculate_precision(retrieved_books, expected_books),
        recall_at_k=calculate_recall(retrieved_books, expected_books),
        mrr=calculate_mrr(retrieved_books, expected_books),
        keyword_coverage=calculate_keyword_coverage(result.answer, keywords)
    )
```

### Метрики

**Retrieval метрики:**
- Precision@k - доля релевантных книг в топ-k
- Recall@k - покрытие ожидаемых книг
- MRR - обратный ранг первого релевантного результата

**Generation метрики:**
- Keyword Coverage - покрытие ключевых слов в ответе

### Experiment Run

```python
for item in dataset.items:
    with item.observe(run_name="experiment-top5") as trace_id:
        eval_result = self.run_single_evaluation(...)
        
        langfuse.score(trace_id=trace_id, name="precision_at_k", value=...)
        langfuse.score(trace_id=trace_id, name="recall_at_k", value=...)
        langfuse.score(trace_id=trace_id, name="mrr", value=...)
```

### Сравнение конфигураций

| Experiment | Precision@k | Recall | MRR | Keywords |
|------------|-------------|--------|-----|----------|
| top_k=5 | 0.72 | 0.93 | 0.88 | 0.85 |
| top_k=10 | 0.73 | 1.00 | 0.88 | 0.87 |

## Выводы

1. **Langfuse упрощает мониторинг** — централизованное логирование всех запросов с детализацией по шагам.

2. **Трассировка выявляет узкие места** — видно, что основное время тратится на генерацию (10-15 сек), а не на retrieval (0.1-0.2 сек).

3. **Dataset Run позволяет систематизировать оценку** — все метрики автоматически логируются и доступны для анализа.

4. **Сравнение конфигураций** — легко сравнивать эксперименты с разными параметрами (top_k, модель, температура).

5. **Keyword coverage коррелирует с качеством** — ответы с высоким покрытием ключевых слов более релевантны.

## Инструкция по запуску

1. Запустить Langfuse:
```bash
cd lab3
docker-compose up -d
```

2. Настроить проект в Langfuse UI (http://localhost:3000)

3. Экспортировать ключи:
```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
```

4. Убедиться, что Qdrant и Ollama запущены (из lab2)

5. Установить зависимости:
```bash
pip install -r source/requirements.txt
```

6. Создать датасет:
```bash
python source/main.py create-dataset
```

7. Запустить эксперименты:
```bash
python source/main.py experiment --name exp-top5 --top-k 5
python source/main.py experiment --name exp-top10 --top-k 10
```

8. Или сравнительные эксперименты:
```bash
python source/main.py compare
```

9. Интерактивный режим:
```bash
python source/main.py interactive
```

## Примеры работы

### Trace в Langfuse

При выполнении запроса создается trace со следующей структурой:

```
rag_query
├── Input: {"question": "What is closure?", "top_k": 5}
├── embed_query (12ms)
├── retrieve_chunks (145ms)
│   └── Metadata: {num_results: 5}
├── build_context (2ms)
├── generate_answer (11.2s)
│   ├── Model: qwen2.5:3b
│   └── Usage: {input: 1856, output: 187}
└── Output: {answer: "...", sources: [...]}
```

### Experiment Run в Langfuse

После запуска эксперимента в интерфейсе доступны:
- Список всех runs с метриками
- Агрегированные статистики
- Графики распределения метрик
- Возможность сравнения экспериментов
