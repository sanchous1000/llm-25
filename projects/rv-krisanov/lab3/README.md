# Lab 3 - Langfuse Integration & RAG Evaluation

## Концепция

### Как это работает

**Langfuse** - это платформа для логирования, трассировки и оценки LLM-приложений.

#### Основные компоненты:

1. **Traces** (трассировки) - полное взаимодействие пользователь->агент
2. **Spans** - отдельные шаги внутри trace (retrieval, LLM call, tool use)
3. **Observations** - события внутри span (промпты, ответы)
4. **Datasets** - наборы тестовых данных (вопрос + эталон)
5. **Experiments** - прогоны оценки на датасете с метриками

#### Флоу для RAG оценки:

```
1. Загрузка датасета в Langfuse
   |-- Dataset Items: {input: query, expected_output: [relevant_doc_ids]}
   
2. Experiment Run через SDK
   |-- Для каждого элемента датасета:
   |   |-- Запуск RAG retrieval
   |   |-- Сравнение с эталоном
   |   |-- Расчет Recall@k, Precision@k, MRR
   |   \-- Логирование метрик как Evaluation
   
3. Просмотр в Langfuse UI
   |-- Сравнение экспериментов
   |-- Метрики по запросам
   \-- Выводы о качестве
```

### Как валидировать

1. **Шаг 1-2**: Проверить traces в Langfuse UI после запроса к RAG
2. **Шаг 3**: Проверить spans - видны ли retrieval step + LLM step
3. **Шаг 4**: Dataset создан, видны items в UI
4. **Шаг 5**: Experiment Run завершен, metrics отображаются в UI

---

## Реализация

### Шаг 1: Настройка Langfuse

Langfuse уже развернут (из lab2), креды в `.env`:

```bash
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=http://localhost:3001
```

### Шаг 2: Интеграция логирования

[OK] Уже сделано в `lab2/scripts/agentic.py`:
- `CallbackHandler` для автоматического логирования LLM calls
- Traces создаются автоматически при вызове agent

### Шаг 3: Логирование промежуточных шагов

[OK] Langfuse автоматически логирует:
- Tool calls (`remember`) как spans
- LLM prompts/responses как observations
- Metadata (модель, токены, время)

### Шаг 4: Загрузка датасета

Используем evaluation_queries.yaml из lab2.

Скрипт: `scripts/upload_dataset.py`

### Шаг 5: Запуск эксперимента

Скрипт: `scripts/run_experiment.py`

Прогоняет RAG на датасете, считает метрики, логирует в Langfuse.

---

## Структура файлов

```
lab3/
|-- README.md                      # Этот файл
|-- .env.example                   # Пример переменных окружения
|-- evaluation_queries.yaml        # Тестовые запросы D&D 5e (из lab2)
|-- scripts/
|   |-- upload_dnd.py              # Загрузка D&D датасета
|   |-- evaluate_dnd.py            # RAG эксперимент с retrieval метриками
|   |-- upload_squad.py            # Загрузка SQuAD датасета
|   |-- evaluate_squad.py          # SQuAD эксперимент
|   |-- configs/
|   |   \-- config_baseline.yaml   # Конфигурация (из lab2)
|   |-- app.py                     # FastAPI сервер (из lab2)
|   \-- agentic.py                 # Агент с tool calling (из lab2)
\-- results/
    \-- experiment_results.json    # Результаты экспериментов
```

---

## Запуск

### Предварительные требования

1. Langfuse должен быть запущен (из lab2):
   ```bash
   docker-compose up -d  # если используется docker
   ```

2. Qdrant должен быть запущен с коллекцией из lab2:
   ```bash
   # Проверить доступность
   curl http://localhost:6333/collections/test_collection
   ```

3. Создать `.env` файл на основе `.env.example`:
   ```bash
   cp .env.example .env
   # Заполнить креды Langfuse, Qdrant, OpenAI
   ```

### 1. Загрузить датасет D&D в Langfuse

```bash
uv run python scripts/upload_dnd.py
```

**Результат:**
- Датасет `dnd5e_evaluation` с 20 queries
- Доступен в UI: `http://localhost:3001/datasets/dnd5e_evaluation`

### 2. Запустить RAG эксперимент

```bash
uv run python scripts/evaluate_dnd.py
```

**Результат:**
- Retrieval + LLM генерация для каждого query
- Метрики: Recall@5/10, Precision@5/10, MRR
- Traces и scores в Langfuse

### Альтернативно: SQuAD датасет

```bash
# Загрузить SQuAD
uv run python scripts/upload_squad.py

# Запустить эксперимент
uv run python scripts/evaluate_squad.py
```

### 3. Проверить результаты в Langfuse UI

Откройте: `http://localhost:3001`

1. **Datasets** -> `dnd5e_rag_evaluation` - список queries
2. **Traces** - отдельные запросы с spans (retrieval)
3. **Scores** - метрики по каждому запросу

### 4. Анализ результатов

```bash
cat results/experiment_results.json
```

Посмотрите на агрегированные метрики:
- `recall@5`, `recall@10` - полнота поиска
- `precision@5`, `precision@10` - точность поиска
- `MRR` - средняя позиция первого релевантного документа

---

## Результаты экспериментов

### Датасет: D&D 5e Evaluation

**Конфигурация:**
- Коллекция: `dnd5e_baseline`
- Embedding модель: `all-MiniLM-L6-v2`
- Chunk size: 256 tokens
- Overlap: 30 tokens
- HNSW: m=16, ef_construct=100
- Retrieval: top-10
- LLM: gpt-5-nano

**Метрики (агрегированные по 20 запросам):**

| Метрика | Значение | Интерпретация |
|---------|----------|---------------|
| **Recall@5** | 1.0000 (100%) | Все релевантные документы найдены в топ-5 |
| **Recall@10** | 1.0000 (100%) | Все релевантные документы найдены в топ-10 |
| **Precision@5** | 0.3500 (35%) | 35% документов в топ-5 релевантны |
| **Precision@10** | 0.2450 (24.5%) | 24.5% документов в топ-10 релевантны |
| **MRR** | 0.8583 | Первый релевантный документ обычно на позиции 1-2 |

### Анализ

**Сильные стороны:**
- Отличный **recall@5/10 = 100%** - система не пропускает релевантные документы
- Высокий **MRR = 0.86** - релевантные документы в топе выдачи
- Все эталонные документы присутствуют в результатах поиска

**Слабые стороны:**
- Низкая **precision** - много нерелевантных документов в выдаче
- 65-75% документов в топ-k не релевантны запросу
- LLM получает много "шума" в контексте

**Выводы:**
1. **Базовая конфигурация работает хорошо для recall** - не теряем важную информацию
2. **Precision можно улучшить** через:
   - Reranking (cross-encoder для переранжирования топ-k)
   - Меньший chunk_size для более специфичных чанков
   - Гибридный поиск (dense + BM25)
3. **Высокий MRR** показывает, что алгоритм ранжирования работает корректно

---

## Метрики

- **Recall@5, Recall@10**: Сколько релевантных чанков нашлось
- **Precision@5, Precision@10**: Сколько найденных чанков релевантны
- **MRR**: Позиция первого релевантного чанка

