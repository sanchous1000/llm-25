# Лабораторная работа №3 — Развертывание Langfuse для логирования LLM-запросов

## Описание задания

Развернуть Langfuse для централизованного логирования, трассировки и оценки качества RAG системы из Лабораторной работы 2.

**Основные этапы:**
1. Развернуть Langfuse локально через Docker Compose
2. Интегрировать логирование LLM запросов и ответов
3. Логировать полное взаимодействие, включая промежуточные шаги (retrieval, generation)
4. Загрузить датасет в Langfuse для оценки качества
5. Провести эксперименты с вычислением метрик (Recall@k, Precision@k, MRR)
6. Проанализировать результаты в интерфейсе Langfuse

## Использованные технологии и модели

### Инфраструктура
- **Langfuse:** v2.x (Docker Compose: langfuse/langfuse:2, PostgreSQL)
- **RAG система:** Из Lab2 (Elasticsearch + Ollama)
- **LLM:** Ollama qwen2.5:3b через OpenAI-совместимый API
- **Векторное хранилище:** Elasticsearch 9.2.3
- **Эмбеддинги:** intfloat/multilingual-e5-base

### Библиотеки
- langfuse>=2.0.0, < 3.0.0 (Python SDK)
- dotenv (переменные окружения)
- tqdm (прогресс-бары)
- Модули из Lab2: config_utils, es_utils, rag_pipeline, evaluate

### Датасеты
Использованы размеченные данные из Lab2:
- **Baseline:** expected_chunks_4530f7569b81.json (10 вопросов, 371 chunk)
- **Markdown:** expected_chunks_e6233de3342d.json (10 вопросов, 991 chunk)

## Результаты работы

### Сравнение конфигураций через Langfuse

Проведена оценка двух конфигураций RAG системы с полным логированием в Langfuse.

#### Сравнительная таблица метрик (k=5)

| Конфигурация | Recall@5 | Precision@5 | MRR | Чанков |
|-------------|----------|-------------|-----|---------|
| **Baseline (Recursive, 1024)** | **0.177** | **0.220** | **0.350** | 371 |
| Markdown (512) | 0.114 | 0.140 | 0.253 | 991 |
| **Разница** | **-36%** | **-36%** | **-28%** | **+167%** |

#### Детальные метрики

**Baseline (Recursive, 1024):**
- Recall@5: mean=0.177, std=0.252, min=0.0, max=0.833
- Precision@5: mean=0.220, std=0.316, min=0.0, max=1.0
- MRR: mean=0.350, std=0.436, min=0.0, max=1.0

**Markdown (512):**
- Recall@5: mean=0.114, std=0.183, min=0.0, max=0.6
- Precision@5: mean=0.140, std=0.201, min=0.0, max=0.6
- MRR: mean=0.253, std=0.389, min=0.0, max=1.0

### Логирование в Langfuse

Для каждого вопроса создается **Trace** со следующей структурой:

#### Spans (промежуточные шаги):
1. **retrieval** — поиск релевантных документов в Elasticsearch
   - Input: вопрос, top_k
   - Output: retrieved_chunk_ids, sources (с текстом и score)
   - Время выполнения: ~0.2-0.5s

2. **llm_generation** — генерация ответа LLM
   - Input: вопрос, количество чанков контекста
   - Output: answer, answer_length
   - Время выполнения: ~1-11s (зависит от длины ответа)

3. **metrics_calculation** — вычисление метрик качества
   - Input: retrieved_chunk_ids, expected_chunk_ids, k
   - Output: recall@5, precision@5, mrr
   - Время выполнения: <0.1s

#### Scores (метрики):
- recall@5 (NUMERIC)
- precision@5 (NUMERIC)
- mrr (NUMERIC)

#### Metadata:
- dataset_name
- index_name
- model_name
- question_index
- num_expected_chunks

### Примеры ответов LLM

#### Вопрос: "Как работать с callbacks и webhooks?"

**Baseline (Recursive, 1024):**
```
Советую обратить внимание на два основных типа асинхронной работы с API: 
callbacks (обратные вызовы) и webhooks.

**Обратные вызовы**: Это механизм, который позволяет внешнему разработчику 
подключать свои функции к операциям пути вашего приложения...
```
- Метрики: Recall@5=0.60, Precision@5=0.60, MRR=1.0
- Источники: 5 чанков из tutorial/openapi-callbacks.md

**Markdown (512):**
```
Советую обратить внимание на два основных типа асинхронной работы с API: 
callbacks (обратные вызовы) и webhooks...
```
- Метрики: Recall@5=0.60, Precision@5=0.60, MRR=1.0
- Источники: 5 чанков из tutorial/openapi-callbacks.md

**Вывод:** Обе конфигурации показали одинаковые метрики (максимальные) для этого вопроса.

#### Вопрос: "Что такое зависимости в контексте FastAPI?"

**Baseline (Recursive, 1024):**
- Метрики: Recall@5=0.429, Precision@5=0.600, MRR=1.0
- Ответ: Практический с примерами кода
- Источники: global-dependencies, dependencies-in-path-operation-decorators

**Markdown (512):**
- Метрики: Recall@5=0.286, Precision@5=0.400, MRR=0.333
- Ответ: Теоретический, фокус на типах зависимостей
- Источники: sub-dependencies, features, bigger-applications

**Вывод:** Baseline показывает лучшие метрики (+50% Recall@5) и дает более практичные ответы.

## Выводы

### 1. Развертывание и интеграция Langfuse

**Выполнено:**
- Langfuse развернут локально через Docker Compose (версия 2.x)
- Создан проект и получены API ключи
- Настроены переменные окружения (.env)
- Интеграция с RAG системой из Lab2 через импорт модулей

**Преимущества Langfuse:**
- Централизованное логирование всех LLM запросов
- Детальная трассировка промежуточных шагов (spans)
- Автоматическое вычисление и хранение метрик (scores)
- Удобный UI для анализа и сравнения результатов
- Связь traces с датасетами для воспроизводимых экспериментов

### 2. Логирование RAG pipeline

**Что логируется:**
- Входные данные: вопрос пользователя
- Промежуточные шаги:
  - Векторизация запроса (embeddings)
  - Поиск релевантных документов (retrieval)
  - Генерация ответа LLM (llm_generation)
  - Вычисление метрик (metrics_calculation)
- Выходные данные: ответ LLM, chunk_ids, метрики
- Метаданные: dataset_name, index_name, model_name

**Структура трассировки:**
```
Trace (rag_evaluation)
├── Span: retrieval (0.2-0.5s)
├── Span: llm_generation (1-11s)
└── Span: metrics_calculation (<0.1s)
```

### 3. Датасеты и эксперименты

**Подход:**
Использованы размеченные данные из Lab2 (expected_chunks_*.json) вместо создания новой разметки.

**Датасеты в Langfuse:**
- `fastapi_rag_baseline_recursive_1024` — 10 вопросов, chunk_ids для версии 4530f7569b81
- `fastapi_rag_markdown_512` — 10 вопросов, chunk_ids для версии e6233de3342d

**Формат dataset item:**
```json
{
  "input": {"question": "..."},
  "expected_output": {"expected_chunk_ids": [...]},
  "metadata": {"question_index": 1, "num_expected_chunks": 9}
}
```

### 4. Метрики и качество

**Baseline (Recursive, 1024) лучше по всем метрикам:**
- +36% Recall@5 (0.177 vs 0.114)
- +36% Precision@5 (0.220 vs 0.140)
- +28% MRR (0.350 vs 0.253)

**Причины:**
- Больший размер чанка сохраняет контекст
- Меньшая фрагментация документов
- Больший overlap (128 vs 50 токенов)
- Разметка изначально для Baseline, перенос через косинусную близость для Markdown не идеален

**Вариабельность:**
- Высокая std (0.25-0.44) указывает на разное качество retrieval для разных вопросов
- Некоторые вопросы имеют 0 метрики (не найдено релевантных документов в top-5)
- Лучшие результаты: MRR=1.0 (релевантный документ на первой позиции)

### 5. Анализ через Langfuse UI

**Возможности UI:**
- Просмотр всех traces с фильтрацией по датасету/метаданным
- Детальный просмотр spans с input/output каждого шага
- Визуализация timeline выполнения
- Сравнение scores между traces
- Связь с dataset items через Linked Runs

**Экспорт данных:**
- Созданы JSON файлы с ответами LLM для обеих конфигураций
- Сводный файл для сравнения (llm_answers_comparison.json)
- Возможность программного анализа результатов

### 6. Улучшения и рекомендации

**Текущие ограничения:**
- Датасет мал (10 вопросов)
- Разметка изначально для одной конфигурации
- Нет LLM-based метрик (faithfulness, answer_relevancy)

**Возможные улучшения:**
- Увеличить датасет до 50+ вопросов
- Интегрировать RAGAS для оценки качества ответов LLM
- Добавить A/B тестирование разных промптов
- Настроить алерты на низкие метрики
- Использовать feedback пользователей для дообучения

## Структура проекта

```
lab3/
├── source/
│   ├── docker-compose.yml               # Docker для Langfuse + PostgreSQL
│   ├── requirements.txt                 # Зависимости (langfuse, dotenv, tqdm)
│   ├── config.yaml                      # Конфигурация RAG
│   ├── .env                             # Переменные окружения (ключи, хосты)
│   ├── .gitignore                       # Игнорируемые файлы
│   │
│   ├── create_dataset_from_lab2.py      # Создание датасетов из Lab2
│   ├── run_experiments_langfuse.py      # Запуск экспериментов с логированием
│   └── export_llm_answers.py            # Экспорт ответов LLM в JSON
│
├── data/
│   ├── experiments_results.json                    # Сводные метрики
│   ├── llm_answers_baseline_recursive_1024.json   # Ответы Baseline
│   ├── llm_answers_markdown_512.json              # Ответы Markdown
│   └── llm_answers_comparison.json                # Сравнение конфигураций
│
└── README.md                            # Этот файл
```

## Инструкция по запуску

### 1. Установка зависимостей

```bash
cd lab3/source
pip install -r requirements.txt
```

### 2. Запуск Langfuse

```bash
# Запуск Docker контейнеров
docker-compose up -d

# Проверка статуса
docker-compose ps
```

Откройте UI: `http://localhost:3000`
- Создайте аккаунт и проект
- Получите API ключи: Settings → API Keys

### 3. Настройка переменных окружения

Создайте `.env` файл:

```env
# Langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx
LANGFUSE_HOST=http://localhost:3000

# Elasticsearch (из Lab2)
ELASTICSEARCH_HOST=https://localhost:9200
ELASTICSEARCH_USER=elastic
ELASTICSEARCH_PASSWORD=your_password
ELASTICSEARCH_VERIFY_CERTS=false
```

### 4. Создание датасетов из Lab2

```bash
cd lab3
python source/create_dataset_from_lab2.py
```

Создает два датасета в Langfuse:
- `fastapi_rag_baseline_recursive_1024` (10 items)
- `fastapi_rag_markdown_512` (10 items)

### 5. Запуск экспериментов

```bash
python source/run_experiments_langfuse.py
```

Для каждого вопроса выполняется:
1. Retrieval — поиск релевантных документов
2. LLM Generation — генерация ответа
3. Metrics Calculation — вычисление Recall@5, Precision@5, MRR
4. Логирование в Langfuse — создание trace со spans и scores

Результаты сохраняются в `data/experiments_results.json`.

### 6. Экспорт ответов LLM

```bash
python source/export_llm_answers.py
```

Извлекает из Langfuse traces все ответы LLM и сохраняет в JSON файлы для анализа.

### 7. Просмотр результатов

**В Langfuse UI:**
```
http://localhost:3000/datasets
http://localhost:3000/traces
```

Фильтры для traces:
```
Name: rag_evaluation_fastapi_rag_baseline_recursive_1024
metadata.dataset_name: fastapi_rag_baseline_recursive_1024
```

**В JSON файлах:**
- `data/llm_answers_comparison.json` — сравнение ответов обеих конфигураций
- `data/experiments_results.json` — сводные метрики

## Примеры использования

### Создание датасета

```python
from create_dataset_from_lab2 import load_lab2_expected_chunks, create_langfuse_dataset

# Загрузка размеченных данных из Lab2
questions_and_chunks = load_lab2_expected_chunks("4530f7569b81")

# Создание датасета в Langfuse
create_langfuse_dataset(
    dataset_name="fastapi_rag_baseline_recursive_1024",
    questions_and_chunks=questions_and_chunks,
    metadata={"source": "lab2", "splitter_type": "recursive"}
)
```

### Запуск эксперимента

```python
from run_experiments_langfuse import run_experiment

results = run_experiment(
    dataset_name="fastapi_rag_baseline_recursive_1024",
    index_name="fastapi_docs_baseline_recursive_1024",
    config_path=Path("lab2/source/config.yaml"),
    k=5,
)

print(f"Recall@5: {results['recall@5']['mean']:.3f}")
print(f"Precision@5: {results['precision@5']['mean']:.3f}")
print(f"MRR: {results['mrr']['mean']:.3f}")
```

### Экспорт ответов

```python
from export_llm_answers import extract_answers_from_dataset

answers = extract_answers_from_dataset(langfuse, "fastapi_rag_baseline_recursive_1024")

for item in answers:
    print(f"Q: {item['question']}")
    print(f"A: {item['answer'][:100]}...")
    print(f"Recall@5: {item['metrics']['recall@5']}")
```

## Интеграция с Lab2

Lab3 **повторно использует** модули из Lab2 вместо дублирования кода:

```python
# В начале каждого скрипта lab3
import sys
from pathlib import Path

lab3_dir = Path(__file__).parent.parent
lab2_dir = lab3_dir.parent / "lab2"
sys.path.insert(0, str(lab2_dir / "source"))

# Импорт из lab2
from config_utils import load_config
from es_utils import get_es_client
from rag_pipeline import rag_query, build_prompt, call_llm
from evaluate import search_chunks, calculate_recall_at_k, ...
```

**Преимущества:**
- Нет дублирования кода
- Изменения в Lab2 автоматически применяются к Lab3
- Централизованная логика метрик
- Упрощенная поддержка

## Полезные команды

### Docker Compose

```bash
# Запуск
docker-compose up -d

# Остановка
docker-compose down

# Просмотр логов
docker-compose logs langfuse

# Полная очистка (включая данные)
docker-compose down -v
```

### Проверка систем

```bash
# Langfuse
curl http://localhost:3000

# Elasticsearch
curl -k https://localhost:9200 -u elastic:password

# Проверка индексов
cd ../lab2
python source/clean_elasticsearch.py --list
```

### Управление датасетами

```bash
# Создание из Lab2
python source/create_dataset_from_lab2.py

# Просмотр в UI
http://localhost:3000/datasets
```

## Сравнение с Lab2

| Аспект | Lab2 | Lab3 (с Langfuse) |
|--------|------|-------------------|
| **Логирование** | Нет | Полное (traces, spans, scores) |
| **Метрики** | Только Recall, Precision, MRR | То же + визуализация в UI |
| **Эксперименты** | Локальные JSON файлы | Centralized в Langfuse |
| **Анализ** | Ручной (через JSON) | UI с фильтрами и графиками |
| **Воспроизводимость** | Частичная | Полная (dataset items linked) |
| **Мониторинг** | Нет | Да (в реальном времени) |