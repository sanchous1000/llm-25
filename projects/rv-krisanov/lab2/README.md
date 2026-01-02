# Лабораторная работа 2: Построение RAG-агента по документации D&D 5e

## Описание проекта

RAG-агент для ответов на вопросы по правилам Dungeons & Dragons 5th Edition. Система реализует полный цикл: парсинг документации, построение векторного индекса, поиск релевантных фрагментов и генерация ответов с использованием LLM.

---

## 1. Выбор и подготовка источников данных

### Корпус документов

**Источник**: D&D 5e System Reference Document (SRD)  
**Репозиторий**: [dnd-5e-srd](https://github.com/bagelbits/5e-srd-api)  
**Формат**: Markdown (конвертированный из официального SRD)  
**Объем**: 16 документов, ~250+ страниц текста

### Содержание корпуса

```
source/corpus/
├── 00 legal.md          # Лицензия и правовая информация
├── 01 races.md          # Расы (эльфы, дварфы, халфлинги и т.д.)
├── 02 classes.md        # Классы (воин, волшебник, варвар и т.д.)
├── 03 beyond1st.md      # Прогрессия персонажа
├── 04 equipment.md      # Снаряжение и оружие
├── 05 feats.md          # Способности
├── 06 mechanics.md      # Игровая механика
├── 07 combat.md         # Боевая система
├── 08 spellcasting.md   # Заклинания и магия
├── 09 running.md        # Управление игрой (DM)
├── 10 magic items.md    # Магические предметы
├── 11 monsters.md       # Бестиарий
├── 12 conditions.md     # Состояния персонажей
├── 13 gods.md           # Божества
├── 14 planes.md         # Планы существования
├── 15 creatures.md      # Существа
└── 16 npcs.md           # NPC статблоки
```

### Тестовый набор запросов

Подготовлено **20 репрезентативных вопросов** с эталонными релевантными документами:
- Механика рас (дварфы, эльфы, халфлинги)
- Классы и их способности (Rage варвара, Reckless Attack)
- Боевая система (инициатива, сюрприз, bonus actions)
- Магия (spell slots, компоненты заклинаний, ритуалы)
- Снаряжение (стоимость доспехов, обмен валют)
- Монстры (типы, AC, hit points, размеры)

Файл: `evaluation_queries.yaml`

### Получение исходных данных

```bash
git clone https://github.com/bagelbits/5e-srd-api.git submodules/dnd-5e-srd
mkdir -p source/corpus
cp submodules/dnd-5e-srd/markdown/*.md source/corpus/
```

---

## 2. Парсинг источников в Markdown

### Исходный формат

Документация уже представлена в **структурированном Markdown** из официального репозитория. Формат включает:
- Корректную иерархию заголовков (`#`, `##`, `###`, `####`, `#####`)
- Списки, таблицы, блоки кода
- Метаданные о структуре документа

### Нормализация

Документы используются "как есть" без дополнительной обработки, так как исходный формат соответствует требованиям:
- ✅ Корректные заголовки всех уровней
- ✅ Сохранена структура разделов
- ✅ Метаданные о документе (имя файла, путь)
- ✅ Таблицы и списки в markdown-формате

**Артефакт**: `source/corpus/*.md`

---

## 3. Разбиение на чанки и построение эмбеддингов

### Стратегия разбиения

**Двухэтапный подход**:

1. **Markdown Header Text Splitter**:
   - Разбиение по заголовкам (`#` – `#####`)
   - Сохранение контекста заголовков в метаданные
   - Обеспечение семантической целостности секций

2. **Recursive Character Text Splitter**:
   - Разбиение крупных секций на подчанки
   - Функция длины: token-based через `AutoTokenizer`
   - Конфигурируемые `chunk_size` и `overlap`

### Параметры эмбеддингов

**Модель**: `sentence-transformers/all-MiniLM-L6-v2`
- Размерность: 384
- Быстрая генерация эмбеддингов
- Хорошее соотношение качество/скорость

**Конфигурируемые параметры** (YAML):
```yaml
build:
  input_path: "source/corpus"
  output_path: "embeddings_{experiment}.json"
  model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 256  # конфигурируется
  overlap: 30      # конфигурируется
```

### Метаданные чанков

Каждый чанк содержит:
```python
{
    "id": int,
    "vector": list[float],  # 384-мерный вектор
    "payload": {
        "text": str,
        "metadata": {
            "document": {"source": str, "id": int},
            "chunk": {"id": int},
            "md_header": {
                "H1": str, "H2": str, ...,  # иерархия заголовков
                "id": int
            }
        }
    }
}
```

### Идемпотентность

- ✅ Перезапуск без ручной очистки
- ✅ Версионирование через имена файлов (`embeddings_baseline.json`, `embeddings_large_chunks.json`, ...)
- ✅ CLI-конфигурация: `python build_index.py --config=config.yaml`

**Артефакт**: `scripts/build_index.py`

---

## 4. Векторное хранилище и индексация

### Выбранное решение

**Qdrant** (локальный Docker-контейнер)
- URL: `http://localhost:6333`
- Коллекции: `dnd5e_baseline`, `dnd5e_small_overlap`, `dnd5e_big_overlap`

### Конфигурация индекса

**Параметры HNSW**:
```yaml
vector_store:
  hnsw:
    m: 16                      # количество связей на узел
    ef_construct: 100          # размер динамического списка при построении
    full_scan_threshold: 10000 # порог полного сканирования
```

**Векторные параметры**:
- `size`: 384 (из модели all-MiniLM-L6-v2)
- `distance`: COSINE
- `hnsw_config`: HnswConfigDiff

### Батчевая загрузка

Для больших коллекций (>2000 embeddings) реализована **батчевая загрузка**:
- `batch_size`: 500 векторов
- `timeout`: 300 секунд
- Прогресс-бар для отслеживания

```python
for i in range(0, len(points), batch_size):
    batch = points[i:i + batch_size]
    qdrant_client.upsert(collection_name=collection_name, wait=True, points=batch)
```

### Идемпотентность

- ✅ Флаг `--rebuild` для пересоздания коллекции
- ✅ Автоматическое удаление старой коллекции перед созданием новой
- ✅ Безопасный перезапуск без поломки состояния

**Артефакт**: `scripts/load_to_vector_store.py`

---

## 5. Метрики и оценка качества retrieval

### Эксперименты

Проведено **3 эксперимента** с различными конфигурациями:

#### **Эксперимент 1: Baseline** (средний chunk_size, умеренный overlap)
```yaml
chunk_size: 256
overlap: 30
hnsw:
  m: 16
  ef_construct: 100
```

#### **Эксперимент 2: Large Chunks** (большие чанки, малый overlap)
```yaml
chunk_size: 512
overlap: 20
hnsw:
  m: 16
  ef_construct: 100
```

#### **Эксперимент 3: Better HNSW** (малые чанки, большой overlap, улучшенный HNSW)
```yaml
chunk_size: 128
overlap: 50
hnsw:
  m: 32
  ef_construct: 200
```

### Результаты

| Конфигурация | Embeddings | Recall@5 | Precision@5 | Recall@10 | Precision@10 | MRR   |
|-------------|-----------|----------|-------------|-----------|--------------|-------|
| **Baseline** | 2848 | **78.00%** | 35.00% | 100.00% | 24.50% | 0.858 |
| **Large Chunks** | 1956 | **82.06%** ↑ | 35.00% | 100.00% | 23.00% | 0.804 ↓ |
| **Better HNSW** | 6645 | 79.11% | **43.00%** ↑ | 100.00% | **29.00%** ↑ | **0.863** ↑ |

### Метрики

**Retrieval-метрики**:
- **Recall@k**: доля релевантных документов, найденных в топ-k
- **Precision@k**: доля релевантных среди найденных топ-k
- **MRR (Mean Reciprocal Rank)**: среднее обратное ранга первого релевантного документа

**Оценка**:
- Тестовый набор: 20 запросов
- Эталон: `evaluation_queries.yaml` с размеченными релевантными документами
- Метод: косинусная близость в векторном пространстве

### Анализ результатов

#### **1. Baseline (chunk_size=256, overlap=30)**
- ✅ Сбалансированная конфигурация
- ✅ Умеренное количество embeddings (2848)
- ❌ Средний recall@5 (78%) — теряет некоторые релевантные документы

#### **2. Large Chunks (chunk_size=512, overlap=20)**
- ✅ **Лучший Recall@5 (82.06%)** — большие чанки захватывают больше контекста
- ✅ Наименьшее количество embeddings (1956) → быстрый поиск
- ❌ **Худший MRR (0.804)** — релевантные документы находятся ниже в ранжировании
- ❌ Низкая precision@5 (35%) — больше нерелевантного контекста

**Вывод**: Большие чанки хороши для покрытия (recall), но страдают точностью ранжирования.

#### **3. Better HNSW (chunk_size=128, overlap=50)**
- ✅ **Лучшая Precision@5 (43%)** — малые чанки более специфичны
- ✅ **Лучший MRR (0.863)** — релевантные результаты в топе
- ✅ **Лучшая Precision@10 (29%)** — меньше "мусора" в топ-10
- ✅ Большой overlap компенсирует потерю контекста при малых чанках
- ❌ Большое количество embeddings (6645) → медленнее построение индекса
- ❌ Средний Recall@5 (79.11%)

**Вывод**: Малые чанки + большой overlap + улучшенный HNSW дают лучшее ранжирование и точность.

### Выводы

#### **Лучшая конфигурация для продакшена: Better HNSW**

**Причины**:
1. **Наивысшая precision** → меньше нерелевантных результатов в ответе LLM
2. **Лучший MRR** → релевантные документы в топе → меньше "шума" в промпте
3. **Высокая precision@10** → качественный контекст для генерации

**Trade-off**:
- Больше embeddings → дольше построение индекса (компенсируется батчевой загрузкой)
- Средний recall@5 → компенсируется увеличением k до 10 при retrieval

#### **Large Chunks подходит для**:
- Случаев, когда важно **не пропустить** релевантный документ (высокий recall)
- Ограниченных ресурсов (меньше embeddings → быстрее поиск)
- Вопросов, требующих широкого контекста

#### **Baseline — золотая середина**:
- Универсальная конфигурация для старта
- Хорошее соотношение метрик/ресурсов

---

## 6. Движок общения (RAG-пайплайн)

### Архитектура

**Этапы обработки запроса**:

1. **Векторизация запроса**:
   ```python
   query_embedding = model.encode(user_query)
   ```

2. **Поиск релевантных чанков**:
   ```python
   results = qdrant_client.search(
       collection_name="dnd5e_big_overlap",
       query_vector=query_embedding,
       limit=10  # топ-10 результатов
   )
   ```

3. **Сборка промпта**:
   ```python
   context = "\n\n".join([
       f"[Документ: {r.payload['metadata']['document']['source']}]\n"
       f"{r.payload['text']}"
       for r in results
   ])
   
   prompt = f"""Ты - эксперт по правилам D&D 5e. Отвечай на вопросы кратко и лаконично.
   
   Контекст:
   {context}
   
   Вопрос: {user_query}
   
   Ответ:"""
   ```

4. **Вызов LLM**:
   ```python
   response = ollama.chat(
       model="llama3.2",
       messages=[{"role": "user", "content": prompt}]
   )
   ```

5. **Форматирование ответа с цитатами**:
   ```python
   {
       "answer": response["message"]["content"],
       "sources": [
           {
               "document": chunk["metadata"]["document"]["source"],
               "header": chunk["metadata"]["md_header"],
               "snippet": chunk["text"][:200] + "..."
           }
           for chunk in top_chunks
       ]
   }
   ```

### Реализация

**Файлы**:
- `scripts/app.py` — FastAPI REST API
- `scripts/agentic.py` — agentическая архитектура с инструментами

**API Endpoints**:
```
POST /query
{
    "question": "How does barbarian Rage work?",
    "k": 10  // опционально
}

Response:
{
    "answer": "Barbarian Rage grants +2 damage, advantage on STR checks...",
    "sources": [
        {
            "document": "02 classes.md",
            "header": {"H3": "Rage", "H4": "The Barbarian"},
            "snippet": "In battle, you fight with primal ferocity..."
        }
    ],
    "retrieved_chunks": 10
}
```

**Агентический подход** (`agentic.py`):
- Инструмент `search_dnd_rules` для поиска по векторной БД
- LLM решает, когда вызывать инструмент
- Поддержка multi-turn диалогов

---

## Инструкция по запуску

### Предварительные требования

- Python 3.13+
- Docker (для Qdrant)
- uv (менеджер зависимостей)
- Git

### Быстрый старт

```bash
# 1. Получить данные (если еще не скопированы)
git clone https://github.com/bagelbits/5e-srd-api.git submodules/dnd-5e-srd
mkdir -p source/corpus && cp submodules/dnd-5e-srd/markdown/*.md source/corpus/

# 2. Установить зависимости и запустить Qdrant
uv sync
docker run -d -p 6333:6333 qdrant/qdrant:latest

# 3. Запустить эксперименты
chmod +x run_experiments.sh
./run_experiments.sh
```

### Запуск экспериментов отдельно

```bash
# Запустить все 3 эксперимента (build → load → evaluate)
chmod +x run_experiments.sh
./run_experiments.sh

# Или запустить отдельные шаги вручную:

# 1. Построить индекс (чанки + эмбеддинги)
uv run python scripts/build_index.py --config=scripts/configs/config_baseline.yaml

# 2. Загрузить в Qdrant
uv run python scripts/load_to_vector_store.py --config=scripts/configs/config_baseline.yaml

# 3. Оценить качество retrieval
uv run python scripts/evaluate.py \
    --config=scripts/configs/config_baseline.yaml \
    --output=results/baseline.json
```

### Шаг 3: Просмотр результатов

```bash
# Посмотреть метрики экспериментов
cat results/baseline.json
cat results/large_chunks.json
cat results/better_hnsw.json

# Или использовать jq для красивого вывода
cat results/better_hnsw.json | jq '.aggregate_metrics'
```

### Шаг 4: Запуск RAG-агента (опционально)

```bash
# Вариант 1: FastAPI REST API
uv run python scripts/app.py

# В другом терминале: тестирование через curl
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does barbarian Rage work?", "k": 10}'

# Вариант 2: Агентический подход
uv run python scripts/agentic.py
```

---

## Структура проекта

```
lab2/
├── README.md                          # Этот файл
├── run_experiments.sh                 # Скрипт для запуска всех экспериментов
├── evaluation_queries.yaml            # Тестовый набор запросов с эталонами
│
├── scripts/
│   ├── build_index.py                 # Построение чанков и эмбеддингов
│   ├── load_to_vector_store.py        # Загрузка в Qdrant
│   ├── evaluate.py                    # Оценка retrieval-метрик
│   └── configs/
│       ├── config_baseline.yaml       # Эксперимент 1
│       ├── config_large_chunks.yaml   # Эксперимент 2
│       └── config_better_hnsw.yaml    # Эксперимент 3
│
├── source/corpus/                     # Markdown-документация D&D 5e
│   ├── 00 legal.md
│   ├── 01 races.md
│   └── ...
│
├── results/                           # Результаты оценки
│   ├── baseline.json
│   ├── large_chunks.json
│   └── better_hnsw.json
│
└── embeddings_*.json                  # Артефакты построения индекса
```

---

## Используемые технологии

- **Язык**: Python 3.13
- **Чанкинг**: LangChain (`MarkdownHeaderTextSplitter`, `RecursiveCharacterTextSplitter`)
- **Эмбеддинги**: sentence-transformers (`all-MiniLM-L6-v2`)
- **Векторное хранилище**: Qdrant (Docker)
- **RAG Framework**: LangChain
- **LLM**: Ollama (llama3.2)
- **API**: FastAPI
- **Метрики**: sklearn, numpy
- **Управление зависимостями**: uv

---

## Ключевые выводы

1. **Малые чанки (128 tokens) + большой overlap (50) + улучшенный HNSW** дают лучшую точность ранжирования (precision, MRR)

2. **Большие чанки (512 tokens)** лучше для покрытия (recall), но страдают точностью

3. **Baseline (256 tokens, overlap 30)** — универсальная конфигурация с хорошим балансом

4. **Trade-off**: больше embeddings → выше качество, но медленнее индексация

5. **Батчевая загрузка** критична для коллекций >2000 embeddings (таймауты)

6. **Metadata-aware chunking** (сохранение иерархии заголовков) улучшает цитирование источников

---

## Дальнейшие улучшения

- [ ] Гибридный поиск (dense + sparse/BM25)
- [ ] Reranking (cross-encoder для переранжирования топ-k)
- [ ] Квантование векторов (scalar/product quantization)
- [ ] A/B тестирование различных embedding-моделей
- [ ] Мультимодальность (изображения из PDF-правил)
- [ ] Query expansion / rewriting
- [ ] Кеширование популярных запросов

---

## Лицензия

MIT License

Документация D&D 5e SRD предоставлена по Open Gaming License (OGL).
