# Лаба 3 - Развёртывание Langfuse и логирование LLM/RAG-запросов

В этой лабораторной работе был развёрнут Langfuse (self-hosted) для централизованного логирования запросов к LLM и промежуточных шагов RAG-пайплайна. Далее Langfuse был интегрирован в локальное LLM-приложение (Ollama + RAG из Лабы 2). Также был создан датасет (на основе вопросов по документации FastAPI) и выполнена оценка retrieval-качества через Experiment Run с вычислением метрик Recall@k / Precision@k / MRR.

---

## Используемые компоненты
- **Langfuse** (self-hosted, Docker Compose) - логирование и трассировка.
- **Ollama** (OpenAI-совместимый REST `/v1/chat/completions`) - локальная LLM.
- **RAG-пайплайн** из Лабы 2:
  - эмбеддинги `multilingual-e5-*`
  - FAISS HNSW индекс (`faiss_hnsw.npy`, сериализация FAISS через `serialize_index`)
  - опционально BM25 для hybrid retrieval
- Python библиотеки:
  - `langfuse`, `python-dotenv`
  - `openai` (клиент к Ollama)
  - `sentence-transformers`, `faiss-cpu`, `rank-bm25` (если hybrid)

---

## Структура результата (ключевые файлы)
- `lab3.py` - реализация трассировки Langfuse для "онлайн" RAG-ответа: span’ы retrieval/prompt_build/llm_generate, user_id/session_id.
- `langfuse_eval.py` - запуск Experiment Run на Dataset с вычислением метрик retrieval (Recall/Precision/MRR).
- `questions.json` - набор из 15 вопросов с `gold_paths` (эталонные документы корпуса).

---

## 1) Развёртывание Langfuse (Docker Compose)
### Что делали
1. Запустили Langfuse локально через `docker compose up -d`.
2. Открыли UI в браузере (локальный адрес, порт `3000`).
3. Создали проект в Langfuse и получили ключи:
   - `LANGFUSE_PUBLIC_KEY`
   - `LANGFUSE_SECRET_KEY`

### Настройка окружения
В LLM/RAG-проект добавили `.env`:
```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://localhost:3000
````

Проверка подключения выполнялась через `langfuse.auth_check()` и `langfuse.flush()`.

---

## 2) Интеграция логирования запросов LLM (пункт 2 задания)

### Где реализовано

Файл: `lab3.py`

### Как работает

Логирование сделано через **один trace на один пользовательский запрос**:

* создаётся корневой span `rag_turn`
* сохраняются входные данные (вопрос, конфигурация, модель, параметры генерации)
* далее в trace добавляются вложенные наблюдения (spans/generation)

Также сохраняются `user_id` и `session_id` через `propagate_attributes(...)` - это позволяет анализировать историю запросов по пользователям и сессиям.

---

## 3) Логирование промежуточных шагов RAG (пункт 3 задания)

### Где реализовано

Файл: `lab3.py`

### Какие шаги логируются

Внутри одного запроса создаются вложенные шаги:

1. **retrieval** - поиск релевантных чанков:

   * записывается top-k результатов
   * источники (`doc_path`) и оценки (`score`)
   * длительность шага (latency)
2. **prompt_build** - сборка контекста для LLM:

   * какие источники попали в контекст
   * сколько чанков включили
3. **llm_generate** - генерация ответа:

   * входные сообщения (messages) и гиперпараметры
   * выходной текст
   * (если доступно) usage токенов и время

Это даёт возможность в Langfuse UI раскрыть trace и увидеть, на каких документах и кусках строился ответ.

---

## 4) Датасет и загрузка (пункт 4 задания)

### Что использовали как датасет

Использовали собственный набор вопросов по корпусу документации FastAPI:

* 15 элементов
* каждый элемент содержит:

  * `question` (вопрос пользователя)
  * `gold_paths` (эталонные документы, в которых должен находиться ответ)

Файл: `questions.json`

---

## 5) Experiment Run и оценка retrieval-метрик (пункт 5 задания)

### Где реализовано

Файл: `langfuse_eval.py`

### Как работает

1. `rag_task(item)` принимает dataset item:

   * достаёт вопрос из `item.input`
   * достаёт `gold_paths` (нормализация: `expected_output` может быть dict или list)
2. `rag_inference_returning_sources(question)`:

   * выполняет retrieval (dense/sparse/hybrid по `config.yaml`)
   * вызывает LLM через Ollama (OpenAI-compatible)
   * возвращает:

     * `answer_text`
     * `retrieved` (список источников/чанков)
3. evaluators считают метрики:

   * Recall@5, Precision@5, MRR@5
   * Recall@10, Precision@10, MRR@10
4. запуск эксперимента:

   * `dataset.run_experiment(...)`
   * результаты автоматически отображаются в Langfuse в разделе Dataset Runs / Experiments

### Что фиксировали

* Метрики retrieval на каждом item датасета
* Итоговые агрегаты по прогону
* В metadata эксперимента можно фиксировать конфигурацию retrieval (mode/top_k/chunk_size/run_id), чтобы сравнивать разные варианты.

---

## Итоги

1. Langfuse успешно развёрнут локально и доступен через браузер.
2. В приложение добавлена трассировка:

   * логируются запросы пользователя, параметры модели, ответы
   * логируются промежуточные шаги RAG (retrieval/контекст/генерация)
   * поддерживается user_id/session_id
3. Загружен и использован датасет вопросов по документации.
4. Реализован Experiment Run, где retrieval-метрики считаются автоматически и доступны в Langfuse UI для анализа и сравнения конфигураций.

---

## Как воспроизвести

1. Поднять Langfuse:

```bash
docker compose up -d
```

2. Проверить переменные окружения (`.env`) в проекте.

3. Запустить оценку:

```bash
python langfuse_load_dataset.py
python langfuse_eval.py
```

4. Открыть Langfuse UI и посмотреть:

* Traces (по запросам)
* Datasets -> Runs / Experiments (по метрикам)

