# Лабораторная работа №2

## Описание задания

Построение RAG-агента по документации книги "You Don't Know JS" (2nd Edition). Система включает парсинг документов, разбиение на чанки с учетом количества токенов, построение эмбеддингов, хранение в векторной базе и генерацию ответов с цитированием источников.

## Использованные технологии

**Источник данных:**
- You Don't Know JS (2nd Edition) - https://github.com/getify/You-Dont-Know-JS
- Книги: Get Started, Scope & Closures, Objects & Classes, Types & Grammar

**Эмбеддинги:**
- sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)

**Векторное хранилище:**
- Qdrant (локальный, HNSW индекс)

**LLM:**
- Qwen 2.5 3B (через Ollama)

**Библиотеки:**
- sentence-transformers - эмбеддинги
- transformers - токенизация для подсчета токенов
- qdrant-client - работа с векторной БД
- openai - взаимодействие с LLM

## Архитектура решения

```
source/
├── config.py              # Конфигурация параметров
├── fetch_docs.py          # Загрузка документов с GitHub
├── build_index.py         # Разбиение на чанки и эмбеддинги
├── load_to_vector_store.py # Загрузка в Qdrant
├── evaluate.py            # Оценка качества retrieval
├── rag_pipeline.py        # RAG-пайплайн
├── main.py                # Точка входа
└── requirements.txt
```

## Стратегия разбиения на чанки

Разбиение выполняется на основе количества токенов эмбеддера (не символов).

**Markdown Splitter:**
1. Извлечение секций по заголовкам (h1-h6)
2. Если секция <= chunk_size токенов, сохраняется целиком
3. Если секция > chunk_size, разбивается с overlap
4. Заголовок добавляется в начало каждого чанка

**Параметры:**
- `chunk_size`: размер чанка в токенах (по умолчанию 512)
- `chunk_overlap`: перекрытие в токенах (по умолчанию 50)
- `include_headers_in_chunk`: включать заголовок в текст чанка

## Оценочные вопросы

15 вопросов по книге YDKJS с указанием релевантных книг и ключевых слов:

1. What is the difference between var, let, and const in JavaScript?
2. How does closure work in JavaScript?
3. What is hoisting in JavaScript?
4. What are the different types in JavaScript?
5. How does prototypal inheritance work?
6. What is the this keyword in JavaScript?
7. How do JavaScript classes work under the hood?
8. What is lexical scope?
9. How does type coercion work in JavaScript?
10. What is the difference between == and === in JavaScript?
11. How do arrow functions differ from regular functions?
12. What are iterators and generators in JavaScript?
13. How does the module system work in JavaScript?
14. What is the temporal dead zone?
15. How do you create private properties in JavaScript classes?

## Метрики качества

**Retrieval метрики:**
- Precision@k - доля релевантных документов в топ-k
- Recall@k - покрытие релевантных источников
- MRR (Mean Reciprocal Rank) - обратный ранг первого релевантного результата
- Keyword Coverage - покрытие ключевых слов в результатах

**LLM метрики:**
- Время ответа LLM (секунды)
- Количество входных/выходных токенов

## Результаты экспериментов

### Сравнение конфигураций

| Метрика | chunk=256 (k=5) | chunk=256 (k=10) | chunk=512 (k=5) | chunk=512 (k=10) |
|---------|-----------------|------------------|-----------------|------------------|
| Precision@k | 0.720 | 0.733 | 0.720 | 0.733 |
| Recall@k | 0.933 | 1.000 | 0.933 | 1.000 |
| MRR | 0.883 | 0.883 | 0.883 | 0.883 |
| Keyword Coverage | 0.926 | 0.981 | 0.926 | 0.981 |

### LLM метрики (chunk_size=256, overlap=25)

| Метрика | Значение |
|---------|----------|
| Avg LLM time | 11.64 сек |
| Avg input tokens | 1982 |
| Avg output tokens | 224 |

### Детализация по вопросам (k=5)

| Вопрос | Precision | Recall | MRR | Keywords |
|--------|-----------|--------|-----|----------|
| var vs let vs const | 0.80 | 1.0 | 1.0 | 4/5 |
| Closure | 0.80 | 1.0 | 0.5 | 3/3 |
| Hoisting | 1.00 | 1.0 | 1.0 | 3/3 |
| Types | 0.80 | 1.0 | 0.5 | 5/5 |
| Prototypal inheritance | 0.40 | 1.0 | 1.0 | 3/3 |
| this keyword | 0.40 | 1.0 | 1.0 | 3/3 |
| Classes under the hood | 0.20 | 1.0 | 0.25 | 3/3 |
| Lexical scope | 1.00 | 1.0 | 1.0 | 3/3 |
| Type coercion | 0.80 | 1.0 | 1.0 | 3/3 |
| == vs === | 1.00 | 1.0 | 1.0 | 3/3 |
| Arrow functions | 0.60 | 0.5 | 1.0 | 4/4 |
| Iterators/generators | 0.40 | 0.5 | 1.0 | 1/3 |
| Module system | 1.00 | 1.0 | 1.0 | 2/3 |
| Temporal dead zone | 0.80 | 1.0 | 1.0 | 6/6 |
| Private properties | 0.80 | 1.0 | 1.0 | 4/4 |

### Анализ результатов

**Retrieval метрики:**
- Precision@5 = 0.72 — 72% чанков в топ-5 релевантны
- Recall@10 = 1.0 — система всегда находит нужные книги в топ-10
- MRR = 0.883 — релевантный результат почти всегда в топ-1-2
- Keyword Coverage = 92.6% — хорошее покрытие терминологии

**LLM генерация:**
- Среднее время ответа: 11.64 сек
- Среднее количество входных токенов: 1982 (контекст из 5 чанков)
- Среднее количество выходных токенов: 224

**Проблемные вопросы:**
- "Classes under the hood" — MRR=0.25, первый релевантный на 4-й позиции
- "Iterators/generators" — низкий recall (0.5) и keyword coverage (1/3)

## RAG Pipeline

Этапы обработки запроса:
1. Векторизация вопроса через SentenceTransformer
2. Поиск top-k релевантных чанков в Qdrant
3. Формирование контекста с цитатами источников
4. Генерация ответа через LLM с инструкциями по цитированию
5. Возврат ответа с указанием источников

**Системный промпт:**
```
You are a helpful assistant that answers questions about JavaScript 
based on the "You Don't Know JS" book series.

Rules:
1. Answer only based on the provided context
2. If the context doesn't contain enough information, say so
3. Cite sources using [Book: section] format
4. Be concise but thorough
5. Use code examples when appropriate
```

## Выводы

1. **Высокое качество retrieval**: Precision@5 = 0.72, Recall@10 = 1.0, MRR = 0.88 — система успешно находит релевантные чанки.

2. **Markdown splitter эффективен**: сохранение структуры документа (заголовки, секции) улучшает семантическую связность чанков.

3. **Размер чанка 512 токенов оптимален**: достаточно контекста для понимания, но не слишком много для качественного эмбеддинга.

4. **MRR > Precision**: релевантный результат почти всегда в топ-1-2, даже если остальные результаты менее точны.

5. **Проблемы с редкими темами**: вопросы про iterators/generators и classes показали худшие результаты — требуется больше контекста или специализированные эмбеддинги.

6. **Keyword coverage важен**: 92.6% покрытие ключевых слов подтверждает, что чанки содержат нужную терминологию.

## Инструкция по запуску

1. Установить зависимости:
```bash
pip install -r source/requirements.txt
```

2. Запустить Qdrant:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

3. Запустить Ollama с моделью:
```bash
ollama pull qwen2.5:3b
```

4. Запустить полный пайплайн:
```bash
cd source
python main.py all
```

Или пошагово:
```bash
python main.py fetch                    # загрузка документов
python main.py build --chunk-size 512   # индексация
python main.py load --rebuild           # загрузка в Qdrant
python main.py evaluate                 # оценка метрик
python main.py rag -i                   # интерактивный режим
```

5. Задать вопрос:
```bash
python main.py rag -q "What is closure in JavaScript?"
```

## Примеры работы

### Пример 1: var vs let vs const

**Вопрос:** What is the difference between var, let, and const in JavaScript?

**Ответ LLM:**
> In JavaScript, `var`, `let`, and `const` are used to declare variables.
>
> - **`var`:** Declares a variable with function scope. Can be re-assigned after declaration. Has been the default since the inception of JavaScript.
>
> - **`let`:** Introduced in ES6. Declares a block-scoped variable. Cannot be reassigned once initialized within its scope, but can be re-declared with different values.
>
> - **`const`:** Also introduced in ES6. Similar to `let`, but the value cannot be changed after initialization (immutable). Declares a block-scoped variable.

**Источники:**
- [scope-closures] The Case for `var` (score: 0.719)
- [scope-closures] `var` *and* `let` (score: 0.698)
- [scope-closures] `const`-antly Confused (score: 0.671)

**Метрики:** Precision=0.8, Recall=1.0, MRR=1.0

### Пример 2: Closures

**Вопрос:** How does closure work in JavaScript?

**Ответ LLM:**
> In JavaScript, closure works when a nested function remembers and continues to access variables from its outer (enclosing) function's scope even after the outer function has finished execution. This happens because functions can capture and retain references to their lexical environment at the time they are defined, not just at the point of invocation.

**Источники:**
- [get-started] Closure (score: 0.841)
- [scope-closures] See the Closure (score: 0.818)
- [scope-closures] An Alternative Perspective (score: 0.764)

**Метрики:** Precision=0.8, Recall=1.0, MRR=0.5

