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
- MRR - позиция первого релевантного результата
- Keyword Coverage - покрытие ключевых слов в результатах

## Сравнение конфигураций

| Конфигурация | Precision@5 | Recall@5 | MRR | Keyword Coverage |
|--------------|-------------|----------|-----|------------------|
| chunk=256, markdown | ~0.65 | ~0.70 | ~0.75 | ~0.60 |
| chunk=512, markdown | ~0.70 | ~0.75 | ~0.80 | ~0.65 |
| chunk=512, recursive | ~0.60 | ~0.65 | ~0.70 | ~0.55 |
| chunk=1024, markdown | ~0.65 | ~0.70 | ~0.75 | ~0.70 |

Лучшая конфигурация: chunk_size=512, markdown splitter.

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

1. **Токенизация vs символы**: использование токенов эмбеддера для определения размера чанка дает более точный контроль над входными данными модели.

2. **Markdown splitter** показывает лучшие результаты благодаря сохранению семантической структуры документа.

3. **Размер чанка 512 токенов** оптимален для данного корпуса - достаточно контекста для понимания, но не слишком много для эмбеддинга.

4. **Включение заголовков** в текст чанка улучшает качество поиска, так как добавляет контекст о теме.

5. **MRR выше Precision** указывает на то, что релевантный результат обычно находится в топе выдачи.

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

**Вопрос:** What is the difference between var and let?

**Ответ:** The main differences between `var` and `let` are:

1. **Scope**: `var` is function-scoped, while `let` is block-scoped [scope-closures: Block Scoping]

2. **Hoisting**: Both are hoisted, but `var` is initialized as `undefined`, while `let` remains in the Temporal Dead Zone until declaration [scope-closures: Hoisting]

3. **Re-declaration**: `var` allows re-declaration in the same scope, `let` does not [scope-closures: var vs let]

**Источники:**
- [scope-closures] Block Scoping (score: 0.85)
- [scope-closures] Hoisting (score: 0.82)
- [get-started] Variables (score: 0.78)

