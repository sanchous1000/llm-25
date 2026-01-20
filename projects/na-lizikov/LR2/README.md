# Лабораторная работа №2: Построение RAG-агента по документации
## Описание задания
Разработана система Retrieval-Augmented Generation (RAG) для обработки документации Python. Система включает полный пайплайн:
- Парсинг исходных документов (PDF, DOCX, HTML)
- Разбиение на чанки с различными стратегиями
- Векторизация с использованием dense и sparse эмбеддингов
- Гибридный поиск в Elasticsearch
- Генерация ответов с использованием локальной LLM (Ollama/Mistral)
- Оценка качества retrieval с помощью метрик Recall@k, Precision@k, MRR

## Использованные технологии и модели
**LLM:**
- Mistral (через Ollama) - для генерации ответов

**Фреймворки и библиотеки:**
- *Elasticsearch 8.12+* - гибридный поиск (dense + sparse)
- *Sentence Transformers* - генерация dense эмбеддингов
- *Transformers* - токенизация текста
- *Rank-BM25* - sparse эмбеддинги
- *PDFPlumber* - парсинг PDF документов
- *python-docx* - работа с DOCX файлами
- *BeautifulSoup4* - парсинг HTML документации
- *Ollama API* - взаимодействие с локальной LLM

**Модели эмбеддингов:**
- BAAI/bge-base-en - основная модель для dense эмбеддингов
- Поддержка OpenAI text-embedding-3-large (опционально)

## Результаты работы
### Метрики качества retrieval
На основе 15 оценочных вопросов были получены следующие результаты:

|Метрика    |   k=5     |   k=10    |
|-----------|-----------|-----------|
|Recall@k	|   0.133	|   0.400   |
|Precision@k|	0.027	|   0.040   |
|   MRR     |	0.050	|   0.076   |

## Пример работы RAG-агента
```text
Вопрос: Как работать с кодом и байт-кодом в Python C API?

Ответ:
Для работы со словом кода (code object) в Python C API, используйте PyCompilerFlags.cf_flags. Обратите внимание, что многие флаги CO_FUTURE\_ считаются обязательными в текущих версиях Python и установка их не имеет никакого эффекта. (Источник и путь: python-3.13-docs-html\c-api\code.html)...

Источники:
- curses.html (python-3.13-docs-html\c-api\curses.html)
- complex.html (python-3.13-docs-html\c-api\complex.html)
- code.html (python-3.13-docs-html\c-api\code.html)
```

## Ключевые возможности системы
1. Гибкие стратегии разбиения текста:
    - Recursive splitter (рекурсивное разбиение)
    - Markdown splitter (по заголовкам h1-h3)
    - Hybrid splitter (гибридный подход)

2. Гибридный поиск:
    - Комбинация semantic (dense) и keyword (sparse) поиска
    - Настраиваемые веса для каждого типа поиска

3. Конфигурируемость:
    - Все параметры настройки в config.yaml
    - Поддержка пересборки индекса при изменении параметров
    - Версионирование конфигураций

## Выводы
### Положительные результаты:
1. Гибридный подход показал лучшие результаты: сочетание dense и sparse эмбеддингов улучшило recall с 0.133 до 0.400 при k=10
2. Markdown splitter эффективен для структурированной документации: лучше сохраняет семантические границы
3. Elasticsearch обеспечил стабильный гибридный поиск: поддержка dense_vector и text поиска в одном запросе

### Выявленные проблемы и решения:
- Низкая точность при малых k: Precision@5 составил всего 0.027

    *Решение: увеличение k до 10 улучшило recall, но снизило precision*

- Проблемы с границами чанков: часть релевантной информации терялась на границах

    *Решение: увеличение overlap и chunk_size для непрерывности контекста*

- Ограничения модели эмбеддингов: BAAI/bge-base-en не всегда корректно обрабатывает технические термины

    *Решение: тестирование альтернативных моделей (e5-large-v2, bge-m3)*

- Качество парсинга HTML: некоторые структурные элементы терялись

    *Решение: улучшение парсера для Python документации*

## Инструкция по запуску
**1. Установка зависимостей**
``` bash
pip install -r requirements.txt
```

**2. Настройка окружения**

Создайте структуру директорий:
``` text
data/
├── raw/      # исходные документы (PDF, DOCX, HTML)
├── markdown/ # преобразованные файлы
└── chunks/   # чанки и эмбеддинги
```

**3. Запуск Docker контейнера с Elasticsearch**
``` bash
docker run -d -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "xpack.security.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.12.0
```

**4. Настройка конфигурации**

Отредактируйте config.yaml:
``` yaml
chunk_size: 300
overlap: 50
splitter_type: hybrid
embedding_type: hybrid
embedding_model: BAAI/bge-base-en
```

**5. Пайплайн обработки документов**
``` bash
# 1. Парсинг исходных документов
python scripts/parse_docs.py

# 2. Разбиение на чанки и генерация эмбеддингов
python scripts/build_index.py

# 3. Создание индекса в Elasticsearch
python scripts/create_index.py

# 4. Загрузка данных в Elasticsearch
python scripts/load_to_elasticsearch.py

# 5. Запуск RAG-чата
python scripts/rag_chat.py
```

**6. Оценка качества**
``` bash
python scripts/evaluate.py
```

**7. Дополнительные параметры**

Для принудительной пересборки индекса:
``` bash
python scripts/build_index.py --force-rebuild
```

Для тестирования разных конфигураций:
``` bash
python scripts/build_index.py --chunk-size 500 --overlap 100 --splitter markdown
```

*Примечания*

1. *Для использования OpenAI эмбеддингов укажите API ключ в config.yaml*
2. *Вопросы для оценки должны быть в файле data/questions.json*
3. *Все исходные документы помещаются в data/raw/*

