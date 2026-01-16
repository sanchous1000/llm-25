# Langfuse Logging Lab3

## Задание

Развертывание Langfuse для централизованного логирования LLM-запросов: интеграция с RAG-агентом из Lab2, трейсинг всех шагов (retrieval, generation), загрузка датасета и оценка метрик через Experiment Run.

## Запуск

### 1. Запуск RAG системы (lab2)
```bash
cd ../lab2
docker-compose up -d
```

### 2. Запуск Langfuse и RAG приложения (lab3)
```bash
cd ../lab3
docker-compose up -d --build
```

### 3. Настройка API ключей
1. Открыть http://localhost:3000
2. Создать аккаунт и проект
3. Settings → API Keys

### 4. Тестирование подключения
```bash
docker-compose exec app python test_connection.py
```

### 5. Загрузка датасета в Langfuse
```bash
docker-compose exec rag-app python src/dataset_loader.py --load data/eval_queries.json
```

### 6. Запуск эксперимента с оценкой
```bash
docker-compose exec rag-app python src/experiment_runner.py
```

## Метрики эксперимента

Параметры конфигурации:
- chunk_size: 512
- overlap: 50
- splitter: recursive
- embedding_model: all-MiniLM-L6-v2
- llm_model: llama3.2:3b
- top_k: 5

Результаты retrieval (20 тестовых запросов):

Recall@3: 0.23
Precision@3: 0.20
Recall@5: 0.26
Precision@5: 0.15
Recall@10: 0.36
Precision@10: 0.11
MRR: 0.41

Метрики рассчитаны на датасете из `data/eval_queries.json`.  
Все эксперименты логируются в Langfuse и доступны для анализа в UI.
