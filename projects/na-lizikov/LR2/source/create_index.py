from elasticsearch import Elasticsearch
import yaml
import os

# Определяем путь к корню проекта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

CONFIG = yaml.safe_load(open(CONFIG_PATH))
MODEL_NAME = CONFIG["embedding_model"]

# Автоматически определяем размерность эмбеддингов из модели
from sentence_transformers import SentenceTransformer
temp_model = SentenceTransformer(MODEL_NAME)
EMBEDDING_DIM = temp_model.get_sentence_embedding_dimension()
del temp_model  # Освобождаем память

INDEX_NAME = "rag_docs"

# Подключаемся к Elasticsearch без HTTPS и без security
es = Elasticsearch("http://localhost:9200")

# Проверяем доступность Elasticsearch
if not es.ping():
    raise RuntimeError("Elasticsearch недоступен. Проверь, что контейнер запущен.")

# Если индекс уже есть — удаляем
if es.indices.exists(index=INDEX_NAME):
    es.indices.delete(index=INDEX_NAME)

# Создаём индекс с поддержкой dense embeddings и hybrid search
es.indices.create(
    index=INDEX_NAME,
    mappings={
        "properties": {
            "text": {
                "type": "text",
                "analyzer": "standard"
            },
            "source": {
                "type": "keyword"
            },
            "relative_path": {
                "type": "keyword"
            },
            "chunk_id": {
                "type": "keyword"
            },
            "embedding": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIM,
                "index": True,
                "similarity": "cosine"
            }
        }
    },
    settings={
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }
)

print(f"Индекс '{INDEX_NAME}' успешно создан с поддержкой dense embeddings")
