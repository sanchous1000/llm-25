from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import yaml
import os

# Определяем путь к корню проекта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

CONFIG = yaml.safe_load(open(CONFIG_PATH))
MODEL_NAME = CONFIG["embedding_model"]

# Настройка клиента для совместимости с Elasticsearch 9.x
# Используем headers для указания совместимой версии API
es = Elasticsearch(
    "http://localhost:9200",
    request_timeout=30,
    headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8"}
)
model = SentenceTransformer(MODEL_NAME)

INDEX_NAME = "rag_docs"


def hybrid_search(query, k=5, dense_weight=0.7, sparse_weight=0.3):
    """
    Гибридный поиск: комбинация dense (semantic) и sparse (keyword) поиска
    
    Args:
        query: поисковый запрос
        k: количество результатов
        dense_weight: вес для semantic поиска (0-1)
        sparse_weight: вес для keyword поиска (0-1)
    """
    query_vector = model.encode(query).tolist()

    # Dense search (semantic)
    dense_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }

    # Sparse search (keyword)
    sparse_query = {
        "match": {
            "text": {
                "query": query,
                "boost": 1.0
            }
        }
    }

    # Гибридный поиск с весами
    response = es.search(
        index=INDEX_NAME,
        size=k,
        query={
            "bool": {
                "should": [
                    {
                        "script_score": {
                            "query": dense_query["script_score"]["query"],
                            "script": dense_query["script_score"]["script"],
                            "boost": dense_weight
                        }
                    },
                    {
                        "match": {
                            "text": {
                                "query": query,
                                "boost": sparse_weight
                            }
                        }
                    }
                ]
            }
        }
    )

    return response["hits"]["hits"]
