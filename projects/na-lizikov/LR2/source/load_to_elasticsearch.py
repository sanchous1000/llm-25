import json
import os
from elasticsearch import Elasticsearch
from tqdm import tqdm

# Определяем путь к корню проекта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INDEX_NAME = "rag_docs"
es = Elasticsearch("http://localhost:9200")

# Проверяем доступность Elasticsearch
if not es.ping():
    raise RuntimeError("Elasticsearch недоступен. Проверь, что контейнер запущен.")

# Проверяем существование индекса
if not es.indices.exists(index=INDEX_NAME):
    raise RuntimeError(f"Индекс '{INDEX_NAME}' не существует. Сначала запусти create_index.py")

CHUNKS_PATH = os.path.join(PROJECT_ROOT, "data", "chunks", "chunks.json")
print(f"Загрузка чанков в индекс '{INDEX_NAME}'...")
with open(CHUNKS_PATH, encoding="utf-8") as f:
    chunks = json.load(f)

for chunk in tqdm(chunks, desc="Загрузка"):
    doc = {
        "text": chunk["text"],
        "source": chunk.get("source", ""),
        "relative_path": chunk.get("relative_path", ""),
        "chunk_id": chunk.get("chunk_id", chunk.get("id", "")),
        "embedding": chunk["embedding"]
    }
    es.index(index=INDEX_NAME, document=doc, id=chunk.get("chunk_id", chunk.get("id")))

es.indices.refresh(index=INDEX_NAME)
print(f"Документы загружены в Elasticsearch. Всего: {len(chunks)}")
