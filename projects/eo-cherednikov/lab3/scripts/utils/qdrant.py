from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantCollection:
    def __init__(self, name, host, port) -> None:
        self.client = QdrantClient(host=host, port=port)
        self.collection = name
        self.dist_map = {"cosine": Distance.COSINE, "dot": Distance.DOT, "euclidean": Distance.EUCLID}

    def ensure_exists(self, vec_size: int, distance: str, recreate: bool):
        if recreate:
            self.client.delete_collection(self.collection)
        self.client.get_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=vec_size, distance=self.dist_map[distance])
        )

    def search(self, query_vector: List[float], top_k: int):
        result = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k
        )
        return result.points

    def upload(self, records: List[Dict[str, Any]], vectors: List[List[float]], batch: int = 128):
        buf = []
        for i, (rec, vec) in enumerate(zip(records, vectors)):
            payload = {
                "id": rec["id"],
                "heading": rec["heading"],
                "level": rec["level"],
                "text": rec["text"],
                "file_path": rec.get("file_path", "")
            }
            # Добавляем page_number если есть (для PDF файлов)
            if "page_number" in rec:
                payload["page_number"] = rec["page_number"]
            
            buf.append(PointStruct(
                id=i,
                vector=vec,
                payload=payload
            ))
            if len(buf) >= batch:
                self.client.upsert(collection_name=self.collection, points=buf)
                buf = []
        if buf:
            self.client.upsert(collection_name=self.collection, points=buf)