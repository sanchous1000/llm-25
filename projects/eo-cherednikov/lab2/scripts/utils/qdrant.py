"""Утилиты для работы с Qdrant векторным хранилищем."""
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models


class QdrantCollection:
    """Класс для работы с коллекцией Qdrant."""
    
    def __init__(self, name: str, host: str = "localhost", port: int = 6333):
        """
        Инициализация клиента Qdrant.
        
        Args:
            name: Название коллекции
            host: Хост Qdrant сервера
            port: Порт Qdrant сервера
        """
        self.name = name
        self.client = QdrantClient(host=host, port=port)
    
    def ensure_exists(
        self, 
        vec_size: int, 
        distance: str = "cosine",
        recreate: bool = False,
        hnsw_config: Optional[Dict] = None
    ):
        """
        Создать коллекцию, если она не существует.
        
        Args:
            vec_size: Размерность векторов
            distance: Метрика расстояния (cosine, dot, euclidean)
            recreate: Пересоздать коллекцию, если она существует
            hnsw_config: Конфигурация HNSW (M, ef_construction, ef_search)
        """
        distance_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclidean": Distance.EUCLIDEAN
        }
        
        distance_metric = distance_map.get(distance.lower(), Distance.COSINE)
        
        # Проверяем существование коллекции
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.name for c in collections)
        
        if recreate and collection_exists:
            self.client.delete_collection(self.name)
            collection_exists = False
        
        if not collection_exists:
            # Настройки HNSW по умолчанию
            hnsw_config_dict = hnsw_config or {
                "m": 16,
                "ef_construction": 100,
                "ef_search": 50
            }
            
            vector_params = VectorParams(
                size=vec_size,
                distance=distance_metric,
                hnsw_config=models.HnswConfigDiff(**hnsw_config_dict) if hnsw_config else None
            )
            
            self.client.create_collection(
                collection_name=self.name,
                vectors_config=vector_params
            )
    
    def upload(self, records: List[Dict[str, Any]], vectors: List[List[float]]):
        """
        Загрузить записи с эмбеддингами в коллекцию.
        
        Args:
            records: Список записей с метаданными
            vectors: Список векторов эмбеддингов
        """
        points = []
        for record, vector in zip(records, vectors):
            record_id = record.get("id", "")
            # Преобразуем ID в int для Qdrant (используем hash если строка)
            if isinstance(record_id, str):
                point_id = hash(record_id)
            elif isinstance(record_id, int):
                point_id = record_id
            else:
                point_id = hash(str(record))
            
            payload = {
                "id": str(record_id),  # Сохраняем исходный ID в payload для совместимости
                "text": record.get("text", ""),
                "file_path": record.get("file_path", ""),
                "heading": record.get("heading", ""),
                **record.get("metadata", {})
            }
            
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.name,
            points=points
        )
    
    def search(self, query_vector: List[float], top_k: int = 10) -> List[Any]:
        """
        Поиск похожих векторов.
        
        Args:
            query_vector: Вектор запроса
            top_k: Количество результатов
            
        Returns:
            Список найденных точек
        """
        results = self.client.search(
            collection_name=self.name,
            query_vector=query_vector,
            limit=top_k
        )
        return results
    
    def delete_collection(self):
        """Удалить коллекцию."""
        self.client.delete_collection(self.name)

