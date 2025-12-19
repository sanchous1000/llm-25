from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, CollectionStatus,
    HnswConfigDiff, OptimizersConfigDiff
)

from config import Config
from embeddings import EmbeddingGenerator


class VectorStore:
    def __init__(self, config: Config, embedding_generator: EmbeddingGenerator):
        self.config = config
        self.embedding_generator = embedding_generator
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        if self.config.vector_store.provider == "qdrant":
            self.client = QdrantClient(
                url=self.config.vector_store.url,
                api_key=self.config.vector_store.api_key,
            )
        else:
            raise ValueError(f"Unsupported vector store provider: {self.config.vector_store.provider}")
    
    def create_collection(self, vector_size: int, force_recreate: bool = False):
        collection_name = self.config.vector_store.collection_name
        
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)
        
        if collection_exists:
            if force_recreate or self.config.vector_store.recreate_collection:
                self.client.delete_collection(collection_name)
                collection_exists = False
        
        if not collection_exists:
            hnsw_config = HnswConfigDiff(
                m=self.config.vector_store.hnsw_config.get("m", 16),
                ef_construct=self.config.vector_store.hnsw_config.get("ef_construction", 100),
            )
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
                hnsw_config=hnsw_config,
            )
    
    def upload_chunks(self, chunks_with_embeddings: List[Dict[str, Any]]):
        collection_name = self.config.vector_store.collection_name
        
        if chunks_with_embeddings and "embedding" in chunks_with_embeddings[0]:
            vector_size = len(chunks_with_embeddings[0]["embedding"])
        else:
            raise ValueError("Chunks must have embeddings for dense/hybrid search")
        
        self.create_collection(vector_size)
        
        points = []
        for i, chunk in enumerate(chunks_with_embeddings):
            point_id = i
            vector = chunk.get("embedding", [])
            payload = {
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {}),
                "token_count": chunk.get("token_count", 0),
                "embedding_type": chunk.get("embedding_type", "dense"),
            }
            
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            ))
        
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch
            )
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        collection_name = self.config.vector_store.collection_name
        
        query_result = self.client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
        )
        
        results = []
        for point in query_result.points:
            results.append({
                "text": point.payload.get("text", ""),
                "metadata": point.payload.get("metadata", {}),
                "score": getattr(point, 'score', 0.0),
                "id": point.id,
            })
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        corpus_texts: List[str],
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        dense_results = self.search(query_embedding, top_k=top_k * 2)
        
        sparse_scores = self.embedding_generator.generate_sparse_embeddings(query, corpus_texts)
        
        collection_name = self.config.vector_store.collection_name
        all_points = self.client.scroll(
            collection_name=collection_name,
            limit=10000
        )[0]
        
        combined_scores = {}
        for result in dense_results:
            point_id = result["id"]
            combined_scores[point_id] = {
                "dense_score": result["score"],
                "sparse_score": 0.0,
                "result": result,
            }
        
        for i, point in enumerate(all_points):
            point_id = point.id
            if point_id < len(sparse_scores):
                sparse_score = sparse_scores[point_id]
                if point_id in combined_scores:
                    combined_scores[point_id]["sparse_score"] = sparse_score
                else:
                    combined_scores[point_id] = {
                        "dense_score": 0.0,
                        "sparse_score": sparse_score,
                        "result": {
                            "text": point.payload["text"],
                            "metadata": point.payload["metadata"],
                            "score": 0.0,
                            "id": point_id,
                        }
                    }
        
        max_dense = max(s["dense_score"] for s in combined_scores.values()) if combined_scores else 1.0
        max_sparse = max(s["sparse_score"] for s in combined_scores.values()) if combined_scores else 1.0
        
        final_results = []
        for point_id, scores in combined_scores.items():
            normalized_dense = scores["dense_score"] / max_dense if max_dense > 0 else 0
            normalized_sparse = scores["sparse_score"] / max_sparse if max_sparse > 0 else 0
            
            combined_score = dense_weight * normalized_dense + sparse_weight * normalized_sparse
            result = scores["result"].copy()
            result["score"] = combined_score
            result["dense_score"] = normalized_dense
            result["sparse_score"] = normalized_sparse
            final_results.append(result)
        
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:top_k]
    
    def get_collection_info(self) -> Dict[str, Any]:
        collection_name = self.config.vector_store.collection_name
        info = self.client.get_collection(collection_name)
        
        hnsw_config = None
        if info.config and info.config.params and info.config.params.vectors:
            if hasattr(info.config.params.vectors, 'hnsw_config') and info.config.params.vectors.hnsw_config:
                hnsw_config = {
                    "m": getattr(info.config.params.vectors.hnsw_config, 'm', None),
                    "ef_construct": getattr(info.config.params.vectors.hnsw_config, 'ef_construct', None),
                }
        
        return {
            "name": collection_name,
            "vectors_count": getattr(info, 'points_count', 0),
            "indexed_vectors_count": getattr(info, 'indexed_vectors_count', 0),
            "status": str(info.status) if hasattr(info, 'status') else "unknown",
            "config": {
                "hnsw_config": hnsw_config
            }
        }

