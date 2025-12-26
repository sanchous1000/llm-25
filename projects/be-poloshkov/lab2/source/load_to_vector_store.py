import argparse
import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    HnswConfigDiff,
)

from config import load_config, Config


def get_latest_index(config: Config) -> tuple[list[dict], np.ndarray, dict]:
    index_dir = Path(config.index_dir)
    processed_dir = Path(config.processed_dir)
    
    meta_files = list(index_dir.glob("meta_*.json"))
    if not meta_files:
        raise FileNotFoundError("No index found. Run build_index.py first.")
    
    latest_meta = max(meta_files, key=lambda f: f.stat().st_mtime)
    config_hash = latest_meta.stem.replace("meta_", "")
    
    with open(latest_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    chunks_file = processed_dir / f"chunks_{config_hash}.json"
    embeddings_file = index_dir / f"embeddings_{config_hash}.npy"
    
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    embeddings = np.load(embeddings_file)
    
    return chunks, embeddings, meta


def create_collection(client: QdrantClient, config: Config, embedding_dim: int, recreate: bool = False):
    collections = [c.name for c in client.get_collections().collections]
    
    if config.collection_name in collections:
        if recreate:
            print(f"Dropping existing collection: {config.collection_name}")
            client.delete_collection(config.collection_name)
        else:
            print(f"Collection {config.collection_name} already exists")
            return False
    
    print(f"Creating collection: {config.collection_name}")
    client.create_collection(
        collection_name=config.collection_name,
        vectors_config=VectorParams(
            size=embedding_dim,
            distance=Distance.COSINE,
            hnsw_config=HnswConfigDiff(
                m=config.hnsw_m,
                ef_construct=config.hnsw_ef_construct,
            )
        )
    )
    return True


def upload_chunks(client: QdrantClient, config: Config, chunks: list[dict], embeddings: np.ndarray):
    points = []
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point = PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={
                "chunk_id": chunk["id"],
                "text": chunk["text"],
                "book": chunk["book"],
                "file": chunk["file"],
                "section": chunk["section"],
                "token_count": chunk["token_count"],
            }
        )
        points.append(point)
    
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=config.collection_name,
            points=batch
        )
        print(f"  Uploaded {min(i + batch_size, len(points))}/{len(points)} points")
    
    print(f"Total uploaded: {len(points)} points")


def load_to_vector_store(config: Config = None, rebuild: bool = False):
    if config is None:
        config = load_config()
    
    chunks, embeddings, meta = get_latest_index(config)
    
    print(f"Loaded index: {len(chunks)} chunks, embedding dim: {meta['embedding_dim']}")
    
    client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)
    
    created = create_collection(client, config, meta["embedding_dim"], recreate=rebuild)
    
    if created or rebuild:
        upload_chunks(client, config, chunks, embeddings)
    
    info = client.get_collection(config.collection_name)
    print(f"\nCollection info:")
    print(f"  Points: {info.points_count}")
    
    return client


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load index to vector store")
    parser.add_argument("--rebuild", action="store_true", help="Drop and recreate collection")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    
    args = parser.parse_args()
    
    config = load_config(
        qdrant_host=args.host,
        qdrant_port=args.port
    )
    
    load_to_vector_store(config, rebuild=args.rebuild)

