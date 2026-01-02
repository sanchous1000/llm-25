from typing import Any
from qdrant_client.models import Distance, HnswConfigDiff, VectorParams
from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient
import json
import click
import yaml
import logging


def upsert_qdrant(
    embeddings: list[dict],
    qdrant_client: QdrantClient,
    hnsw_config: dict[str, Any],
    vector_size: int = 384,
    collection_name: str = "dnd5e_baseline",
    distance: Distance = Distance.COSINE,
    recreate: bool = True,
    batch_size: int = 500,
):
        if hnsw_config is None:
            hnsw_config = {}
        size = len(embeddings[0]["vector"]) if embeddings else vector_size
        if recreate:
            try:
                qdrant_client.delete_collection(collection_name=collection_name)
            except Exception:
                logging.warning("Collection not found")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=size, 
                    distance=distance,
                    hnsw_config=HnswConfigDiff(
                        **hnsw_config
                    )
                ),
            )
        points = [PointStruct(**emb) for emb in embeddings]
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(collection_name=collection_name, wait=True, points=batch)
            print(f"Uploaded batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")


@click.command()
@click.option("--config", default="scripts/config.yaml", help="Path to YAML config")
def main(config: str):
    with open(config) as f:
        cfg = yaml.safe_load(f)["vector_store"]
    
    qdrant_client = QdrantClient(url=cfg["qdrant_url"], timeout=300)
    
    with open(cfg["embeddings_path"], "r") as f:
        data = json.load(f)
        embeddings = data.get("embeddings", data)
    
    upsert_qdrant(
        embeddings, 
        qdrant_client, 
        collection_name=cfg["collection_name"], 
        recreate=cfg["rebuild"],
        hnsw_config=cfg["hnsw"],
    )
    
    print(f"Loaded {len(embeddings)} embeddings to {cfg['collection_name']}")

if __name__ == "__main__":
    main()