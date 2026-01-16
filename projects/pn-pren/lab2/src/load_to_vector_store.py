import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np

import config


def load_to_qdrant(index_dir, collection_name=None, rebuild=False):
    index_dir = Path(index_dir)
    
    if collection_name is None:
        collection_name = config.COLLECTION_NAME
    
    with open(index_dir / 'chunks.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    with open(index_dir / 'index_config.json', 'r', encoding='utf-8') as f:
        index_config = json.load(f)
    
    if not index_config.get('use_sparse'):
        embeddings = np.load(index_dir / 'embeddings.npy')
        vector_size = embeddings.shape[1]
    else:
        raise ValueError("Sparse embeddings didnt work(")
    
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    
    if rebuild:
        try:
            client.delete_collection(collection_name)
            print(f"Delete collection: {collection_name}")
        except:
            pass
    
    try:
        client.get_collection(collection_name)
        print(f"Collection {collection_name} already exists")
        if not rebuild:
            return
    except:
        print(f"Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    
    points = []
    for i, chunk in enumerate(chunks):
        point = PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                'text': chunk['text'],
                'source': chunk['source'],
                'chunk_id': chunk['chunk_id'],
                'original_file': chunk['original_file'],
                'file_type': chunk['file_type']
            }
        )
        points.append(point)
    
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        print(f"Uploaded {min(i + batch_size, len(points))}/{len(points)} points")
    
    print(f"\nLoaded {len(points)} points to collection '{collection_name}'")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--index-dir', default='output/index')
    args = parser.parse_args()
    
    load_to_qdrant(args.index_dir, config.COLLECTION_NAME, True)
