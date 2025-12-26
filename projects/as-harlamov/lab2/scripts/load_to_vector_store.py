#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

from config import Config
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
import json


def main():
    parser = argparse.ArgumentParser(description="Load embeddings to vector store")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild collection")
    parser.add_argument("--drop-and-reindex", action="store_true", help="Drop and recreate collection")
    
    args = parser.parse_args()
    
    config = Config()
    
    if args.drop_and_reindex:
        config.vector_store.recreate_collection = True
    
    project_root = Path(__file__).parent.parent
    embeddings_dir = project_root / "data/embeddings"
    embeddings_file = embeddings_dir / "embeddings.json"
    
    if not embeddings_file.exists():
        print(f"Embeddings file not found: {embeddings_file}")
        print("Please run build_index.py first")
        return
    
    print("Loading embeddings...")
    embedding_generator = EmbeddingGenerator(config)
    chunks_with_embeddings = embedding_generator.load_embeddings(embeddings_file)
    
    print(f"Loaded {len(chunks_with_embeddings)} chunks with embeddings")
    
    vector_store = VectorStore(config, embedding_generator)
    
    print(f"Uploading to vector store: {config.vector_store.url}")
    print(f"Collection: {config.vector_store.collection_name}")
    
    if args.rebuild or args.drop_and_reindex:
        print("Rebuilding collection...")
    
    vector_store.upload_chunks(chunks_with_embeddings)
    
    info = vector_store.get_collection_info()
    print(f"\nCollection created/updated successfully")
    print(f"  Collection name: {info['name']}")
    print(f"  Vectors count: {info['vectors_count']}")
    print(f"  Status: {info['status']}")


if __name__ == "__main__":
    main()

