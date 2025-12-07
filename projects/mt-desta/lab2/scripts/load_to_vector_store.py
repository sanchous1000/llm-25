"""
Script to load pre-built chunks and embeddings into vector store.
This is a separate module as required by the lab specification.
It can be used to load data that was previously processed by build_index.py.
"""
import os
import yaml
import argparse
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from build_index import load_config, load_documents
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.retrievers import BM25Retriever
import pickle

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

def load_to_vector_store(rebuild: bool = False, drop_and_reindex: bool = False):
    """
    Load chunks and embeddings into vector store.
    
    This function essentially calls build_index, but is provided as a separate
    module as specified in the requirements. It ensures the vector store is
    properly loaded with the current configuration.
    
    Args:
        rebuild: If True, rebuild the index from scratch
        drop_and_reindex: Alias for rebuild
    """
    from build_index import build_index
    
    # This is essentially a wrapper around build_index
    # The actual implementation is in build_index.py to avoid code duplication
    build_index(rebuild=rebuild or drop_and_reindex)

def verify_vector_store():
    """Verify that the vector store is properly loaded and accessible."""
    config = load_config()
    vector_store_path = str(Path(__file__).parent.parent / config["vector_store"]["path"])
    vector_type = config["vectorization"]["type"]
    
    print(f"Verifying vector store at: {vector_store_path}")
    print(f"Vectorization type: {vector_type}")
    
    if vector_type in ["dense", "hybrid"]:
        try:
            embedding_function = SentenceTransformerEmbeddings(
                model_name=config["embedding"]["model_name"]
            )
            vector_store = Chroma(
                persist_directory=vector_store_path,
                embedding_function=embedding_function,
                collection_name=config["vector_store"]["collection_name"]
            )
            
            # Try a simple query
            results = vector_store.similarity_search("test", k=1)
            print(f"✓ Dense vector store loaded successfully. Found {len(results)} test result(s).")
            
            # Get collection info
            collection = vector_store._collection
            count = collection.count()
            print(f"✓ Collection contains {count} documents.")
            
        except Exception as e:
            print(f"✗ Error loading dense vector store: {e}")
            return False
    
    if vector_type in ["sparse", "hybrid"]:
        try:
            bm25_path = os.path.join(vector_store_path, "bm25_retriever.pkl")
            if os.path.exists(bm25_path):
                with open(bm25_path, "rb") as f:
                    retriever = pickle.load(f)
                print(f"✓ BM25 retriever loaded successfully.")
            else:
                print(f"✗ BM25 retriever not found at {bm25_path}")
                return False
        except Exception as e:
            print(f"✗ Error loading BM25 retriever: {e}")
            return False
    
    print("\n✓ Vector store verification complete!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load chunks and embeddings into vector store")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index from scratch")
    parser.add_argument("--drop-and-reindex", action="store_true", help="Alias for --rebuild: Drop existing index and reindex")
    parser.add_argument("--verify", action="store_true", help="Verify that vector store is properly loaded")
    args = parser.parse_args()
    
    if args.verify:
        verify_vector_store()
    else:
        load_to_vector_store(rebuild=args.rebuild or args.drop_and_reindex)

