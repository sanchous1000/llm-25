from dataclasses import dataclass

@dataclass
class IndexConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    splitter_type: str = "recursive"
    embedding_model: str = "intfloat/multilingual-e5-large"
    collection_name: str = "rag_collection"
    vector_size: int = 1024
    storage_path: str = "./qdrant_storage"
