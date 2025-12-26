import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Config:
    # Data paths
    data_dir: str = "data"
    raw_docs_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    index_dir: str = "data/index"
    
    # Chunking parameters
    chunk_size: int = 256  # in tokens
    chunk_overlap: int = 25  # in tokens
    splitter_type: Literal["recursive", "markdown", "hybrid"] = "markdown"
    include_headers_in_chunk: bool = True
    
    # Embedding parameters
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_type: Literal["dense", "sparse", "hybrid"] = "dense"
    
    # Vector store parameters
    vector_store: Literal["qdrant", "faiss"] = "qdrant"
    collection_name: str = "ydkjs_docs"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    
    # HNSW index parameters
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    hnsw_ef_search: int = 50
    
    # RAG parameters
    top_k: int = 5
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:3b"
    llm_temperature: float = 0.3
    
    # Evaluation
    eval_questions_file: str = "data/eval_questions.json"
    
    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.raw_docs_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)


def load_config(**overrides) -> Config:
    config = Config()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config

