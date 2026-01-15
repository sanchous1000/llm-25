import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env file from source directory
load_dotenv()


@dataclass
class Config:
    # Langfuse
    langfuse_host: str = "http://localhost:3000"
    # Use `export LANGFUSE_PUBLIC_KEY=...`
    langfuse_public_key: str = ""
    # Use `export LANGFUSE_SECRET_KEY=...`
    langfuse_secret_key: str = ""
    
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "ydkjs_docs"
    
    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # LLM
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:3b"
    llm_temperature: float = 0.3
    
    # RAG
    top_k: int = 5
    
    # Dataset
    dataset_name: str = "ydkjs-qa"
    
    def __post_init__(self):
        self.langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", self.langfuse_public_key)
        self.langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", self.langfuse_secret_key)
        self.langfuse_host = os.getenv("LANGFUSE_HOST", self.langfuse_host)


def load_config(**overrides) -> Config:
    config = Config()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
