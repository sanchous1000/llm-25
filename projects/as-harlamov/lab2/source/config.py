import os
from pathlib import Path
from typing import Optional
import yaml
from dotenv import load_dotenv

load_dotenv()


class ChunkingConfig:
    def __init__(self, config: dict):
        self.strategy = config.get("strategy", "recursive")
        self.chunk_size = config.get("chunk_size", 500)
        self.chunk_overlap = config.get("chunk_overlap", 50)
        self.include_headers = config.get("include_headers", True)
        self.separators = config.get("separators", ["\n\n", "\n", " ", ""])


class EmbeddingConfig:
    def __init__(self, config: dict):
        self.type = config.get("type", "dense")
        self.model = config.get("model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.provider = config.get("provider", "sentence-transformers")
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.bm25_k1 = config.get("bm25_k1", 1.5)
        self.bm25_b = config.get("bm25_b", 0.75)


class VectorStoreConfig:
    def __init__(self, config: dict):
        self.provider = config.get("provider", "qdrant")
        self.url = config.get("url", os.getenv("QDRANT_URL", "http://localhost:6333"))
        self.api_key = config.get("api_key") or os.getenv("QDRANT_API_KEY")
        self.collection_name = config.get("collection_name", "rag_documents")
        self.hnsw_config = config.get("hnsw_config", {})
        self.recreate_collection = config.get("recreate_collection", False)


class RAGConfig:
    def __init__(self, config: dict):
        self.top_k = config.get("top_k", 5)
        self.llm_provider = config.get("llm_provider", "openai")
        self.llm_model = config.get("llm_model", "gpt-3.5-turbo")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
        if self.llm_provider == "local":
            self.base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
        else:
            self.base_url = None


class Config:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        else:
            config_path = Path(config_path)
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        
        self.documents = config.get("documents", {})
        self.chunking = ChunkingConfig(config.get("chunking", {}))
        self.embeddings = EmbeddingConfig(config.get("embeddings", {}))
        self.vector_store = VectorStoreConfig(config.get("vector_store", {}))
        self.rag = RAGConfig(config.get("rag", {}))
        self.evaluation = config.get("evaluation", {})

