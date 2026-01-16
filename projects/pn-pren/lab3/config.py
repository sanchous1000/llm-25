import os

# Chunking and embedding settings
CHUNK_SIZE = 512
OVERLAP = 50
SPLITTER_TYPE = 'recursive'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K = 5
LLM_MODEL = 'llama3.2:3b'
COLLECTION_NAME = 'docs'

# Service hosts
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'localhost')
OLLAMA_PORT = int(os.getenv('OLLAMA_PORT', 11434))

# Langfuse settings
LANGFUSE_HOST = os.getenv('LANGFUSE_HOST', 'http://localhost:3000')
LANGFUSE_PUBLIC_KEY = os.getenv('LANGFUSE_PUBLIC_KEY', '')
LANGFUSE_SECRET_KEY = os.getenv('LANGFUSE_SECRET_KEY', '')

# Dataset settings
DATASET_NAME = 'rag-eval-dataset'
