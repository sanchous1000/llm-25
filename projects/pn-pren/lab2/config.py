import os

CHUNK_SIZE = int(512)
OVERLAP = int(50)
SPLITTER_TYPE = 'recursive'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K = int(5)
LLM_MODEL = 'llama3.2:3b'
COLLECTION_NAME = 'docs'

QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'localhost')
OLLAMA_PORT = int(os.getenv('OLLAMA_PORT', 11434))
