import os

CHUNK_SIZE = int(512)
OVERLAP = int(50)
SPLITTER_TYPE = 'recursive'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K = int(5)
LLM_MODEL = 'llama3.2:3b'
COLLECTION_NAME = 'docs'

