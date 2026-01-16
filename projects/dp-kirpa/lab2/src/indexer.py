import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from src.config import IndexConfig
import uuid
from transformers import AutoTokenizer

class Indexer:
    def __init__(self, config: IndexConfig):
        self.cfg = config
        self.embeddings = HuggingFaceEmbeddings(model_name=self.cfg.embedding_model)
        self.client = QdrantClient(path=self.cfg.storage_path) 
    
    def count_tokens(text: str) -> int:
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.embedding_model)
        return len(tokenizer.encode(text))

    def load_documents(self, input_dir):
        docs = []
        for f in os.listdir(input_dir):
            if f.endswith(".md"):
                loader = TextLoader(os.path.join(input_dir, f), encoding='utf-8')
                docs.extend(loader.load())
        return docs

    def split_documents(self, docs):
        if self.cfg.splitter_type == "markdown":
            headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
            splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.cfg.chunk_size, 
                chunk_overlap=self.cfg.chunk_overlap,
                length_function=self.count_tokens,
            )
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.cfg.chunk_size, 
                chunk_overlap=self.cfg.chunk_overlap,
                length_function=self.count_tokens,
                separators=["\n## ", "\n### ", "\n", " ", ""]
            )
            
        return text_splitter.split_documents(docs)

    def rebuild_index(self, input_dir):
        print(f"Rebuilding index with {self.cfg.splitter_type}, size={self.cfg.chunk_size}...")
        
        if self.client.collection_exists(self.cfg.collection_name):
            self.client.delete_collection(self.cfg.collection_name)
        
        self.client.create_collection(
            collection_name=self.cfg.collection_name,
            vectors_config=VectorParams(size=self.cfg.vector_size, distance=Distance.COSINE)
        )

        raw_docs = self.load_documents(input_dir)
        chunks = self.split_documents(raw_docs)
        print(f"Created {len(chunks)} chunks.")

        points = []
        
        texts = [d.page_content for d in chunks]
        Metas = [d.metadata for d in chunks]
        
        texts_for_embed = [f"passage: {t}" for t in texts]
        
        vectors = self.embeddings.embed_documents(texts_for_embed)

        for i, vector in enumerate(vectors):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": texts[i],
                    "source": Metas[i].get("source", "unknown"),
                    "config_chunk_size": self.cfg.chunk_size
                }
            ))

        self.client.upsert(collection_name=self.cfg.collection_name, points=points)
        print("Indexing complete.")

if __name__ == "__main__":
    indexer = Indexer(IndexConfig(chunk_size=500, overlap=50))
    indexer.rebuild_index("data/processed")

