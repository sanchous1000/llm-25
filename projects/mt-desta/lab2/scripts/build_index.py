import os
import yaml
import argparse
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
import pickle

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def load_documents(source_dir):
    documents = []
    # Point to the markdown directory
    source_path = Path(__file__).parent.parent / source_dir / "markdown"
    if not source_path.exists():
        print(f"Source directory {source_path} does not exist.")
        return []

    for file_path in source_path.glob("*.md"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Simple frontmatter parser
            metadata = {}
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    text_content = parts[2]
                    try:
                        metadata = yaml.safe_load(frontmatter)
                    except yaml.YAMLError:
                        print(f"Warning: Could not parse frontmatter in {file_path.name}")
                else:
                    text_content = content
            else:
                text_content = content

            # Create a LangChain Document
            from langchain_core.documents import Document
            doc = Document(page_content=text_content, metadata=metadata)
            documents.append(doc)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
        
    return documents

def build_index(rebuild=False):
    config = load_config()
    
    # Vector Store Path
    vector_store_path = str(Path(__file__).parent.parent / config["vector_store"]["path"])
    
    if rebuild and os.path.exists(vector_store_path):
        import shutil
        import time
        print(f"Removing existing vector store at {vector_store_path}...")
        try:
            shutil.rmtree(vector_store_path)
        except PermissionError:
            print("Permission denied. Retrying in 1 second...")
            time.sleep(1)
            try:
                shutil.rmtree(vector_store_path)
            except Exception as e:
                print(f"Failed to remove directory: {e}")
                print("Please ensure no other processes are using the vector store.")
                return
        print("Cleared existing vector store.")

    # Load Documents
    print("Loading documents...")
    docs = load_documents(config["data"]["source_dir"])
    if not docs:
        print("No documents found.")
        return

    # Split Text by Headers
    md_header_splits = []
    if config["splitter"].get("include_headers", True):
        print("Splitting text by headers...")
        from langchain_text_splitters import MarkdownHeaderTextSplitter
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        for doc in docs:
            splits = markdown_splitter.split_text(doc.page_content)
            for split in splits:
                split.metadata.update(doc.metadata)
                # Extract page number from Header 2 if it contains "Page"
                if 'Header 2' in split.metadata:
                    header2 = split.metadata['Header 2']
                    if 'Page' in header2:
                        try:
                            # Extract page number (e.g., "Page 5" -> 5)
                            import re
                            page_match = re.search(r'Page\s+(\d+)', header2)
                            if page_match:
                                split.metadata['page'] = int(page_match.group(1)) - 1  # 0-indexed
                        except:
                            pass
                    elif 'Slide' in header2:
                        try:
                            # Extract slide number (e.g., "Slide 3" -> 3)
                            import re
                            slide_match = re.search(r'Slide\s+(\d+)', header2)
                            if slide_match:
                                split.metadata['slide'] = int(slide_match.group(1))
                        except:
                            pass
                md_header_splits.append(split)
    else:
        print("Skipping header splitting...")
        md_header_splits = docs

    # Split Text by Characters/Tokens
    splitter_type = config["splitter"]["type"]
    print(f"Splitting text using {splitter_type} splitter...")
    
    chunks = []
    if splitter_type == "markdown_only":
        chunks = md_header_splits
    else:
        if splitter_type == "token":
            text_splitter = TokenTextSplitter(
                chunk_size=config["data"]["chunk_size"],
                chunk_overlap=config["data"]["chunk_overlap"]
            )
        else: # default to recursive
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config["data"]["chunk_size"],
                chunk_overlap=config["data"]["chunk_overlap"]
            )
        chunks = text_splitter.split_documents(md_header_splits)
        
    print(f"Created {len(chunks)} chunks.")

    # Create Embeddings and Store
    vector_type = config["vectorization"]["type"]
    print(f"Vectorization type: {vector_type}")
    
    if vector_type == "dense":
        print("Creating embeddings and storing in Chroma...")
        embedding_function = SentenceTransformerEmbeddings(model_name=config["embedding"]["model_name"])
        
        # Prepare HNSW metadata
        hnsw_config = config["vector_store"].get("hnsw", {})
        collection_metadata = {
            "hnsw:space": "cosine", # Default
            "hnsw:construction_ef": hnsw_config.get("ef_construction", 100),
            "hnsw:M": hnsw_config.get("M", 16),
            "hnsw:search_ef": hnsw_config.get("ef_search", 10)
        }
        
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=vector_store_path,
            collection_name=config["vector_store"]["collection_name"],
            collection_metadata=collection_metadata
        )
        print(f"Index built successfully at {vector_store_path}")
    elif vector_type == "sparse":
        print("Creating BM25 retriever...")
        retriever = BM25Retriever.from_documents(chunks)
        
        # Ensure directory exists
        os.makedirs(vector_store_path, exist_ok=True)
        bm25_path = os.path.join(vector_store_path, "bm25_retriever.pkl")
        
        with open(bm25_path, "wb") as f:
            pickle.dump(retriever, f)
        print(f"BM25 retriever saved to {bm25_path}")
        
    elif vector_type == "hybrid":
        print("Creating Hybrid (Dense + Sparse) index...")
        
        # 1. Dense (Chroma)
        print("Step 1: Creating embeddings and storing in Chroma...")
        embedding_function = SentenceTransformerEmbeddings(model_name=config["embedding"]["model_name"])
        
        # Prepare HNSW metadata
        hnsw_config = config["vector_store"].get("hnsw", {})
        collection_metadata = {
            "hnsw:space": "cosine", # Default
            "hnsw:construction_ef": hnsw_config.get("ef_construction", 100),
            "hnsw:M": hnsw_config.get("M", 16),
            "hnsw:search_ef": hnsw_config.get("ef_search", 10)
        }
        
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=vector_store_path,
            collection_name=config["vector_store"]["collection_name"],
            collection_metadata=collection_metadata
        )
        
        # 2. Sparse (BM25)
        print("Step 2: Creating BM25 retriever...")
        retriever = BM25Retriever.from_documents(chunks)
        
        bm25_path = os.path.join(vector_store_path, "bm25_retriever.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump(retriever, f)
        print(f"Hybrid index built. Chroma at {vector_store_path}, BM25 at {bm25_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index from scratch")
    parser.add_argument("--drop-and-reindex", action="store_true", help="Alias for --rebuild: Drop existing index and reindex")
    args = parser.parse_args()
    build_index(rebuild=args.rebuild or args.drop_and_reindex)
