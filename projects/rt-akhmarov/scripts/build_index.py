import os
import argparse
import shutil
import pickle
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

SOURCE_DIR = "verl_sources"  
BASE_DB_DIR = "artifacts"
MODEL_NAME = "intfloat/e5-large-v2"

def load_documents(source_dir: str) -> List[Document]:
    loader = DirectoryLoader(source_dir, glob="**/*.*", loader_cls=TextLoader)
    docs = loader.load()
    print(f"📄 Загружено файлов: {len(docs)}")
    return docs

def split_documents(docs: List[Document], chunk_size: int, overlap: int, add_header_to_text: bool):
    separators = [
        # 1. RST Заголовки (Текст + перенос + подчеркивание)
        # Было: r'(?m)\n(?=^.+\n[=\-~^]{3,}\s*$)'
        # Стало: \n + lookahead (не пустая строка, перенос, подчеркивание, конец строки/абзаца)
        r'\n(?=[^\n]+\n[=\-~^]{3,}\s*(?:\n|$))',
        
        # 2. Markdown Заголовки (# Header)
        # Было: r'(?m)\n(?=^#{1,6}\s)'
        # Стало: просто убираем ^, так как мы и так после \n
        r'\n(?=#{1,6}\s)',
        
        # 3. "Ленивые" заголовки (Жирным), которые НЕ списки
        # Было: r'(?m)\n(?=^\s*(?![-\*]\s)\*\*.*\*\*\s*$)'
        # Заменяем $ на (?:\n|$) и убираем ^
        r'\n(?=\s*(?![-\*]\s)\*\*.*?\*\*\s*(?:\n|$))',
        
        r'\n(?=\.\.\s+code::)',  # .. code:: 
        r'\n(?=class\s+)',       # class ...
        r'\n(?=def\s+)',         # def ...
        
        r'\n\n',
        r'\n',
        r' '
    ]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def token_length_function(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=separators,
        is_separator_regex=True,
        keep_separator=True,
        length_function=token_length_function
    )
    
    final_chunks = []
    
    print("✂️  Начинаю умное разбиение (Fixed Regex)...")
    
    for doc in docs:
        content_len = len(doc.page_content)
        
        if content_len < chunk_size * 1.2: 
            if add_header_to_text:
                filename = os.path.basename(doc.metadata.get('source', 'unknown'))
                doc.page_content = f"Source Document: {filename}\nContent:\n{doc.page_content}"
            final_chunks.append(doc)
            continue
            
        chunks = splitter.split_documents([doc])
        
        for chunk in chunks:
            if add_header_to_text:
                filename = os.path.basename(chunk.metadata.get('source', 'unknown'))
                chunk.page_content = f"Source Document: {filename}\nContent:\n{chunk.page_content}"
            final_chunks.append(chunk)

    print(f"✅ Обработано документов: {len(docs)}")
    print(f"✂️  Итого чанков: {len(final_chunks)}")
    
    return final_chunks

def build_dense_index(chunks, model_name, persist_path):
    """Строит векторную базу (ChromaDB)."""
    print(f"🧠 Генерирую эмбеддинги: {model_name}...")
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    
    _ = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_path
    )
    print(f"✅ Dense индекс сохранен в: {persist_path}")

def build_sparse_index(chunks, persist_path):
    print("🔍 Строю BM25 индекс...")
    retriever = BM25Retriever.from_documents(chunks)
    
    os.makedirs(persist_path, exist_ok=True)
    file_path = os.path.join(persist_path, "bm25_retriever.pkl")
    
    with open(file_path, "wb") as f:
        pickle.dump(retriever, f)
    print(f"✅ Sparse индекс сохранен в: {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Сборка RAG индекса")
    
    parser.add_argument("--chunk_size", type=int, default=512, help="Размер чанка")
    parser.add_argument("--overlap", type=int, default=100, help="Перекрытие чанков")
    parser.add_argument("--emb_model", type=str, default="intfloat/e5-large-v2", help="Модель эмбеддингов")
    parser.add_argument("--add_header", action="store_true", help="Добавлять имя файла в текст чанка")
    parser.add_argument("--type", type=str, choices=["dense", "sparse", "hybrid"], default="dense", help="Тип индекса")

    args = parser.parse_args()

    version_name = f"{args.type}_cs{args.chunk_size}_ov{args.overlap}_head{args.add_header}"
    output_path = os.path.join(BASE_DB_DIR, version_name)

    if os.path.exists(output_path):
        print(f"🗑️  Удаляю старую версию: {output_path}")
        shutil.rmtree(output_path)

    docs = load_documents(SOURCE_DIR)
    chunks = split_documents(docs, args.chunk_size, args.overlap, args.add_header)

    if args.type in ["dense", "hybrid"]:
        build_dense_index(chunks, args.emb_model, output_path)
    
    if args.type in ["sparse", "hybrid"]:
        build_sparse_index(chunks, output_path)

if __name__ == "__main__":
    main()