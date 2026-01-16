import os
import json
import hashlib
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

LIMIT = 1024

class EmbeddingConfig:
    def __init__(
        self, 
        data_dir: str,
        output_dir: str = "./embeddings_output",
        
        split_strategy: str = "hybrid",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        include_headers_in_content: bool = True,
        
        dense_model: str = "BAAI/bge-m3",
        use_sparse: bool = True,
        sparse_type: str = "bm25",
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.split_strategy = split_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.include_headers_in_content = include_headers_in_content
        self.dense_model = dense_model
        self.use_sparse = use_sparse
        self.sparse_type = sparse_type
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_hash(self) -> str:
        config_str = (
            f"{self.split_strategy}-{self.chunk_size}-{self.chunk_overlap}-"
            f"{self.include_headers_in_content}-{self.dense_model}-"
            f"{self.use_sparse}-{self.sparse_type}"
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

class EmbeddingBuilder:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.chunks = []
        
        print(f"Загрузка модели эмбеддингов: {self.config.dense_model}")
        self.dense_model = SentenceTransformer(self.config.dense_model)
        print("Модель загружена")

    def load_markdown_files(self) -> List[Dict[str, Any]]:
        documents = []
        for md_file in self.config.data_dir.rglob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                metadata = {}

                metadata['source_path'] = str(md_file.relative_to(self.config.data_dir))
                documents.append({"text": content, "metadata": metadata})
            except Exception as e:
                print(f"Ошибка чтения {md_file}: {e}")
        return documents

    def split_documents(self, documents: List[Dict[str, Any]]):
        all_chunks = []
        
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", " ", ".", "!", "?", ",", "，", "。", ""],
        )

        print(f"Разбивка документов (Стратегия: {self.config.split_strategy})...")

        for doc in documents:
            text = doc['text']
            source_metadata = doc['metadata']

            if self.config.split_strategy == "recursive":
                splits = recursive_splitter.split_text(text)
                for split in splits:
                    all_chunks.append({
                        "content": split,
                        "metadata": source_metadata.copy()
                    })

            elif self.config.split_strategy in ["markdown", "hybrid"]:
                try:
                    if text.startswith("---"):
                        text = text.split("---", 2)[-1] 
                    
                    md_splits = markdown_splitter.split_text(text)
                except Exception as e:
                    md_splits = [text]
                    print(f"Markdown splitter failed for {source_metadata}, falling back to recursive.")

                if self.config.split_strategy == "hybrid":
                    for section in md_splits:
                        content_with_headers = ""
                        if self.config.include_headers_in_content:
                            headers = []
                            for k, v in section.metadata.items():
                                headers.append(v)
                            content_with_headers = " | ".join(headers) + "\n" + section.page_content
                        else:
                            content_with_headers = section.page_content
                        
                        final_splits = recursive_splitter.split_text(content_with_headers)
                        
                        for split in final_splits:
                            combined_meta = source_metadata.copy()
                            combined_meta.update(section.metadata) 
                            
                            all_chunks.append({
                                "content": split,
                                "metadata": combined_meta
                            })
                else:
                    for section in md_splits:
                        combined_meta = source_metadata.copy()
                        combined_meta.update(section.metadata)
                        all_chunks.append({
                            "content": section.page_content,
                            "metadata": combined_meta
                        })

        tokenizer = self.dense_model.tokenizer
        for chunk in all_chunks:
            encoded = tokenizer.encode(chunk["content"])
            if len(encoded) > LIMIT:
                splits = recursive_splitter.split_text(chunk["content"])
                for split in splits:
                    all_chunks.append({
                        "content": split,
                        "metadata": chunk["metadata"].copy()
                    })
                all_chunks.remove(chunk)

        self.chunks = all_chunks
        print(f"Всего создано чанков: {len(all_chunks)}")

    def generate_embeddings(self):
        if not self.chunks:
            print("Нет чанков для обработки.")
            return

        texts = [c['content'] for c in self.chunks]
        
        print(f"Генерация Dense векторов ({self.config.dense_model})")
        dense_vectors = self.dense_model.encode(
            texts, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )

        for i, chunk in enumerate(self.chunks):
            chunk['dense_vector'] = dense_vectors[i].tolist()

        if self.config.use_sparse:
            print("Генерация Sparse векторов")
            
            vectorizer = TfidfVectorizer(
                max_features=100,
                norm='l2'
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            feature_names = vectorizer.get_feature_names_out()
            
            for i, chunk in enumerate(self.chunks):
                row = tfidf_matrix[i]
                indices = row.nonzero()[1]
                values = row.data
                
                sparse_dict = {feature_names[idx].encode('utf-8').decode(): float(val) for idx, val in zip(indices, values)}
                chunk['sparse_vector'] = sparse_dict

    def save_artifacts(self):
        config_hash = self.config.get_hash()
        version_tag = f"v{config_hash}"
        output_file = self.config.output_dir / f"chunks_{self.config.dense_model.replace('/', '-')}_{version_tag}.jsonl"
        
        print(f"Сохранение результатов в {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in self.chunks:
                record = {
                    "id": f"{chunk['metadata'].get('source_path', 'unknown')}_{chunk.get('idx', '')}",
                    "text": chunk['content'],
                    "metadata": chunk['metadata'],
                    "vector": chunk.get('dense_vector'),
                    "sparse_vector": chunk.get('sparse_vector')
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        config_file = self.config.output_dir / f"config_{version_tag}.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
            
        print("Сохранение завершено")
        return output_file

    def check_version(self) -> bool:
        config_hash = self.config.get_hash()
        pattern = f"chunks_*_{config_hash}.jsonl"
        
        existing_files = list(self.config.output_dir.glob(pattern))
        if existing_files:
            print(f"Обнаружен актуальный билд с конфигурацией {config_hash}: {existing_files[0].name}")
            return True
        return False

    def run(self, force=False):
        if not force and self.check_version():
            print("Пропуск обработки (актуальные данные уже существуют). Используйте --force для пересборки.")
            return

        print("Запуск пайплайна")
        docs = self.load_markdown_files()
        self.split_documents(docs)
        self.generate_embeddings()
        self.save_artifacts()
        print("Готово")

def main():
    parser = argparse.ArgumentParser(description="Построение чанков и эмбеддингов")
    
    parser.add_argument("--data_dir", required=True, help="Папка с нормализованными MD файлами")
    parser.add_argument("--output_dir", default="./embeddings_output", help="Папка для сохранения результата")
    
    parser.add_argument("--strategy", choices=["recursive", "markdown", "hybrid"], default="hybrid")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    parser.add_argument("--no_headers_in_content", action="store_false", dest="include_headers_in_content", help="Не включать текст заголовков в сам чанк")
    
    parser.add_argument("--dense_model", default="BAAI/bge-m3", help="Название модели HuggingFace для Dense")
    parser.add_argument("--no_sparse", action="store_false", dest="use_sparse", help="Отключить генерацию Sparse векторов")
    
    parser.add_argument("--force", action="store_true", help="Принудительно пересобрать, даже если конфиг не менялся")

    args = parser.parse_args()
    load_dotenv()

    config = EmbeddingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split_strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        include_headers_in_content=args.include_headers_in_content,
        dense_model=args.dense_model,
        use_sparse=args.use_sparse
    )

    builder = EmbeddingBuilder(config)
    builder.run(force=args.force)

if __name__ == "__main__":
    main()