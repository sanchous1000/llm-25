import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import yaml
from dotenv import load_dotenv
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent))
from source.chunking import MarkdownSplitter
from source.embeddings import DenseEmbedder
from source.utils import load_config, compute_config_hash, save_metadata
load_dotenv()


class IndexBuilder:
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = load_config(config_path)
        self.config_hash = compute_config_hash(self.config)
        self._setup_paths()
    
    def _setup_paths(self):
        base_dir = Path(self.config['paths']['output_dir'])
        version = self.config_hash[:8]
        base_dir = base_dir / f"v_{version}"
        self.chunks_path = base_dir / self.config['paths']['chunks_dir']
        self.embeddings_path = base_dir / self.config['paths']['embeddings_dir']
        self.index_path = base_dir / self.config['paths']['index_dir']
        
        for path in [self.chunks_path, self.embeddings_path, self.index_path]:
            path.mkdir(parents=True, exist_ok=True)
        
    
    def _load_documents(self) :
        """
        Загружает документы из корпуса.
        
        Returns:
            Список документов с метаданными
        """
        corpus_dir = Path(self.config['paths']['corpus_dir'])
        documents = []
        
        for file_path in corpus_dir.glob('*.md'):
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = self._extract_metadata(content)
            metadata['file_path'] = str(file_path)
            metadata['filename'] = file_path.name
            
            documents.append({
                'content': content,
                'metadata': metadata
            })
        
        return documents
    
    def _extract_metadata(self, content):
        """Извлекает метаданные из frontmatter."""
        metadata = {}
        
        if content.startswith('---'):
            end_idx = content.find('---', 3)
            if end_idx != -1:
                frontmatter = content[3:end_idx].strip()
                metadata = yaml.safe_load(frontmatter) or {}

        
        return metadata
    
    def _create_splitter(self):
        # Передаем модель эмбеддингов для использования правильного токенизатора
        embedding_model = self.config['embeddings']['dense']['model']
        return MarkdownSplitter(
            chunk_size=self.config['chunking']['chunk_size'],
            chunk_overlap=self.config['chunking']['chunk_overlap'],
            header_levels=self.config['chunking']['header_levels'],
            include_headers_in_text=self.config['chunking']['include_headers_in_text'],
            min_chunk_size=self.config['chunking'].get('min_chunk_size', 50),
            tokenizer_model=embedding_model
        )

    
    def _create_embedder(self):
        return DenseEmbedder(self.config['embeddings']['dense'])
    
    def build_chunks(self, documents):
        """
        Разбивает документы на чанки.
        
        Args:
            documents: Список документов
            
        Returns:
            Список чанков с метаданными
        """
        start_time = time.time()
        splitter = self._create_splitter()
        all_chunks = []
        
        for doc in tqdm(documents, desc="Разбиение документов"):

            chunks = splitter.split_document(
                doc['content'],
                doc['metadata']
            )
            all_chunks.extend(chunks)
        
        elapsed_time = time.time() - start_time
        self._save_chunks(all_chunks)
        
        return all_chunks
    
    def _save_chunks(self, chunks):
        """Сохраняет чанки на диск."""
        chunks_file = self.chunks_path / 'chunks.json'
        
        def convert_dates(obj):
            if isinstance(obj, dict):
                return {k: convert_dates(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_dates(item) for item in obj]
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            else:
                return obj
        
        chunks_serializable = convert_dates(chunks)
        
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_serializable, f, ensure_ascii=False, indent=2)
    
    def build_embeddings(self, chunks):
        """
        Создает эмбеддинги для чанков.
        
        Args:
            chunks: Список чанков
            
        Returns:
            Словарь с эмбеддингами и метаданными
        """
        start_time = time.time()
        embedder = self._create_embedder()
        # Извлекаем тексты чанков
        texts = [chunk['text'] for chunk in chunks]
        # Векторизуем
        embeddings_data = embedder.embed_texts(texts)
        elapsed_time = time.time() - start_time
        self._save_embeddings(embeddings_data)
        
        return embeddings_data
    
    def _save_embeddings(self, embeddings_data):
        """Сохраняет эмбеддинги на диск."""
        if 'dense' in embeddings_data:
            np.save(
                self.embeddings_path / 'dense_embeddings.npy',
                embeddings_data['dense']
            )
    
    def build(self):
        """Основной метод построения индекса."""
        total_start_time = time.time()
        
        print("Построение индекса...")
        documents = self._load_documents()
        if not documents:
            print("Документы не найдены!")
            return

        chunks = self.build_chunks(documents)
        if not chunks:
            print("Чанки не созданы!")
            return
        
        embeddings = self.build_embeddings(chunks)
        save_metadata(self.index_path, self.config, self.config_hash, len(documents), len(chunks))
        total_elapsed_time = time.time() - total_start_time
        
        print(f"Документов: {len(documents)}")
        print(f"Чанков: {len(chunks)}")
        print(f"Путь: {self.index_path.parent}")
        print(f"Общее время: {total_elapsed_time:.2f} секунд ({total_elapsed_time/60:.2f} минут)")


def main(config):
    try:
        builder = IndexBuilder(config_path=config)
        builder.build()
    except Exception as e:
        print(f"Ошибка построения индекса: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main(config='config.yaml')