"""
Скрипт для разбиения документов на чанки и построения гибридных эмбеддингов.
Выполняет задание 3: разбиение на чанки и построение эмбеддингов (sparse/dense).
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from utils import get_device, get_config_hash, tokenize, load_documents


class MarkdownChunker:
    """Класс для разбиения Markdown документов на чанки."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.splitter_config = config.get("splitter", {})
        self.chunk_size = self.splitter_config.get("chunk_size", 500)
        self.chunk_overlap = self.splitter_config.get("chunk_overlap", 50)
        self.include_headers = self.splitter_config.get("include_headers", True)
        self.splitter_type = self.splitter_config.get("type", "hybrid")
        
    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """Извлекает метаданные из YAML front matter."""
        metadata = {}
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                yaml_content = parts[1].strip()
                for line in yaml_content.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip()
                        metadata[key] = value
                content = parts[2].strip()
        return metadata, content
    
    def split_markdown(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Разбивает Markdown документ на чанки."""
        doc_metadata, doc_content = self.extract_metadata(content)
        # Объединяем метаданные
        full_metadata = {**metadata, **doc_metadata}
        
        chunks = []
        
        if self.splitter_type == "hybrid":
            # Гибридный подход: сначала по заголовкам, потом рекурсивно
            headers_to_split_on = self.splitter_config.get("headers_to_split_on", [
                ["#", "Header 1"],
                ["##", "Header 2"],
                ["###", "Header 3"]
            ])
            
            # Преобразуем в формат для langchain
            headers = [(h[0], h[1]) for h in headers_to_split_on]
            
            md_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers,
                strip_headers=False
            )
            
            # Разбиваем по заголовкам
            md_chunks = md_splitter.split_text(doc_content)
            
            # Затем разбиваем каждый кусок рекурсивно
            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            
            for md_chunk in md_chunks:
                if isinstance(md_chunk, Document):
                    chunk_text = md_chunk.page_content
                    chunk_meta = md_chunk.metadata.copy()
                else:
                    chunk_text = str(md_chunk)
                    chunk_meta = {}
                
                # Добавляем заголовки в текст чанка, если нужно
                if self.include_headers and chunk_meta:
                    header_text = " ".join([f"{k}: {v}" for k, v in chunk_meta.items() if k.startswith("Header")])
                    if header_text:
                        chunk_text = f"{header_text}\n\n{chunk_text}"
                
                # Разбиваем на более мелкие чанки
                sub_chunks = recursive_splitter.split_text(chunk_text)
                
                for i, sub_chunk in enumerate(sub_chunks):
                    chunk_metadata = {
                        **full_metadata,
                        **chunk_meta,
                        "chunk_index": i,
                        "chunk_size": len(sub_chunk)
                    }
                    chunks.append({
                        "text": sub_chunk,
                        "metadata": chunk_metadata
                    })
        else:
            # Простой рекурсивный сплиттер
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            text_chunks = splitter.split_text(doc_content)
            
            for i, chunk_text in enumerate(text_chunks):
                chunk_metadata = {
                    **full_metadata,
                    "chunk_index": i,
                    "chunk_size": len(chunk_text)
                }
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
        
        return chunks


class EmbeddingBuilder:
    """Класс для построения dense и sparse эмбеддингов."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embeddings_config = config.get("embeddings", {})
        self.embedding_type = self.embeddings_config.get("type", "hybrid")
        
        # Инициализация dense модели
        self.dense_model = None
        if self.embedding_type in ["dense", "hybrid"]:
            dense_config = self.embeddings_config.get("dense", {})
            model_name = dense_config.get("model", "BAAI/bge-base-en-v1.5")
            device_preference = dense_config.get("device", None)
            device = get_device(device_preference)
            print(f"Загрузка dense модели: {model_name}")
            print(f"Используемое устройство: {device}")
            self.dense_model = SentenceTransformer(model_name, device=device)
        
        # Инициализация sparse модели (BM25)
        self.bm25 = None
        self.tokenized_corpus = None
    
    def build_dense_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Строит dense эмбеддинги."""
        if not self.dense_model:
            return None
        
        print(f"Построение dense эмбеддингов для {len(texts)} чанков...")
        embeddings = self.dense_model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def build_sparse_embeddings(self, texts: List[str]) -> Optional[Any]:
        """Строит sparse эмбеддинги (BM25)."""
        if not texts:
            print("Warning: Нет текстов для построения sparse эмбеддингов")
            return None
        
        print(f"Построение sparse эмбеддингов (BM25) для {len(texts)} чанков...")
        
        # Токенизация для BM25
        tokenized_corpus = [self._tokenize(text) for text in texts]
        self.tokenized_corpus = tokenized_corpus
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        return self.bm25
    
    def _tokenize(self, text: str) -> List[str]:
        """Простая токенизация для BM25."""
        return tokenize(text)
    
    def get_sparse_scores(self, query: str, top_k: int = 10) -> List[tuple]:
        """Получает sparse scores для запроса."""
        if not self.bm25:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        # Возвращаем топ-k индексов с их scores
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(idx, scores[idx]) for idx in top_indices]

def save_chunks_and_embeddings(
    chunks: List[Dict[str, Any]],
    dense_embeddings: Optional[List[List[float]]],
    sparse_model: Optional[Any],
    output_dir: str,
    config_hash: str
):
    """Сохраняет чанки и эмбеддинги с версионированием."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    version_dir = output_path / f"v_{config_hash}"
    version_dir.mkdir(exist_ok=True)
    
    # Сохраняем чанки
    chunks_file = version_dir / "chunks.json"
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"Сохранено {len(chunks)} чанков в {chunks_file}")
    
    # Сохраняем dense эмбеддинги
    if dense_embeddings:
        dense_file = version_dir / "dense_embeddings.pkl"
        with open(dense_file, 'wb') as f:
            pickle.dump(dense_embeddings, f)
        print(f"Сохранены dense эмбеддинги в {dense_file}")
    
    # Сохраняем sparse модель (BM25)
    if sparse_model:
        sparse_file = version_dir / "sparse_model.pkl"
        with open(sparse_file, 'wb') as f:
            pickle.dump(sparse_model, f)
        print(f"Сохранена sparse модель (BM25) в {sparse_file}")
    
    # Сохраняем метаданные
    metadata = {
        "num_chunks": len(chunks),
        "has_dense": dense_embeddings is not None,
        "has_sparse": sparse_model is not None,
        "config_hash": config_hash
    }
    
    metadata_file = version_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Метаданные сохранены в {metadata_file}")
    
    return version_dir


def main():
    """Основная функция."""
    # Загружаем конфигурацию
    config_file = Path("config/config.json")
    if not config_file.exists():
        print(f"Ошибка: файл конфигурации {config_file} не найден")
        return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("=" * 60)
    print("Построение индекса: разбиение на чанки и эмбеддинги")
    print("=" * 60)
    print(f"Конфигурация: {config_file}")
    
    # Вычисляем хеш конфигурации для версионирования
    config_hash = get_config_hash(config)
    print(f"Хеш конфигурации: {config_hash}")
    
    # Загружаем документы
    input_dir = config.get("input_dir", "parsed_docs")
    print(f"\nЗагрузка документов из {input_dir}...")
    documents = load_documents(input_dir)
    print(f"Загружено документов: {len(documents)}")
    
    if not documents:
        print("Ошибка: не найдено документов для обработки")
        print(f"Проверьте, что в директории {input_dir} есть .md файлы")
        return
    
    # Инициализируем чанкер
    chunker = MarkdownChunker(config)
    
    # Разбиваем на чанки
    print(f"\nРазбиение на чанки (тип: {chunker.splitter_type})...")
    all_chunks = []
    for doc in documents:
        chunks = chunker.split_markdown(doc["content"], doc["metadata"])
        all_chunks.extend(chunks)
    
    print(f"Всего создано чанков: {len(all_chunks)}")
    
    if not all_chunks:
        print("Ошибка: не создано ни одного чанка")
        return
    
    # Извлекаем тексты для эмбеддингов
    texts = [chunk["text"] for chunk in all_chunks]
    
    # Инициализируем builder эмбеддингов
    embedding_builder = EmbeddingBuilder(config)
    
    # Строим эмбеддинги
    dense_embeddings = None
    sparse_model = None
    
    if embedding_builder.embedding_type in ["dense", "hybrid"]:
        dense_embeddings = embedding_builder.build_dense_embeddings(texts)
    
    if embedding_builder.embedding_type in ["sparse", "hybrid"]:
        sparse_model = embedding_builder.build_sparse_embeddings(texts)
    
    # Сохраняем результаты
    output_dir = config.get("output_dir", "chunks")
    version_dir = save_chunks_and_embeddings(
        all_chunks,
        dense_embeddings,
        sparse_model,
        output_dir,
        config_hash
    )
    
    print("\n" + "=" * 60)
    print("Построение индекса завершено!")
    print(f"Результаты сохранены в: {version_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

