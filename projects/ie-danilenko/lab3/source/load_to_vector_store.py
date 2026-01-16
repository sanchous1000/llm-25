"""
Скрипт для загрузки чанков и эмбеддингов в векторное хранилище Faiss.
Выполняет задание 4: развертывание векторного хранилища и загрузка индекса.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from utils import load_chunks
import argparse


class FaissVectorStore:
    """Класс для работы с векторным хранилищем Faiss."""
    
    def __init__(self, index_dir: str = "faiss_index", config: Optional[Dict[str, Any]] = None):
        """
        Инициализация векторного хранилища.
        
        Args:
            index_dir: Директория для сохранения индекса
            config: Конфигурация индекса (HNSW параметры)
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        self.config = config or {}
        self.index_config = self.config.get("index", {})
        
        # HNSW параметры
        self.M = self.index_config.get("M", 32)  # Количество связей
        self.ef_construction = self.index_config.get("ef_construction", 200)  # Параметр построения
        self.ef_search = self.index_config.get("ef_search", 50)  # Параметр поиска
        
        self.index = None
        self.chunks = None
        self.dense_embeddings = None
        self.sparse_model = None
        self.dimension = None
    
    def create_index(self, dimension: int, index_type: str = "HNSW"):
        """
        Создает индекс Faiss.
        
        Args:
            dimension: Размерность векторов
            index_type: Тип индекса (HNSW, Flat, IVF)
        """
        self.dimension = dimension
        
        if index_type == "HNSW":
            # HNSW индекс для приближенного поиска
            self.index = faiss.IndexHNSWFlat(dimension, self.M)
            self.index.hnsw.efConstruction = self.ef_construction
            self.index.hnsw.efSearch = self.ef_search
            print(f"Создан HNSW индекс: dimension={dimension}, M={self.M}, ef_construction={self.ef_construction}")
        elif index_type == "Flat":
            # Точный поиск (L2 расстояние)
            self.index = faiss.IndexFlatL2(dimension)
            print(f"Создан Flat индекс: dimension={dimension}")
        elif index_type == "IVF":
            # IVF индекс с квантованием
            nlist = self.index_config.get("nlist", 100)  # Количество кластеров
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            print(f"Создан IVF индекс: dimension={dimension}, nlist={nlist}")
        else:
            raise ValueError(f"Неизвестный тип индекса: {index_type}")
    
    def load_chunks_and_embeddings(self, chunks_dir: str, version: Optional[str] = None):
        """
        Загружает чанки и эмбеддинги из директории.
        
        Args:
            chunks_dir: Директория с чанками
            version: Версия (хеш конфигурации), если None - загружает последнюю
        """
        chunks_path = Path(chunks_dir)
        
        if not chunks_path.exists():
            raise FileNotFoundError(f"Директория {chunks_dir} не найдена")
        
        # Находим версию
        if version:
            version_dir = chunks_path / f"v_{version}"
        else:
            # Находим последнюю версию
            version_dirs = sorted(chunks_path.glob("v_*"), key=lambda x: x.stat().st_mtime, reverse=True)
            if not version_dirs:
                raise FileNotFoundError(f"Не найдено версий в {chunks_dir}")
            version_dir = version_dirs[0]
            version = version_dir.name.replace("v_", "")
        
        print(f"Загрузка данных из версии: {version_dir}")
        
        self.chunks = load_chunks(chunks_dir, version)
        
        # Загружаем dense эмбеддинги
        dense_file = version_dir / "dense_embeddings.pkl"
        if dense_file.exists():
            with open(dense_file, 'rb') as f:
                self.dense_embeddings = pickle.load(f)
            print(f"Загружены dense эмбеддинги: {len(self.dense_embeddings)} векторов")
            self.dimension = len(self.dense_embeddings[0]) if self.dense_embeddings else None
        else:
            print("Warning: dense эмбеддинги не найдены")
        
        # Загружаем sparse модель (BM25)
        sparse_file = version_dir / "sparse_model.pkl"
        if sparse_file.exists():
            with open(sparse_file, 'rb') as f:
                self.sparse_model = pickle.load(f)
            print("Загружена sparse модель (BM25)")
        else:
            print("Warning: sparse модель не найдена")
        
        return version
    
    def build_index(self, index_type: str = "HNSW", rebuild: bool = False):
        """
        Строит индекс из загруженных эмбеддингов.
        
        Args:
            index_type: Тип индекса
            rebuild: Пересобрать индекс даже если он существует
        """
        if self.dense_embeddings is None:
            raise ValueError("Dense эмбеддинги не загружены")
        
        if self.dimension is None:
            self.dimension = len(self.dense_embeddings[0])
        
        # Проверяем существующий индекс
        index_file = self.index_dir / "index.faiss"
        if index_file.exists() and not rebuild:
            print(f"Индекс уже существует: {index_file}")
            print("Используйте --rebuild для пересборки")
            return
        
        # Создаем новый индекс
        self.create_index(self.dimension, index_type)
        
        # Преобразуем в numpy массив
        embeddings_array = np.array(self.dense_embeddings, dtype=np.float32)
        
        # Обучаем индекс (для IVF)
        if index_type == "IVF":
            print("Обучение IVF индекса...")
            self.index.train(embeddings_array)
        
        # Добавляем векторы
        print(f"Добавление {len(embeddings_array)} векторов в индекс...")
        self.index.add(embeddings_array)
        
        print(f"Индекс построен: {self.index.ntotal} векторов")
    
    def save_index(self):
        """Сохраняет индекс и метаданные."""
        if self.index is None:
            raise ValueError("Индекс не создан")
        
        # Сохраняем индекс
        index_file = self.index_dir / "index.faiss"
        faiss.write_index(self.index, str(index_file))
        print(f"Индекс сохранен: {index_file}")
        
        # Сохраняем метаданные
        metadata = {
            "dimension": self.dimension,
            "num_vectors": self.index.ntotal,
            "index_type": type(self.index).__name__,
            "has_sparse": self.sparse_model is not None,
            "num_chunks": len(self.chunks) if self.chunks else 0,
            "config": {
                "M": self.M,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search
            }
        }
        
        metadata_file = self.index_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Метаданные сохранены: {metadata_file}")
        
        # Сохраняем чанки (ссылка на исходные данные)
        if self.chunks:
            chunks_ref_file = self.index_dir / "chunks_ref.json"
            with open(chunks_ref_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "num_chunks": len(self.chunks),
                    "chunks": self.chunks
                }, f, ensure_ascii=False, indent=2)
            print(f"Ссылки на чанки сохранены: {chunks_ref_file}")
    
    def load_index(self):
        """Загружает существующий индекс."""
        index_file = self.index_dir / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(f"Индекс не найден: {index_file}")
        
        self.index = faiss.read_index(str(index_file))
        self.dimension = self.index.d
        print(f"Индекс загружен: {self.index.ntotal} векторов, dimension={self.dimension}")
        
        # Загружаем метаданные
        metadata_file = self.index_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"Метаданные загружены: {metadata}")
        
        # Загружаем чанки
        chunks_ref_file = self.index_dir / "chunks_ref.json"
        if chunks_ref_file.exists():
            with open(chunks_ref_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            self.chunks = chunks_data.get("chunks", [])
            print(f"Загружено ссылок на чанки: {len(self.chunks)}")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Поиск похожих векторов.
        
        Args:
            query_vector: Вектор запроса
            k: Количество результатов
        
        Returns:
            Список результатов с индексами и расстояниями
        """
        if self.index is None:
            raise ValueError("Индекс не загружен")
        
        # Устанавливаем ef_search для HNSW
        if isinstance(self.index, faiss.IndexHNSWFlat):
            self.index.hnsw.efSearch = self.ef_search
        
        # Поиск
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query_vector, k)
        
        # Формируем результаты
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx != -1:  # -1 означает отсутствие результата
                result = {
                    "index": int(idx),
                    "distance": float(dist),
                    "chunk": self.chunks[idx] if self.chunks and idx < len(self.chunks) else None
                }
                results.append(result)
        
        return results
    
    def reset(self):
        """Сбрасывает индекс (удаляет все файлы)."""
        import shutil
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
            self.index_dir.mkdir(exist_ok=True)
            print(f"Индекс сброшен: {self.index_dir}")
        self.index = None
        self.chunks = None
        self.dense_embeddings = None
        self.sparse_model = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Загрузка индекса в Faiss')
    parser.add_argument('--chunks-dir', type=str, default='chunks',
                       help='Директория с чанками (по умолчанию: chunks)')
    parser.add_argument('--index-dir', type=str, default='faiss_index',
                       help='Директория для индекса (по умолчанию: faiss_index)')
    parser.add_argument('--version', type=str, default=None,
                       help='Версия чанков для загрузки (по умолчанию: последняя)')
    parser.add_argument('--index-type', type=str, choices=['HNSW', 'Flat', 'IVF'], default='HNSW',
                       help='Тип индекса (по умолчанию: HNSW)')
    parser.add_argument('--rebuild', action='store_true',
                       help='Пересобрать индекс даже если он существует')
    parser.add_argument('--config', type=str, default='config/config.json',
                       help='Путь к конфигурационному файлу')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Загрузка индекса в Faiss")
    print("=" * 60)
    
    # Загружаем конфигурацию
    config = {}
    config_file = Path(args.config)
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"Конфигурация загружена: {config_file}")
    else:
        print(f"Warning: конфигурация не найдена, используются значения по умолчанию")
    
    # Добавляем конфигурацию индекса, если её нет
    if "index" not in config:
        config["index"] = {
            "M": 32,
            "ef_construction": 200,
            "ef_search": 50
        }
    
    # Инициализируем хранилище
    vector_store = FaissVectorStore(args.index_dir, config)
    
    # Загружаем чанки и эмбеддинги
    try:
        version = vector_store.load_chunks_and_embeddings(args.chunks_dir, args.version)
        print(f"Версия данных: {version}")
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        exit()
    
    # Строим индекс
    try:
        vector_store.build_index(args.index_type, rebuild=args.rebuild)
    except ValueError as e:
        print(f"Ошибка: {e}")
        exit()
    
    # Сохраняем индекс
    vector_store.save_index()
    
    print("\n" + "=" * 60)
    print("Загрузка индекса завершена!")
    print(f"Индекс сохранен в: {args.index_dir}")
    print("=" * 60)