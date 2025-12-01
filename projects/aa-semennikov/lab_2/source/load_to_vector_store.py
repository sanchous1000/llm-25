import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import click
import numpy as np
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent))
from source.utils import load_config
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, HnswConfigDiff, OptimizersConfigDiff

class QdrantVectorStore:
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = load_config(config_path)
        self.qdrant_config = self.config.get('qdrant', {})
        self.host = self.qdrant_config.get('host', 'localhost')
        self.port = self.qdrant_config.get('port', 6333)
        self.collection_name = self.qdrant_config.get('collection_name', 'documents')
        self.batch_size = self.qdrant_config.get('batch_size', 100)
        self._connect()
    
    def _connect(self):
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            collections = self.client.get_collections()
        except Exception as e:
            print(f"Ошибка подключения к QDrant: {e}")
            raise
    
    def collection_exists(self):
        """
        Проверяет существование коллекции.
        
        Returns:
            True если коллекция существует
        """
        try:
            collections = self.client.get_collections().collections
            return any(c.name == self.collection_name for c in collections)
        except Exception as e:
            return False

    
    def create_collection(self, vector_size, force_recreate = False):
        """
        Создает коллекцию с заданными параметрами.
        
        Args:
            vector_size: Размерность векторов
            force_recreate: Пересоздать коллекцию если она существует
        """
        exists = self.collection_exists()
        
        if exists and not force_recreate:
            return
        
        if exists and force_recreate:
            self.delete_collection()
        
        hnsw_config = self.qdrant_config.get('hnsw', {})
        hnsw = HnswConfigDiff(
            m=hnsw_config.get('m', 16),
            ef_construct=hnsw_config.get('ef_construction', 100),
            full_scan_threshold=10000,
            max_indexing_threads=0,
        )
        
        distance_metric = self.qdrant_config.get('distance_metric', 'cosine')
        distance_map = {
            'cosine': Distance.COSINE,
            'euclid': Distance.EUCLID,
            'dot': Distance.DOT
        }
        distance = distance_map.get(distance_metric, Distance.COSINE)
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance
            ),
            hnsw_config=hnsw,
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=50000
            )
        )
        self._update_search_params()
    
    def _update_search_params(self):
        """Обновляет параметры поиска (ef_search)."""
        hnsw_config = self.qdrant_config.get('hnsw', {})
        ef_search = hnsw_config.get('ef_search', 100)
        # ef_search устанавливается при поиске, не при создании коллекции
        # Сохраняем значение для использования при поиске
        self.ef_search = ef_search
    
    def delete_collection(self):
        """
        Удаляет коллекцию.
        """
        try:
            if self.collection_exists():
                self.client.delete_collection(collection_name=self.collection_name)
                print(f"Коллекция '{self.collection_name}' удалена")
        except Exception as e:
            print(f"Ошибка при удалении коллекции: {e}")
            raise
    
    def get_collection_info(self):
        """
        Получает информацию о коллекции.
        
        Returns:
            Словарь с информацией о коллекции или None
        """
        try:
            if not self.collection_exists():
                return None
            
            collection_info = self.client.get_collection(self.collection_name)
            
            # Преобразуем в словарь для удобства
            info = {
                'points_count': collection_info.points_count,
                'config': {
                    'params': collection_info.config.params,
                    'hnsw_config': collection_info.config.hnsw_config,
                }
            }
            return info
        except Exception as e:
            print(f"Ошибка при получении информации о коллекции: {e}")
            return None
    
    def load_data(self, version_dir):
        """
        Загружает чанки, эмбеддинги из директории.
        
        Args:
            version_dir: Путь к версионной директории (например, 'data/v_8a56deae')
                        Если None, использует последнюю версию
        """
        start_time = time.time()
        
        # Если version_dir не указан, находим последнюю версию
        if version_dir is None:
            version_dir = self._find_latest_version()
            if version_dir is None:
                raise ValueError("Не найдена версионная директория. Укажите --version или убедитесь, что данные были обработаны.")
        
        data_path = Path(version_dir)
        chunks_file = data_path / self.config['paths']['chunks_dir'] / 'chunks.json'
        if not chunks_file.exists():
            raise ValueError(f"Файл chunks.json не найден: {chunks_file}")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        embeddings_dir = data_path / self.config['paths']['embeddings_dir']
        embeddings_file = embeddings_dir / 'dense_embeddings.npy'
        if not embeddings_file.exists():
            raise ValueError(f"Файл эмбеддингов не найден: {embeddings_file}")
        embeddings = np.load(embeddings_file)
        metadata_file = embeddings_dir / 'metadata.json'
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        elapsed_time = time.time() - start_time
        
        return chunks, embeddings, metadata
    
    def _find_latest_version(self):
        """
        Находит последнюю версию данных.
        
        Returns:
            Путь к последней версии или None
        """
        data_dir = Path(self.config['paths']['output_dir'])
        if not data_dir.exists():
            return None
        version_dirs = sorted(data_dir.glob('v_*'), key=lambda p: p.stat().st_mtime, reverse=True)
        if version_dirs:
            return version_dirs[0]
        
        return None
    
    def upload_embeddings(self, chunks, embeddings, clear_existing = False):
        """
        Загружает эмбеддинги в QDrant.
        
        Args:
            chunks: Список чанков с метаданными
            embeddings: Массив эмбеддингов
            clear_existing: Очистить существующие точки перед загрузкой
        """
        start_time = time.time()
        
        # Проверяем/создаем коллекцию
        vector_size = embeddings.shape[1]
        
        if not self.collection_exists():
            self.create_collection(vector_size, force_recreate=False)
        else:
            # Проверяем размерность
            info = self.get_collection_info()
            if info and info.get('config', {}).get('params'):
                try:
                    params = info['config']['params']

                    if hasattr(params, 'vectors') and params.vectors:
                        vectors_config = params.vectors
                        if isinstance(vectors_config, dict):
                            vector_config = next(iter(vectors_config.values()))
                            existing_size = vector_config.size
                        else:
                            existing_size = vectors_config.size
                    else:
                        existing_size = vector_size
                    
                    if existing_size != vector_size:
                        raise ValueError(
                            "Используйте --rebuild или --drop-and-reindex для пересоздания коллекции"
                        )
                except (AttributeError, KeyError, StopIteration) as e:
                    pass
        
        if clear_existing:
            self.create_collection(vector_size, force_recreate=True)
        
        # Проверяем существующие точки
        info = self.get_collection_info()
        existing_count = info.get('points_count', 0) if info else 0
        
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            payload = {
                'text': chunk['text'],
                'chunk_id': chunk.get('chunk_id', idx),
                'document': chunk.get('metadata', {}).get('filename', ''),
                'headers': chunk.get('metadata', {}).get('headers', {}),
                'start_line': chunk.get('metadata', {}).get('start_line'),
                'end_line': chunk.get('metadata', {}).get('end_line'),
            }
            
            for key in ['file_path', 'title', 'description']:
                if key in chunk.get('metadata', {}):
                    payload[key] = chunk['metadata'][key]
            
            point = PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)
        
        for i in tqdm(range(0, len(points), self.batch_size), desc="Загрузка батчей"):
            batch = points[i:i + self.batch_size]
            
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
            except Exception as e:
                raise
        
        final_info = self.get_collection_info()
        final_count = final_info.get('points_count', 0) if final_info else 0
        
        elapsed_time = time.time() - start_time


@click.command()
@click.option('--version', help='Версия данных для загрузки (например, data/v_8a56deae). Если не указано, использует последнюю')
@click.option('--rebuild', is_flag=True, help='Пересоздать коллекцию с существующими параметрами')
@click.option('--drop-and-reindex', is_flag=True, help='Удалить коллекцию и создать заново c новым индексом')
@click.option('--reset', is_flag=True, help='Быстрый сброс: удалить и создать пустую коллекцию')

def main(version, rebuild, drop_and_reindex, reset):    
    total_start_time = time.time()
    
    try:
        store = QdrantVectorStore(config_path='config.yaml')
        
        if reset:
            print("Быстрый сброс коллекции...")
            
            if store.collection_exists():
                store.delete_collection()
            
            # Определяем размерность векторов из последней версии данных
            try:
                if version:
                    data_path = Path(version)
                else:
                    data_path = store._find_latest_version()
                
                if not data_path or not data_path.exists():
                    raise ValueError("Не найдена директория с данными для определения размерности")
                
                # Загружаем только метаданные для определения размерности
                embeddings_dir = data_path / store.config['paths']['embeddings_dir']
                metadata_file = embeddings_dir / 'metadata.json'
                
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Пытаемся получить размерность из метаданных
                    vector_size = None
                    if 'config' in metadata and 'embeddings' in metadata['config']:
                        dense_config = metadata['config']['embeddings'].get('dense', {})
                        if 'dimension' in dense_config and dense_config['dimension']:
                            vector_size = dense_config['dimension']
                    
                    # Если не нашли в метаданных, загружаем эмбеддинги
                    if not vector_size:
                        embeddings_file = embeddings_dir / 'dense_embeddings.npy'
                        if embeddings_file.exists():
                            embeddings = np.load(embeddings_file)
                            vector_size = embeddings.shape[1]
                else:
                    # Загружаем эмбеддинги напрямую
                    embeddings_file = embeddings_dir / 'dense_embeddings.npy'
                    if not embeddings_file.exists():
                        raise ValueError(f"Файл эмбеддингов не найден: {embeddings_file}")
                    
                    embeddings = np.load(embeddings_file)
                    vector_size = embeddings.shape[1]
                
                # Создаем пустую коллекцию
                store.create_collection(vector_size, force_recreate=False)
                
                print("Коллекция успешно сброшена")
                print(f"Размерность векторов: {vector_size}")
                print(f"Коллекция готова к загрузке данных")
                
            except Exception as e:
                print(f"Ошибка при сбросе коллекции: {e}")
                raise
            
            return
        
        # Загружаем данные
        print("Загрузка чанков и эмбеддингов...")
        chunks, embeddings, metadata = store.load_data(version_dir=version)
        
        print(f"Загружено: {len(chunks)} чанков, эмбеддинги {embeddings.shape}")
        
        # Определяем режим загрузки
        if drop_and_reindex:
            print("Режим: drop-and-reindex (полная пересборка)")
            store.create_collection(embeddings.shape[1], force_recreate=True)
            clear_existing = False  # Уже пересоздали
        elif rebuild:
            print("Режим: rebuild (обновление данных)")
            clear_existing = True
        else:
            print("Режим: инкрементальная загрузка")
            clear_existing = False
        
        # Загружаем эмбеддинги
        store.upload_embeddings(chunks, embeddings, clear_existing=clear_existing)
        
        # Выводим статистику
        print("\n" + "=" * 60)
        print("Загрузка завершена")
        total_elapsed_time = time.time() - total_start_time
        print(f"Общее время выполнения: {total_elapsed_time:.2f} секунд ({total_elapsed_time/60:.2f} минут)")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()