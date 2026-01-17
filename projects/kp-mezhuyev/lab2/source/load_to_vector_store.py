"""
Скрипт для загрузки чанков и эмбеддингов в Elasticsearch.

Поддерживает:
- Создание индекса с векторным поиском (dense_vector)
- Загрузку чанков с метаданными
- HNSW параметры для оптимизации поиска
- Пересборку/переиндексацию при изменении параметров
"""
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm

from config_utils import load_config
from es_utils import get_es_client


def load_index_data(version: str, lab2_dir: Path) -> tuple[list[dict], np.ndarray, dict]:
    """Загружает чанки, эмбеддинги и метаданные индекса.
    
    Args:
        version: Версия индекса.
        lab2_dir: Базовая директория lab2.
    
    Returns:
        Кортеж (чанки, эмбеддинги, метаданные).
    """
    index_dir = lab2_dir / "data" / "index" / version
    chunks_dir = lab2_dir / "data" / "chunks" / version
    embeddings_dir = lab2_dir / "data" / "embeddings" / version
    
    # Загружаем метаданные
    metadata_file = index_dir / "metadata.json"
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Загружаем чанки
    chunks_file = chunks_dir / "chunks.jsonl"
    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    
    # Загружаем эмбеддинги
    embeddings_file = embeddings_dir / "embeddings.npy"
    embeddings = np.load(embeddings_file)
    
    print(f"Loaded {len(chunks)} chunks and embeddings with shape {embeddings.shape}")
    
    return chunks, embeddings, metadata


def create_index(
    es_client: Elasticsearch,
    index_name: str,
    embedding_dim: int,
    config: dict[str, Any],
) -> None:
    """Создает индекс в Elasticsearch с векторным поиском.
    
    Args:
        es_client: Клиент Elasticsearch.
        index_name: Имя индекса.
        embedding_dim: Размерность эмбеддингов.
        config: Конфигурация Elasticsearch.
    """
    es_config = config.get("elasticsearch", {})
    hnsw_config = es_config.get("hnsw", {})
    
    # Определяем настройки индекса
    index_settings = {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    }
    
    # Маппинг для векторного поля и метаданных
    mappings = {
        "properties": {
            "embedding": {
                "type": "dense_vector",
                "dims": embedding_dim,
                "index": True,
                "similarity": "cosine",
                "index_options": {
                    "type": "hnsw",
                    "m": hnsw_config.get("m", 16),
                    "ef_construction": hnsw_config.get("ef_construction", 100),
                },
            },
            "text": {"type": "text"},
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "source_path": {"type": "keyword"},
            "title": {"type": "text"},
            "header": {"type": "text"},
            "header_level": {"type": "integer"},
            "chunk_index": {"type": "integer"},
            "token_count": {"type": "integer"},
            "metadata": {
                "type": "object",
                "enabled": True,  # Индексируем все метаданные
            },
        }
    }
    
    # Удаляем индекс, если существует
    if es_client.indices.exists(index=index_name):
        print(f"Index '{index_name}' already exists. Deleting...")
        es_client.indices.delete(index=index_name)
    
    # Создаем индекс
    es_client.indices.create(
        index=index_name,
        settings=index_settings,
        mappings=mappings,
    )
    
    print(f"Index '{index_name}' created successfully")


def load_to_elasticsearch(
    chunks: list[dict],
    embeddings: np.ndarray,
    es_client: Elasticsearch,
    index_name: str,
    batch_size: int = 100,
) -> None:
    """Загружает чанки и эмбеддинги в Elasticsearch.
    
    Args:
        chunks: Список чанков.
        embeddings: Массив эмбеддингов.
        es_client: Клиент Elasticsearch.
        index_name: Имя индекса.
        batch_size: Размер батча для загрузки.
    """
    def generate_docs():
        """Генератор документов для bulk загрузки."""
        for i, chunk in enumerate(chunks):
            doc = {
                "_index": index_name,
                "_id": chunk["id"],
                "_source": {
                    "embedding": embeddings[i].tolist(),
                    "text": chunk["text"],
                    "chunk_id": chunk["id"],
                    "doc_id": chunk.get("doc_id", ""),
                    "source_path": chunk.get("source_path", ""),
                    "title": chunk.get("title", ""),
                    "header": chunk.get("header", ""),
                    "header_level": chunk.get("header_level", 0),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "token_count": chunk.get("token_count", 0),
                    "metadata": chunk.get("metadata", {}),
                },
            }
            yield doc
    
    # Загружаем батчами
    print(f"Loading {len(chunks)} documents to Elasticsearch...")
    success, failed = bulk(
        es_client,
        generate_docs(),
        chunk_size=batch_size,
        request_timeout=60,
    )
    
    print(f"Loaded {success} documents successfully")
    if failed:
        print(f"Failed to load {len(failed)} documents")
        for item in failed[:5]:  # Показываем первые 5 ошибок
            print(f"  Error: {item}")


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="Load chunks and embeddings to Elasticsearch",
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Version of index to load (if not specified, uses latest)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild index even if it exists",
    )
    parser.add_argument(
        "--drop-and-reindex",
        action="store_true",
        help="Drop existing index and reindex",
    )
    
    args = parser.parse_args()
    
    # Определяем базовую директорию lab2
    script_dir = Path(__file__).parent
    lab2_dir = script_dir.parent
    
    # Загружаем конфигурацию
    config_path = lab2_dir / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_config(config_path)
    es_config = config.get("elasticsearch", {})
    
    # Определяем версию индекса
    if args.version:
        version = args.version
    else:
        # Используем последнюю версию (по времени модификации)
        index_dir = lab2_dir / "data" / "index"
        version_dirs = [d for d in index_dir.iterdir() if d.is_dir()]
        if not version_dirs:
            raise ValueError("No index versions found. Run build_index.py first.")
        # Сортируем по времени модификации (самая новая последняя)
        version_dirs.sort(key=lambda d: d.stat().st_mtime)
        version = version_dirs[-1].name
        print(f"Using latest version: {version}")
    
    # Загружаем данные
    chunks, embeddings, metadata = load_index_data(version, lab2_dir)
    
    # Подключаемся к Elasticsearch
    es_client, es_url = get_es_client(es_config)
    print(f"Connected to Elasticsearch: {es_url}")
    
    print(f"Connected to Elasticsearch: {es_client.info()['cluster_name']}")
    
    # Создаем индекс
    index_name = es_config.get("index_name", "fastapi_docs")
    embedding_dim = metadata["statistics"]["embedding_dimension"]
    
    if args.drop_and_reindex or args.rebuild:
        create_index(es_client, index_name, embedding_dim, config)
    elif not es_client.indices.exists(index=index_name):
        create_index(es_client, index_name, embedding_dim, config)
    else:
        print(f"Index '{index_name}' already exists. Use --rebuild to rebuild.")
        return
    
    # Загружаем данные
    batch_size = es_config.get("batch_size", 100)
    load_to_elasticsearch(chunks, embeddings, es_client, index_name, batch_size)
    
    # Обновляем alias для версионирования
    alias_name = f"{index_name}_v{version}"
    if es_client.indices.exists_alias(name=alias_name):
        es_client.indices.delete_alias(index=index_name, name=alias_name)
    es_client.indices.put_alias(index=index_name, name=alias_name)
    
    print(f"\n[OK] Index loaded successfully!")
    print(f"  Index name: {index_name}")
    print(f"  Alias: {alias_name}")
    print(f"  Documents: {len(chunks)}")
    print(f"  Embedding dimension: {embedding_dim}")


if __name__ == "__main__":
    main()
