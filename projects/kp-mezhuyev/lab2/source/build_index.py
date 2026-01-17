"""
Скрипт для построения индекса чанков и эмбеддингов.

Поддерживает:
- Различные стратегии разбиения (recursive, markdown, hybrid)
- Тип эмбеддингов только dense
- Версионирование артефактов
- Повторный запуск без ручной очистки
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import sys
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

script_dir = Path(__file__).parent
lab2_dir = script_dir.parent
source_dir = lab2_dir / "source"
sys.path.insert(0, str(source_dir))

from chunking import Chunk, get_splitter
from config_utils import load_config
from embeddings import format_text_for_e5, get_embedder


def parse_markdown_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Парсит YAML фронтматтер из Markdown файла.
    
    Returns:
        Кортеж (метаданные, тело документа).
    """
    if not content.startswith("---"):
        return {}, content
    
    # Находим конец фронтматтера
    end_idx = content.find("---", 3)
    if end_idx == -1:
        return {}, content
    
    frontmatter_text = content[3:end_idx].strip()
    body = content[end_idx + 3:].strip()
    
    # Парсим YAML
    try:
        metadata = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError:
        metadata = {}
    
    return metadata, body


def load_processed_documents(processed_dir: Path) -> list[dict[str, Any]]:
    """Загружает обработанные Markdown документы."""
    documents = []
    
    for md_file in tqdm(
        list(processed_dir.rglob("*.md")),
        desc="Loading documents",
    ):
        try:
            content = md_file.read_text(encoding="utf-8")
            metadata, body = parse_markdown_frontmatter(content)
            
            # Добавляем путь к метаданным
            rel_path = md_file.relative_to(processed_dir)
            metadata["source_path"] = str(rel_path)
            metadata["file_path"] = str(md_file)
            
            documents.append({
                "metadata": metadata,
                "content": body,
            })
        except Exception as e:
            print(f"Error loading {md_file}: {e}")
            continue
    
    return documents


def create_version_hash(config: dict[str, Any]) -> str:
    """Создает хеш версии на основе параметров конфигурации."""
    # Извлекаем ключевые параметры для версионирования
    version_params = {
        "chunking": config.get("chunking", {}),
        "embeddings": {
            "vectorization_type": config.get("embeddings", {}).get("vectorization_type"),
            "model": config.get("embeddings", {}).get("dense", {}).get("model"),
        },
    }
    
    # Сериализуем в строку и создаем хеш
    version_str = json.dumps(version_params, sort_keys=True)
    version_hash = hashlib.sha256(version_str.encode()).hexdigest()[:12]
    
    return version_hash


def build_index(
    config: dict[str, Any],
    rebuild: bool = False,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Строит индекс чанков и эмбеддингов.
    
    Args:
        config: Конфигурация.
        rebuild: Пересобрать индекс даже если он существует.
        output_dir: Директория для сохранения (по умолчанию из config).
    
    Returns:
        Словарь с метаданными построенного индекса.
    """
    # Загружаем переменные окружения
    load_dotenv()
    
    # Определяем базовую директорию lab2
    script_dir = Path(__file__).parent
    lab2_dir = script_dir.parent
    
    # Определяем пути относительно lab2_dir
    paths_config = config.get("paths", {})
    processed_dir = (lab2_dir / paths_config.get("processed_dir", "data/processed")).resolve()
    chunks_dir = (lab2_dir / paths_config.get("chunks_dir", "data/chunks")).resolve()
    embeddings_dir = (lab2_dir / paths_config.get("embeddings_dir", "data/embeddings")).resolve()
    index_dir = (lab2_dir / paths_config.get("index_dir", "data/index")).resolve()
    
    if output_dir:
        output_dir = Path(output_dir).resolve()
        chunks_dir = output_dir / "chunks"
        embeddings_dir = output_dir / "embeddings"
        index_dir = output_dir / "index"
    
    # Создаем версию
    version_hash = create_version_hash(config)
    versioning_config = config.get("versioning", {})
    
    if versioning_config.get("version_format") == "timestamp":
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        version = version_hash
    
    # Проверяем, существует ли уже индекс с такой версией
    version_dir = index_dir / version
    if version_dir.exists() and not rebuild:
        print(f"Index version {version} already exists. Use --rebuild to rebuild.")
        # Загружаем существующие метаданные
        metadata_file = version_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
    
    print(f"Building index version: {version}")
    
    # Создаем директории
    version_chunks_dir = chunks_dir / version
    version_embeddings_dir = embeddings_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    version_chunks_dir.mkdir(parents=True, exist_ok=True)
    version_embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Загружаем документы
    print("Loading processed documents...")
    documents = load_processed_documents(processed_dir)
    print(f"Loaded {len(documents)} documents")
    
    # Создаем сплиттер
    chunking_config = config.get("chunking", {})
    splitter_type = chunking_config.get("splitter_type", "markdown")
    splitter = get_splitter(splitter_type, config)
    
    # Разбиваем на чанки
    print(f"Splitting documents using {splitter_type} splitter...")
    all_chunks: list[Chunk] = []
    
    for doc in tqdm(documents, desc="Splitting documents"):
        metadata = doc["metadata"]
        content = doc["content"]
        
        chunks = splitter.split(content, metadata)
        all_chunks.extend(chunks)
    
    print(f"Created {len(all_chunks)} chunks")
    
    # Сохраняем чанки
    chunks_file = version_chunks_dir / "chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    
    # Подготавливаем тексты для эмбеддингов
    texts = [chunk.text for chunk in all_chunks]
    
    # Проверяем, нужна ли специальная обработка для e5 моделей
    embeddings_config = config.get("embeddings", {})
    dense_config = embeddings_config.get("dense", {})
    model_name = dense_config.get("model", "")
    
    is_e5_model = "e5" in model_name.lower()
    
    if is_e5_model:
        print("Formatting texts for e5 model (adding 'passage:' prefix)...")
        texts = [format_text_for_e5(text) for text in texts]
    
    # Создаем эмбеддинги
    print("Creating embeddings...")
    vectorization_type = embeddings_config.get("vectorization_type", "dense")
    
    # Используем только dense эмбеддинги
    if vectorization_type != "dense":
        raise ValueError(
            f"Only 'dense' embeddings are supported. "
            f"Got: {vectorization_type}"
        )
    
    embedder = get_embedder(vectorization_type, config)
    embeddings = embedder.embed(texts)
    
    print(f"Created embeddings with shape: {embeddings.shape}")
    
    # Сохраняем эмбеддинги
    embeddings_file = version_embeddings_dir / "embeddings.npy"
    np.save(embeddings_file, embeddings)
    
    # Сохраняем метаданные
    metadata = {
        "version": version,
        "version_hash": version_hash,
        "created_at": datetime.now().isoformat(),
        "config": config,
        "statistics": {
            "num_documents": len(documents),
            "num_chunks": len(all_chunks),
            "embedding_dimension": embedder.get_dimension(),
            "embedding_shape": list(embeddings.shape),
            "splitter_type": splitter_type,
            "vectorization_type": vectorization_type,
            "model_name": model_name,
        },
        "paths": {
            "chunks_file": str(chunks_file.relative_to(lab2_dir)),
            "embeddings_file": str(embeddings_file.relative_to(lab2_dir)),
        },
    }
    
    metadata_file = version_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\nIndex built successfully!")
    print(f"Version: {version}")
    print(f"Chunks: {len(all_chunks)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Metadata saved to: {metadata_file}")
    
    return metadata


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="Build index of chunks and embeddings",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild index even if it exists",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config paths)",
    )
    
    args = parser.parse_args()
    
    # Определяем базовую директорию lab2
    script_dir = Path(__file__).parent
    lab2_dir = script_dir.parent
    
    # Загружаем конфигурацию
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = lab2_dir / args.config
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_config(config_path)
    
    # Определяем output_dir
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Строим индекс
    build_index(config, rebuild=args.rebuild, output_dir=output_dir)


if __name__ == "__main__":
    main()