#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Any, Dict


sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

from config import Config
from chunking import ChunkProcessor
from embeddings import EmbeddingGenerator
from tqdm import tqdm
import json


def load_config_with_overrides(args) -> Config:
    config = Config()

    if args.chunk_size:
        config.chunking.chunk_size = args.chunk_size
    if args.chunk_overlap:
        config.chunking.chunk_overlap = args.chunk_overlap
    if args.strategy:
        config.chunking.strategy = args.strategy
    if args.embedding_type:
        config.embeddings.type = args.embedding_type
    if args.embedding_model:
        config.embeddings.model = args.embedding_model

    return config


def extract_metadata_from_markdown(file_path: Path) -> Dict[str, Any]:
    metadata = {
        "source": str(file_path),
        "filename": file_path.name,
    }

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 2:
            frontmatter = parts[1]
            for line in frontmatter.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    if key and value:
                        metadata[key] = value

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Build chunks and embeddings")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild all chunks and embeddings")
    parser.add_argument("--chunk-size", type=int, help="Chunk size in tokens")
    parser.add_argument("--chunk-overlap", type=int, help="Chunk overlap in tokens")
    parser.add_argument("--strategy", choices=["recursive", "markdown", "hybrid"], help="Chunking strategy")
    parser.add_argument("--embedding-type", choices=["dense", "sparse", "hybrid"], help="Embedding type")
    parser.add_argument("--embedding-model", help="Embedding model name")

    args = parser.parse_args()

    config = load_config_with_overrides(args)

    project_root = Path(__file__).parent.parent
    input_dir = project_root / config.documents.get("output_dir", "data/processed")
    chunks_dir = project_root / "data/chunks"
    embeddings_dir = project_root / "data/embeddings"

    chunks_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    md_files = list(input_dir.glob("*.md"))

    if not md_files:
        print(f"No markdown files found in {input_dir}")
        return

    print(f"Found {len(md_files)} documents to process")
    print(f"Configuration:")
    print(f"  Chunking strategy: {config.chunking.strategy}")
    print(f"  Chunk size: {config.chunking.chunk_size} tokens")
    print(f"  Chunk overlap: {config.chunking.chunk_overlap} tokens")
    print(f"  Embedding type: {config.embeddings.type}")
    print(f"  Embedding model: {config.embeddings.model}")

    embedding_generator = EmbeddingGenerator(config)
    max_model_length = embedding_generator.get_max_sequence_length()
    if max_model_length:
        print(f"  Model max sequence length: {max_model_length} tokens")
        if config.chunking.chunk_size > max_model_length:
            print(f"  Warning: chunk_size exceeds model limit. Large chunks will be split.")

    chunk_processor = ChunkProcessor(config, embedding_generator)

    all_chunks = []
    corpus_texts = []

    for file_path in tqdm(md_files, desc="Processing documents"):
        try:
            metadata = extract_metadata_from_markdown(file_path)
            chunks = chunk_processor.chunk_document(file_path, metadata)

            chunks_file = chunks_dir / f"{file_path.stem}_chunks.json"
            chunk_processor.save_chunks(chunks, chunks_file)

            all_chunks.extend(chunks)
            corpus_texts.extend([chunk["text"] for chunk in chunks])

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    print(f"\nTotal chunks created: {len(all_chunks)}")

    print("\nGenerating embeddings...")
    chunks_with_embeddings = embedding_generator.generate_embeddings(all_chunks, corpus_texts)

    embeddings_file = embeddings_dir / "embeddings.json"
    embedding_generator.save_embeddings(chunks_with_embeddings, embeddings_file)

    config_metadata = {
        "chunking": {
            "strategy": config.chunking.strategy,
            "chunk_size": config.chunking.chunk_size,
            "chunk_overlap": config.chunking.chunk_overlap,
        },
        "embeddings": {
            "type": config.embeddings.type,
            "model": config.embeddings.model,
        },
        "total_chunks": len(chunks_with_embeddings),
    }

    metadata_file = embeddings_dir / "index_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(config_metadata, f, indent=2, ensure_ascii=False)

    print(f"\nIndex built successfully")
    print(f"  Chunks: {len(all_chunks)}")
    print(f"  Embeddings saved to: {embeddings_file}")
    print(f"  Metadata saved to: {metadata_file}")


if __name__ == "__main__":
    main()
