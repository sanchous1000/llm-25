import json
import yaml
from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
from sentence_transformers import SentenceTransformer
import numpy as np
from utils import load_config, get_config_hash


def parse_markdown_frontmatter(md_path: str) -> tuple[Dict, str]:
    with open(md_path, encoding="utf-8") as f:
        content = f.read()
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            meta_str = content[3:end]
            body = content[end + 3:].strip()
            try:
                meta = yaml.safe_load(meta_str) or {}
            except yaml.YAMLError:
                meta = {}
            return meta, body
    return {}, content


def split_markdown(md_text: str, meta: Dict, config) -> List[Dict]:
    headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_chunks = md_splitter.split_text(md_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
        separators=["\n\n", "\n", " ", ""]
    )
    final_chunks = []

    arxiv_id = meta.get("arxiv_id", "unknown")
    title = meta.get("title", "unknown")
    source = meta.get("source", "unknown")
    page = meta.get("page",  "unknown")

    if config["chunking"]["splitter"] == 'markdown_recursive':
        for chunk in md_chunks:
            sub_chunks = text_splitter.split_text(chunk.page_content)
            for sc in sub_chunks:
                headers = " | ".join([f"{k}: {v}" for k, v in chunk.metadata.items()])
                text = f"[{headers}] {sc}" if config["chunking"]["include_headers_in_chunk"] else sc
                final_chunks.append({
                    "text": text,
                    "source": source,
                    "arxiv_id": arxiv_id,
                    "page": page,
                    "title": title
                })
    else:
        for chunk in md_chunks:
            final_chunks.append({
                "text": chunk.page_content,
                "source": source,
                "arxiv_id": arxiv_id,
                "page": page,
                "title": title
            })

    return final_chunks


def main():
    config = load_config()
    config_hash = get_config_hash(config)
    output_dir = Path(f"../artifacts/index_{config_hash}")
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = output_dir / "chunks.json"
    base_data_dir = Path("../data/processed/arxiv").resolve()

    print("Сбор чанков...")
    all_chunks = []

    for arxiv_dir in base_data_dir.iterdir():
        if not arxiv_dir.is_dir():
            continue

        arxiv_id = arxiv_dir.name
        print(f"Обработка документа: {arxiv_id}")

        page_files = sorted(arxiv_dir.glob("page_*.md"))
        if not page_files:
            print(f"  Пропущено: нет файлов page_*.md в {arxiv_dir}")
            continue

        for md_file in page_files:
            abs_path = str(md_file.resolve())

            meta, body = parse_markdown_frontmatter(abs_path)

            if not meta.get("arxiv_id"):
                meta["arxiv_id"] = arxiv_id

            chunks = split_markdown(body, meta, config)

            chunk_base_id = f"{arxiv_id}_{md_file.stem}"

            for chunk in chunks:
                chunk["id"] = chunk_base_id
                all_chunks.append(chunk)

    print(f"Всего чанков: {len(all_chunks)}")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("Генерация эмбеддингов...")
    model = SentenceTransformer(config["embedding"]["model"])
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    np.save(output_dir / "embeddings.npy", embeddings)

    print(f"Индекс сохранён в {output_dir}")


if __name__ == '__main__':
    main()