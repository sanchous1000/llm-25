import argparse
import json
from typing import List, Dict, Any, Tuple

from tqdm import tqdm
from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter

import hashlib
from pathlib import Path


def build_id(s: str):
    return hashlib.sha256(s.encode()).hexdigest()


def iter_md_files(root: str) -> List[str]:
    root_dir = Path(root)
    md_files = sorted(root_dir.rglob("*.md"))
    return [str(md_file.absolute()) for md_file in md_files]


def build_chunks_for_doc(
        raw: str,
        md_path: str,
        levels: Tuple[int, ...],
        chunk_size: int,
        overlap: int,
) -> List[Dict[str, Any]]:
    # 1) Сплит по заголовкам с учётом уровней (h1..h6)
    headers_to_split_on = [("#" * lvl, f"H{lvl}") for lvl in sorted(set(levels)) if 1 <= lvl <= 6]
    assert len(headers_to_split_on) > 0, "Specify levels to split on"

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  # сохраняем заголовки в тексте секции
    )
    docs = header_splitter.split_text(raw)

    # 2) Дробление секций на чанки с MarkdownTextSplitter.
    md_splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    samples: List[Dict[str, Any]] = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}

        # Соберём путь секции из метаданных MarkdownHeaderTextSplitter.
        # Ключи обычно вида "H1","H2","H3", ..., берём по возрастанию уровня.
        section_path = []
        for k in sorted(meta.keys(), key=lambda x: (len(x), x)):
            kl = k.lower()
            if kl.startswith("h") and kl[1:].isdigit():
                section_path.append(meta[k])

        heading = "/".join(section_path) if section_path else Path(md_path).stem
        level = len(section_path) if section_path else 1

        # Реальный чанкинг по Markdown с учётом код-блоков и маркдаун-структуры
        parts = md_splitter.split_text(doc.page_content)

        for chunk_text in parts:
            ct = (chunk_text or "").strip()
            if not ct:
                continue

            samples.append({
                "id": build_id(md_path),
                "heading": heading,
                "level": level,
                "text": ct,
                "file_path": md_path,
            })

    return samples


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Parse Markdown into RAG dataset: split by headings, ignore headings inside code.")
    ap.add_argument("--input_dir", required=True, help="Root directory with .md files (processed recursively)")
    ap.add_argument("--output_jsonl", required=True, help="Output JSONL path")
    ap.add_argument("--levels", default="1,2,3", help="Heading levels to split on, e.g. '1,2,3'")
    ap.add_argument("--chunk_size", type=int, default=512, help="Target chunk size in tokens")
    ap.add_argument("--overlap", type=int, default=64, help="Token overlap between chunks")
    args = ap.parse_args()

    input_dir_abs = Path(args.input_dir).resolve()

    levels = tuple(int(x.strip()) for x in args.levels.split(",") if x.strip())
    files = iter_md_files(args.input_dir)

    all_rows: List[Dict[str, Any]] = []
    for _, file in tqdm(enumerate(files), total=len(files)):
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            raw_doc = f.read()
        relative_path = Path(file).relative_to(input_dir_abs)
        normalized_path = str(relative_path).replace('\\', '/')
        rows = build_chunks_for_doc(
            raw=raw_doc,
            md_path=normalized_path,
            levels=levels,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        all_rows.extend(rows)

    write_jsonl(args.output_jsonl, all_rows)
    print(f"Wrote {len(all_rows)} records to {args.output_jsonl}")


if __name__ == "__main__":
    main()