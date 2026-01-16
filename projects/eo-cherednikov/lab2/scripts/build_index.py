import argparse
import json
from typing import List, Dict, Any, Tuple, Optional

from tqdm import tqdm
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter, 
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter
)

import hashlib
from pathlib import Path

try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: pypdf not installed. PDF parsing will be unavailable. Install with: pip install pypdf")


def build_id(s: str):
    return hashlib.sha256(s.encode()).hexdigest()


def iter_md_files(root: str) -> List[str]:
    root_dir = Path(root)
    md_files = sorted(root_dir.rglob("*.md"))
    return [str(md_file.absolute()) for md_file in md_files]


def iter_pdf_files(root: str) -> List[str]:
    if not PDF_AVAILABLE:
        return []
    root_dir = Path(root)
    pdf_files = sorted(root_dir.rglob("*.pdf"))
    return [str(pdf_file.absolute()) for pdf_file in pdf_files]


def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    if not PDF_AVAILABLE:
        raise ImportError("pypdf is required for PDF parsing. Install with: pip install pypdf")
    
    pages = []
    try:
        reader = PdfReader(file_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text.strip():  # Пропускаем пустые страницы
                pages.append({
                    'page_number': page_num,
                    'text': text.strip(),
                    'file_path': file_path
                })
    except Exception as e:
        print(f"Error parsing PDF {file_path}: {e}")
        return []
    
    return pages


def build_chunks_for_pdf(
        pdf_path: str,
        chunk_size: int,
        overlap: int,
) -> List[Dict[str, Any]]:
    pages = parse_pdf(pdf_path)
    if not pages:
        return []
    
    # Используем RecursiveCharacterTextSplitter для PDF
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    samples: List[Dict[str, Any]] = []
    
    for page_data in pages:
        page_num = page_data['page_number']
        page_text = page_data['text']
        
        # Разбиваем страницу на чанки
        chunks = text_splitter.split_text(page_text)
        
        for chunk_idx, chunk_text in enumerate(chunks):
            ct = (chunk_text or "").strip()
            if not ct:
                continue
            
            # Создаем заголовок на основе номера страницы
            heading = f"Page {page_num}"
            if len(chunks) > 1:
                heading += f" (Part {chunk_idx + 1})"
            
            samples.append({
                "id": build_id(f"{pdf_path}:{page_num}:{chunk_idx}"),
                "heading": heading,
                "level": 1,  # PDF не имеет иерархии заголовков
                "text": ct,
                "file_path": pdf_path,
                "page_number": page_num,  # Дополнительное поле для PDF
            })
    
    return samples


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
        description="Parse documents (Markdown, PDF) into RAG dataset.")
    ap.add_argument("--input_dir", required=True, help="Root directory with documents (processed recursively)")
    ap.add_argument("--output_jsonl", required=True, help="Output JSONL path")
    ap.add_argument("--levels", default="1,2,3", help="Heading levels to split on for Markdown, e.g. '1,2,3'")
    ap.add_argument("--chunk_size", type=int, default=512, help="Target chunk size in tokens")
    ap.add_argument("--overlap", type=int, default=64, help="Token overlap between chunks")
    ap.add_argument("--include_pdf", action="store_true", help="Include PDF files in processing")
    args = ap.parse_args()

    input_dir_abs = Path(args.input_dir).resolve()

    levels = tuple(int(x.strip()) for x in args.levels.split(",") if x.strip())
    
    all_rows: List[Dict[str, Any]] = []
    
    # Обработка Markdown файлов
    md_files = iter_md_files(args.input_dir)
    if md_files:
        print(f"Processing {len(md_files)} Markdown files...")
        for file in tqdm(md_files, desc="Markdown files"):
            try:
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
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    # Обработка PDF файлов
    if args.include_pdf:
        pdf_files = iter_pdf_files(args.input_dir)
        if pdf_files:
            if not PDF_AVAILABLE:
                print("Warning: PDF files found but pypdf is not installed. Skipping PDF files.")
                print("Install pypdf with: pip install pypdf")
            else:
                print(f"Processing {len(pdf_files)} PDF files...")
                for file in tqdm(pdf_files, desc="PDF files"):
                    try:
                        # Парсим PDF используя абсолютный путь
                        rows = build_chunks_for_pdf(
                            pdf_path=file,  # Абсолютный путь для парсинга
                            chunk_size=args.chunk_size,
                            overlap=args.overlap,
                        )
                        # Нормализуем пути в результатах
                        relative_path = Path(file).relative_to(input_dir_abs)
                        normalized_path = str(relative_path).replace('\\', '/')
                        for row in rows:
                            row['file_path'] = normalized_path
                        all_rows.extend(rows)
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
        elif PDF_AVAILABLE:
            print("No PDF files found in the input directory.")
    else:
        print("PDF processing disabled. Use --include_pdf to enable.")

    write_jsonl(args.output_jsonl, all_rows)
    print(f"\nWrote {len(all_rows)} records to {args.output_jsonl}")
    print(f"  - Markdown files: {len(md_files)}")
    if args.include_pdf and PDF_AVAILABLE:
        pdf_files = iter_pdf_files(args.input_dir)
        print(f"  - PDF files: {len(pdf_files)}")


if __name__ == "__main__":
    main()