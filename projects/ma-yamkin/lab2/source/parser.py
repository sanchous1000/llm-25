import fitz
import re
from pathlib import Path
from datetime import datetime


def is_likely_title(line, prev_line=None):
    """Эвристика: заголовки часто короткие, с заглавной буквы, без точки в конце"""
    if len(line) < 5 or len(line) > 100:
        return False
    if line.endswith("."):
        return False
    if prev_line and not prev_line.endswith("."):
        return False
    if re.match(r"^[A-Z][a-z]", line):
        return True
    return False


def pdf_to_markdown_by_page(pdf_path: str):
    """
    Возвращает список строк (по одной на странице), где каждый элемент — markdown для одной страницы.
    """
    doc = fitz.open(pdf_path)
    pages_md = []

    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")["blocks"]
        page_lines = []
        prev_line = None

        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                text = ""
                for span in line["spans"]:
                    text += span["text"]
                text = text.strip()
                if not text:
                    continue

                if is_likely_title(text, prev_line):
                    page_lines.append(f"## {text}")
                else:
                    page_lines.append(text)
                prev_line = text

        page_md = "\n".join(page_lines)
        pages_md.append(page_md)

    doc.close()
    return pages_md


if __name__ == "__main__":
    input_dir = Path("../data/raw/arxiv")
    output_base_dir = Path("../data/processed/arxiv")
    output_base_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.today().strftime("%Y-%m-%d")

    for pdf_file in input_dir.glob("*.pdf"):
        print(f"Обработка {pdf_file.name}...")

        arxiv_id = pdf_file.stem
        output_folder = output_base_dir / arxiv_id
        output_folder.mkdir(exist_ok=True)

        pages_md = pdf_to_markdown_by_page(str(pdf_file))

        for page_num, md_text in enumerate(pages_md, 1):
            md_filename = output_folder / f"page_{page_num:03d}.md"
            with open(md_filename, "w", encoding="utf-8") as f:
                f.write(f"""---
                    source: "{pdf_file}"
                    arxiv_id: "{arxiv_id}"
                    page: {page_num}
                    total_pages: {len(pages_md)}
                    date_converted: "{today}"
                    ---
                    
                    {md_text}
                """)