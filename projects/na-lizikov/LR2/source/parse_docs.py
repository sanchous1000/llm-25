import os
import pdfplumber
from bs4 import BeautifulSoup
from docx import Document
from datetime import datetime

# Определяем путь к корню проекта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "markdown")

os.makedirs(OUT_DIR, exist_ok=True)


def parse_pdf(path):
    """Парсинг PDF-документа постранично"""
    text = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text.append(f"## Страница {i + 1}\n{page_text}")
    return "\n\n".join(text)


def parse_docx(path):
    """Парсинг DOCX-документа"""
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def parse_html(path):
    """Парсинг HTML-документации Python"""
    with open(path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    main = soup.find("div", {"role": "main"})
    if not main:
        return ""

    content = []

    for tag in main.find_all(["h1", "h2", "h3", "p", "li", "pre"]):
        text = tag.get_text().strip()
        if not text:
            continue

        if tag.name in ["h1", "h2", "h3"]:
            level = int(tag.name[1])
            content.append(f"{'#' * level} {text}")
        elif tag.name == "pre":
            content.append(f"```python\n{text}\n```")
        else:
            content.append(text)

    return "\n\n".join(content)


def should_skip(root, file):
    """Фильтрация служебных HTML-файлов"""
    if "_static" in root or "_sources" in root:
        return True
    if file in {"genindex.html", "search.html"}:
        return True
    return False


def process_file(file_path):
    ext = file_path.lower().split(".")[-1]
    if ext == "pdf":
        return parse_pdf(file_path)
    elif ext == "docx":
        return parse_docx(file_path)
    elif ext in {"html", "htm"}:
        return parse_html(file_path)
    return None


for root, _, files in os.walk(RAW_DIR):
    for file in files:
        if should_skip(root, file):
            continue

        full_path = os.path.join(root, file)
        content = process_file(full_path)

        # защита от пустых и слишком коротких документов
        if not content or len(content.split()) < 100:
            continue

        # сохранение структуры каталогов
        rel_path = os.path.relpath(full_path, RAW_DIR)
        out_file = rel_path.replace(os.sep, "_") + ".md"
        out_path = os.path.join(OUT_DIR, out_file)

        meta = f"""---
source: {file}
relative_path: {rel_path}
parsed_at: {datetime.now().isoformat()}
---

"""

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(meta + content)

print("Парсинг завершён")
