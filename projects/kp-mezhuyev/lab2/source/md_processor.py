"""
Этап 2: нормализация Markdown с метаданными.

Скрипт берёт сырые `.md` файлы (например, из `data/row`) и сохраняет
нормализованные Markdown в выходную директорию, добавляя YAML‑фронтматтер
с метаданными источника.
"""
import argparse
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def remove_emoji(text: str) -> str:
    """Удаляет эмодзи и другие графические символы из текста.
    
    Использует Unicode категории для определения эмодзи и графических символов.
    Сохраняет обычные символы, цифры, знаки препинания.
    
    Args:
        text: Текст для очистки от эмодзи.
    
    Returns:
        Текст без эмодзи и графических символов.
    
    Example:
        >>> remove_emoji("Привет 🚀 мир! 😉")
        'Привет  мир! '
    """
    # Удаляем эмодзи и графические символы
    # Emoji обычно имеют категорию So (Symbol, other) или Sk (Symbol, modifier)
    # Также проверяем диапазоны эмодзи в Unicode
    result = []
    for char in text:
        # Проверяем категорию символа
        category = unicodedata.category(char)
        char_code = ord(char)
        
        # Пропускаем эмодзи:
        # - Символы в диапазонах эмодзи
        # - Символы категории So (Symbol, other) кроме некоторых исключений
        is_emoji = (
            # Основные диапазоны эмодзи
            (0x1F600 <= char_code <= 0x1F64F) or  # Emoticons
            (0x1F300 <= char_code <= 0x1F5FF) or  # Misc Symbols and Pictographs
            (0x1F680 <= char_code <= 0x1F6FF) or  # Transport and Map
            (0x1F1E0 <= char_code <= 0x1F1FF) or  # Flags
            (0x2600 <= char_code <= 0x26FF) or  # Misc symbols
            (0x2700 <= char_code <= 0x27BF) or  # Dingbats
            (0xFE00 <= char_code <= 0xFE0F) or  # Variation Selectors
            (0x1F900 <= char_code <= 0x1F9FF) or  # Supplemental Symbols and Pictographs
            (0x1FA00 <= char_code <= 0x1FAFF) or  # Chess Symbols
            # Категория So, но не все (исключаем некоторые полезные символы)
            (category == 'So' and char_code > 0x1F000)
        )
        
        if not is_emoji:
            result.append(char)
    
    return ''.join(result)


@dataclass
class MdDocument:
    """Документ Markdown с метаданными для обработки.
    
    Attributes:
        path: Абсолютный путь к исходному файлу.
        rel_path: Относительный путь от корня входной директории (используется для
            сохранения структуры директорий, генерации doc_id и source_path в метаданных).
        title: Заголовок документа (извлечённый из H1 или сгенерированный из имени файла).
        body: Нормализованное содержимое Markdown.
        output_path: Путь, куда будет записан обработанный файл.
    """
    path: Path
    rel_path: Path
    title: str
    body: str
    output_path: Path


def normalize_markdown(text: str) -> str:
    """
    Нормализация Markdown:
    - Unix-концы строк
    - удаление эмодзи (вне кодовых блоков и HTML-блоков)
    - не более одной пустой строки подряд (вне кодовых блоков и HTML-блоков)
    - единообразный пробел после решёток в заголовках (вне кодовых блоков)
    - сохранение якорных ссылок в заголовках для идентификации секций
    
    Сохраняет структуру HTML-блоков (не удаляет пустые строки внутри <div>, <span> и т.д.)
    и кодовых блоков. Якорные ссылки { #anchor } сохраняются в заголовках для лучшей
    идентификации секций при чанкинге.
    """

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized: list[str] = []
    in_code_block = False
    in_html_block = False
    empty_streak = 0

    for line in lines:
        raw = line.rstrip()
        stripped = raw.strip()

        # Отслеживание кодовых блоков
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            normalized.append(raw)
            empty_streak = 0
            continue

        # Отслеживание HTML-блоков (открывающие и закрывающие теги)
        if not in_code_block:
            if stripped.startswith("<") and stripped.endswith(">"):
                # Проверяем, не является ли это самозакрывающимся тегом или комментарием
                if not any(stripped.startswith(f"<{tag}") for tag in ["!--", "br", "hr", "img", "input", "meta", "link"]):
                    # Открывающий или закрывающий тег
                    if not stripped.startswith("</"):
                        in_html_block = True
                    elif in_html_block:
                        in_html_block = False

        if not in_code_block and not in_html_block:
            # Удаляем эмодзи из строки (но не из кодовых блоков и HTML)
            raw = remove_emoji(raw)
            stripped = raw.strip()
            
            if stripped.startswith("#"):
                # Обеспечиваем один пробел после решёток, но сохраняем якорные ссылки
                raw = re.sub(r"^(#+)\s*(.*)$", lambda m: f"{m.group(1)} {m.group(2).strip()}", stripped)

            if stripped == "":
                empty_streak += 1
                if empty_streak > 1:
                    continue
            else:
                empty_streak = 0
        elif in_html_block:
            # Внутри HTML-блока не ограничиваем пустые строки и не удаляем эмодзи
            empty_streak = 0

        normalized.append(raw)

    return "\n".join(normalized).strip() + "\n"


def _has_h1(text: str) -> bool:
    """Проверяет наличие заголовка H1 в тексте.
    
    Args:
        text: Markdown текст для проверки.
    
    Returns:
        True, если в тексте есть заголовок H1, иначе False.
    """
    return any(line.startswith("# ") for line in text.splitlines())


def extract_title(text: str, fallback: str) -> str:
    """Извлекает заголовок из первой строки H1 в Markdown тексте.
    
    Ищет первую строку, начинающуюся с "# " (заголовок первого уровня),
    и возвращает текст заголовка без символов форматирования и якорных ссылок.
    Удаляет якорные ссылки вида { #anchor } из заголовка.
    Если H1 не найден, возвращает fallback значение.
    
    Args:
        text: Markdown текст для поиска заголовка.
        fallback: Значение по умолчанию, если H1 не найден.
    
    Returns:
        Текст заголовка H1 без якорных ссылок или fallback значение.
    
    Example:
        >>> extract_title("# Заголовок { #anchor }", "Fallback")
        'Заголовок'
    """
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            # Удаляем якорные ссылки вида { #anchor } или {#anchor}
            title = re.sub(r'\s*\{[^}]*#\w[^}]*\}\s*$', '', title).strip()
            return title
    return fallback


def ensure_h1(text: str, title: str) -> str:
    """Гарантирует наличие заголовка H1 в начале документа.
    
    Проверяет, есть ли в тексте заголовок первого уровня (H1).
    Если H1 отсутствует, добавляет его в начало документа с указанным title.
    
    Args:
        text: Markdown текст для проверки.
        title: Заголовок, который будет добавлен, если H1 отсутствует.
    
    Returns:
        Текст с гарантированным наличием H1 заголовка.
    """
    if _has_h1(text):
        return text
    return f"# {title}\n\n{text}"


def yaml_frontmatter(metadata: dict[str, str]) -> str:
    """Генерирует YAML фронтматтер из словаря метаданных.
    
    Создаёт YAML-блок, обрамлённый тройными дефисами (---), который обычно
    размещается в начале Markdown файлов для хранения метаданных.
    
    Args:
        metadata: Словарь с парами ключ-значение для метаданных.
    
    Returns:
        Строка с YAML фронтматтером, заканчивающаяся двумя переносами строк.
    
    Example:
        >>> yaml_frontmatter({"title": "Test", "author": "John"})
        '---\\ntitle: Test\\nauthor: John\\n---\\n\\n'
    """
    body = "\n".join(f"{key}: {value}" for key, value in metadata.items())
    return f"---\n{body}\n---\n\n"


def iter_md_files(root: Path) -> Iterable[Path]:
    """Рекурсивно находит все .md файлы в указанной директории.
    
    Использует rglob для поиска всех Markdown файлов во всех поддиректориях,
    начиная с корневой директории.
    
    Args:
        root: Корневая директория для поиска.
    
    Yields:
        Path объекты для каждого найденного .md файла.
    """
    yield from root.rglob("*.md")


def build_doc(path: Path, input_root: Path, output_root: Path) -> MdDocument:
    """Создаёт объект MdDocument из исходного Markdown файла.
    
    Читает файл, нормализует его содержимое, извлекает или генерирует заголовок,
    вычисляет относительный путь и путь для выходного файла.
    
    Args:
        path: Абсолютный путь к исходному Markdown файлу.
        input_root: Корневая директория входных файлов (для вычисления rel_path).
        output_root: Корневая директория для выходных файлов.
    
    Returns:
        Объект MdDocument с обработанными данными.
    """
    raw = path.read_text(encoding="utf-8")
    normalized = normalize_markdown(raw)

    rel_path = path.relative_to(input_root)
    fallback_title = rel_path.stem.replace("_", " ").replace("-", " ").title()
    title = extract_title(normalized, fallback_title)
    normalized = ensure_h1(normalized, title)

    output_path = output_root / rel_path
    return MdDocument(path=path, rel_path=rel_path, title=title, body=normalized, output_path=output_path)


def write_doc(doc: MdDocument) -> None:
    """Записывает документ в файл с YAML фронтматтером.
    
    Создаёт необходимые директории, генерирует метаданные (включая doc_id на основе
    rel_path) и записывает документ с YAML фронтматтером в начало файла.
    
    Args:
        doc: Объект MdDocument для записи.
    """
    doc.output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "title": doc.title,
        "doc_id": hashlib.sha1(doc.rel_path.as_posix().encode("utf-8")).hexdigest()[:12],
        "source_path": doc.rel_path.as_posix(),
        "source_mtime": datetime.fromtimestamp(doc.path.stat().st_mtime, tz=timezone.utc).isoformat(),
    }
    payload = yaml_frontmatter(meta) + doc.body
    doc.output_path.write_text(payload, encoding="utf-8")


def process(input_dir: Path, output_dir: Path) -> list[MdDocument]:
    """Обрабатывает все Markdown файлы в входной директории.
    
    Находит все .md файлы рекурсивно, нормализует их, добавляет метаданные
    и сохраняет в выходную директорию с сохранением структуры поддиректорий.
    
    Args:
        input_dir: Директория с исходными Markdown файлами.
        output_dir: Директория для сохранения обработанных файлов.
    
    Returns:
        Список всех обработанных документов.
    """
    docs: list[MdDocument] = []
    for md_file in iter_md_files(input_dir):
        doc = build_doc(md_file, input_dir, output_dir)
        write_doc(doc)
        docs.append(doc)
    return docs


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки.
    
    Определяет входную и выходную директории для обработки Markdown файлов.
    Если аргументы не указаны, используются значения по умолчанию.
    
    Returns:
        Объект Namespace с атрибутами input и output.
    """
    parser = argparse.ArgumentParser(description="Normalize Markdown docs with metadata.")
    parser.add_argument("--input", default="data/row", help="Папка с сырыми markdown (по умолчанию data/row).")
    parser.add_argument(
        "--output",
        default="data/processed",
        help="Куда писать нормализованные markdown (по умолчанию data/processed).",
    )
    return parser.parse_args()


def main() -> None:
    """Главная функция: точка входа в программу.
    
    Парсит аргументы командной строки, проверяет существование входной директории
    и запускает обработку всех Markdown файлов.
    
    Относительные пути разрешаются относительно директории lab2 (родительской
    директории для source/), а не относительно текущей рабочей директории.
    """
    args = parse_args()
    
    # Определяем базовую директорию lab2 (родительскую для source/)
    script_dir = Path(__file__).parent
    lab2_dir = script_dir.parent
    
    # Разрешаем пути относительно lab2_dir
    if Path(args.input).is_absolute():
        input_dir = Path(args.input).expanduser().resolve()
    else:
        input_dir = (lab2_dir / args.input).resolve()
    
    if Path(args.output).is_absolute():
        output_dir = Path(args.output).expanduser().resolve()
    else:
        output_dir = (lab2_dir / args.output).resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    docs = process(input_dir, output_dir)
    print(f"Processed {len(docs)} markdown files.")


if __name__ == "__main__":
    main()
