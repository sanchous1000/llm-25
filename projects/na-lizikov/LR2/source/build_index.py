import os
import yaml
import json
import re
import argparse
import hashlib
from datetime import datetime
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Предупреждение: rank_bm25 не установлен. Sparse эмбеддинги (BM25) недоступны.")

# Определяем путь к корню проекта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


def parse_markdown_metadata(content):
    """Извлекает метаданные из frontmatter markdown файла"""
    metadata = {}
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            meta_text = parts[1]
            content = "---".join(parts[2:])
            for line in meta_text.strip().split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()
    return metadata, content


def encode_with_truncation(text, tokenizer, max_tokens=None):
    """Кодирует текст с обрезкой до max_tokens"""
    if max_tokens:
        tokens = tokenizer.encode(text, add_special_tokens=False, max_length=max_tokens, truncation=True)
    else:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokens


def tokenize_text(text, tokenizer):
    """Токенизирует текст для подсчета токенов"""
    return tokenizer.encode(text, add_special_tokens=False)


def recursive_splitter(text, chunk_size, overlap, tokenizer, max_tokens=None):
    """
    Рекурсивный сплиттер: разбивает текст на чанки с учетом структуры
    """
    def split_recursive(text_parts, current_chunk, current_tokens, chunks):
        for part in text_parts:
            part_tokens = encode_with_truncation(part, tokenizer, max_tokens)
            part_token_count = len(part_tokens)
            
            if current_tokens + part_token_count > chunk_size and current_chunk:
                # Сохраняем текущий чанк
                chunk_text = "\n".join(current_chunk).strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "headers": []
                    })
                
                # Начинаем новый чанк с overlap (в токенах)
                if overlap > 0 and current_chunk:
                    # Находим элементы для overlap, начиная с конца
                    overlap_tokens_count = 0
                    overlap_elements = []
                    for elem in reversed(current_chunk):
                        elem_tokens = encode_with_truncation(elem, tokenizer, max_tokens)
                        elem_token_count = len(elem_tokens)
                        if overlap_tokens_count + elem_token_count <= overlap:
                            overlap_elements.insert(0, elem)
                            overlap_tokens_count += elem_token_count
                        else:
                            break
                    
                    if overlap_elements:
                        current_chunk = overlap_elements + [part]
                        current_tokens = overlap_tokens_count + part_token_count
                    else:
                        current_chunk = [part]
                        current_tokens = part_token_count
                else:
                    current_chunk = [part]
                    current_tokens = part_token_count
            else:
                current_chunk.append(part)
                current_tokens += part_token_count
        
        # Добавляем последний чанк
        if current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "headers": []
                })
    
    # Разбиваем на параграфы
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    split_recursive(paragraphs, [], 0, chunks)
    return chunks


def markdown_h1h3_splitter(text, chunk_size, overlap, tokenizer, max_tokens=None):
    """
    Markdown-сплиттер по заголовкам h1-h3
    """
    lines = text.split("\n")
    chunks = []
    current_chunk_lines = []
    current_headers = []
    current_chunk_tokens = 0
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            if current_chunk_lines:
                current_chunk_lines.append("")
            continue
        
        # Проверяем, является ли строка заголовком h1-h3
        if line_stripped.startswith("#") and len(line_stripped) - len(line_stripped.lstrip("#")) <= 3:
            # Если накопили достаточно токенов, сохраняем чанк
            if current_chunk_tokens >= chunk_size and current_chunk_lines:
                chunk_text = "\n".join(current_chunk_lines).strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "headers": current_headers.copy()
                    })
                
                # Начинаем новый чанк с overlap
                if overlap > 0 and len(current_chunk_lines) > overlap:
                    overlap_text = "\n".join(current_chunk_lines[-overlap:])
                    overlap_tokens = encode_with_truncation(overlap_text, tokenizer, max_tokens)
                    current_chunk_lines = current_chunk_lines[-overlap:]
                    current_chunk_tokens = len(overlap_tokens)
                else:
                    current_chunk_lines = []
                    current_chunk_tokens = 0
            
            # Обновляем стек заголовков
            level = len(line_stripped) - len(line_stripped.lstrip("#"))
            header_text = line_stripped.lstrip("#").strip()
            # Удаляем лишние заголовки и добавляем новый
            current_headers = current_headers[:level-1] + [header_text]
            current_chunk_lines.append(line)
            # Добавляем токены заголовка
            line_tokens = encode_with_truncation(line, tokenizer, max_tokens)
            current_chunk_tokens += len(line_tokens)
        else:
            # Обычная строка
            line_tokens = encode_with_truncation(line, tokenizer, max_tokens)
            line_token_count = len(line_tokens)
            
            # Если добавление строки превысит лимит, сохраняем текущий чанк
            if current_chunk_tokens + line_token_count > chunk_size and current_chunk_tokens > 0:
                chunk_text = "\n".join(current_chunk_lines).strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "headers": current_headers.copy()
                    })
                
                # Начинаем новый чанк с overlap
                if overlap > 0 and len(current_chunk_lines) > overlap:
                    overlap_text = "\n".join(current_chunk_lines[-overlap:])
                    overlap_tokens = encode_with_truncation(overlap_text, tokenizer, max_tokens)
                    current_chunk_lines = current_chunk_lines[-overlap:] + [line]
                    current_chunk_tokens = len(overlap_tokens) + line_token_count
                else:
                    current_chunk_lines = [line]
                    current_chunk_tokens = line_token_count
            else:
                current_chunk_lines.append(line)
                current_chunk_tokens += line_token_count
    
    # Добавляем последний чанк
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines).strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "headers": current_headers.copy()
            })
    
    return chunks


def hybrid_splitter(text, chunk_size, overlap, tokenizer, max_tokens=None):
    """
    Гибридный сплиттер: заголовки + окно/overlap
    """
    # Сначала разбиваем по заголовкам
    lines = text.split("\n")
    sections = []
    current_section = []
    current_headers = []
    
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith("#"):
            if current_section:
                sections.append({
                    "headers": current_headers.copy(),
                    "content": "\n".join(current_section)
                })
            level = len(line_stripped) - len(line_stripped.lstrip("#"))
            header_text = line_stripped.lstrip("#").strip()
            current_headers = current_headers[:level-1] + [header_text]
            current_section = [line]
        else:
            current_section.append(line)
    
    if current_section:
        sections.append({
            "headers": current_headers.copy(),
            "content": "\n".join(current_section)
        })
    
    # Теперь разбиваем каждую секцию на чанки с overlap
    chunks = []
    for section in sections:
        section_text = section["content"]
        section_headers = section["headers"]
        
        # Разбиваем секцию на чанки
        section_lines = section_text.split("\n")
        current_chunk_lines = []
        current_chunk_tokens = 0
        
        for line in section_lines:
            line_tokens = encode_with_truncation(line, tokenizer, max_tokens)
            line_token_count = len(line_tokens)
            
            if current_chunk_tokens + line_token_count > chunk_size and current_chunk_tokens > 0:
                chunk_text = "\n".join(current_chunk_lines).strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "headers": section_headers.copy()
                    })
                
                # Overlap
                if overlap > 0 and len(current_chunk_lines) > overlap:
                    overlap_lines = current_chunk_lines[-overlap:]
                    overlap_text = "\n".join(overlap_lines)
                    overlap_tokens = encode_with_truncation(overlap_text, tokenizer, max_tokens)
                    current_chunk_lines = overlap_lines + [line]
                    current_chunk_tokens = len(overlap_tokens) + line_token_count
                else:
                    current_chunk_lines = [line]
                    current_chunk_tokens = line_token_count
            else:
                current_chunk_lines.append(line)
                current_chunk_tokens += line_token_count
        
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines).strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "headers": section_headers.copy()
                })
    
    return chunks


def get_splitter(splitter_type):
    """Возвращает функцию сплиттера по типу"""
    splitters = {
        "recursive": recursive_splitter,
        "markdown": markdown_h1h3_splitter,
        "hybrid": hybrid_splitter
    }
    return splitters.get(splitter_type, markdown_h1h3_splitter)


def get_openai_embedding(text, model_name, api_key):
    """Получает эмбеддинг через OpenAI API"""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model=model_name,
            input=text
        )
        return response.data[0].embedding
    except ImportError:
        raise ImportError("openai не установлен. Установите: pip install openai")
    except Exception as e:
        raise Exception(f"Ошибка при получении эмбеддинга OpenAI: {e}")


def build_bm25_index(chunks):
    """Строит BM25 индекс для sparse эмбеддингов"""
    if not BM25_AVAILABLE:
        raise ImportError("rank_bm25 не установлен. Установите: pip install rank-bm25")
    
    # Токенизируем тексты для BM25
    tokenized_corpus = []
    for chunk in chunks:
        # Простая токенизация (можно улучшить)
        tokens = re.findall(r'\b\w+\b', chunk["text"].lower())
        tokenized_corpus.append(tokens)
    
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def compute_config_hash(config):
    """Вычисляет хеш конфигурации для версионирования"""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def main():
    parser = argparse.ArgumentParser(description="Построение индекса чанков и эмбеддингов")
    parser.add_argument("--chunk-size", type=int, help="Размер чанка в токенах (100-1000)")
    parser.add_argument("--overlap", type=int, help="Размер overlap")
    parser.add_argument("--splitter", choices=["recursive", "markdown", "hybrid"], help="Тип сплиттера")
    parser.add_argument("--embedding-type", choices=["dense", "sparse", "hybrid"], help="Тип векторизации")
    parser.add_argument("--embedding-model", type=str, help="Модель эмбеддингов")
    parser.add_argument("--include-headers", choices=["text", "metadata", "both"], help="Включение заголовков")
    parser.add_argument("--max-pages", type=int, help="Ограничение размера корпуса")
    parser.add_argument("--batch-size", type=int, help="Размер батча для эмбеддингов")
    parser.add_argument("--max-tokens", type=int, help="Максимальная длина токенов")
    parser.add_argument("--openai-api-key", type=str, help="OpenAI API ключ (если используется OpenAI модель)")
    parser.add_argument("--force-rebuild", action="store_true", help="Принудительная пересборка")
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    CONFIG = yaml.safe_load(open(CONFIG_PATH))
    
    # Применяем аргументы командной строки
    CHUNK_SIZE = args.chunk_size if args.chunk_size else CONFIG.get("chunk_size", 300)
    OVERLAP = args.overlap if args.overlap else CONFIG.get("overlap", 50)
    SPLITTER_TYPE = args.splitter if args.splitter else CONFIG.get("splitter_type", "markdown")
    EMBEDDING_TYPE = args.embedding_type if args.embedding_type else CONFIG.get("embedding_type", "dense")
    EMBEDDING_MODEL = args.embedding_model if args.embedding_model else CONFIG.get("embedding_model", "BAAI/bge-base-en")
    INCLUDE_HEADERS = args.include_headers if args.include_headers else CONFIG.get("include_headers", "text")
    MAX_PAGES = args.max_pages if args.max_pages else CONFIG.get("max_pages", None)
    BATCH_SIZE = args.batch_size if args.batch_size else CONFIG.get("batch_size", 128)
    MAX_TOKENS = args.max_tokens if args.max_tokens else CONFIG.get("max_tokens", 512)
    OPENAI_API_KEY = args.openai_api_key or CONFIG.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    
    # Валидация параметров
    if not (100 <= CHUNK_SIZE <= 1000):
        raise ValueError(f"chunk_size должен быть в диапазоне 100-1000, получено: {CHUNK_SIZE}")
    
    # Создаем конфигурацию для версионирования
    build_config = {
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "splitter_type": SPLITTER_TYPE,
        "embedding_type": EMBEDDING_TYPE,
        "embedding_model": EMBEDDING_MODEL,
        "include_headers": INCLUDE_HEADERS,
        "max_pages": MAX_PAGES,
        "batch_size": BATCH_SIZE,
        "max_tokens": MAX_TOKENS
    }
    
    config_hash = compute_config_hash(build_config)
    
    MD_DIR = os.path.join(PROJECT_ROOT, "data", "markdown")
    OUT_DIR = os.path.join(PROJECT_ROOT, "data", "chunks")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Проверяем существующий индекс
    chunks_file = os.path.join(OUT_DIR, "chunks.json")
    metadata_file = os.path.join(OUT_DIR, "metadata.json")
    
    if os.path.exists(chunks_file) and os.path.exists(metadata_file) and not args.force_rebuild:
        with open(metadata_file, "r", encoding="utf-8") as f:
            existing_metadata = json.load(f)
        
        if existing_metadata.get("config_hash") == config_hash:
            print(f"Индекс уже существует с такой же конфигурацией (hash: {config_hash}). Используйте --force-rebuild для пересборки.")
            return
    
    # Инициализация модели и токенизатора
    if EMBEDDING_TYPE in ["dense", "hybrid"]:
        if EMBEDDING_MODEL.startswith("text-embedding") or "openai" in EMBEDDING_MODEL.lower():
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI API ключ требуется для OpenAI моделей. Укажите --openai-api-key или установите OPENAI_API_KEY")
            print(f"Используется OpenAI модель: {EMBEDDING_MODEL}")
            # Для OpenAI используем токенизатор из transformers для подсчета токенов
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        else:
            print(f"Загрузка модели: {EMBEDDING_MODEL}")
            model = SentenceTransformer(EMBEDDING_MODEL)
            tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    
    if EMBEDDING_TYPE in ["sparse", "hybrid"]:
        if not BM25_AVAILABLE:
            raise ImportError("rank_bm25 требуется для sparse/hybrid эмбеддингов. Установите: pip install rank-bm25")
    
    # Получаем функцию сплиттера
    splitter_func = get_splitter(SPLITTER_TYPE)
    
    all_chunks = []
    
    # Подсчитываем общий размер корпуса для ограничения
    total_pages_approx = 0
    files_to_process = []
    
    print(f"Анализ markdown файлов из {MD_DIR}...")
    all_files = sorted(os.listdir(MD_DIR))
    
    # Если установлено ограничение, сначала оцениваем размер файлов
    if MAX_PAGES:
        for file in all_files:
            file_path = os.path.join(MD_DIR, file)
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            metadata, text_content = parse_markdown_metadata(content)
            words = len(text_content.split())
            pages_approx = words / 500
            
            if total_pages_approx + pages_approx <= MAX_PAGES:
                files_to_process.append((file, words, pages_approx))
                total_pages_approx += pages_approx
            else:
                remaining_pages = MAX_PAGES - total_pages_approx
                if remaining_pages > 0.1:
                    files_to_process.append((file, words, pages_approx))
                    total_pages_approx += pages_approx
                break
        print(f"Ограничение: будет обработано ~{total_pages_approx:.1f} страниц ({len(files_to_process)} файлов)")
    else:
        files_to_process = [(file, 0, 0) for file in all_files]
    
    print(f"Обработка {len(files_to_process)} файлов с сплиттером '{SPLITTER_TYPE}'...")
    for file, _, _ in tqdm(files_to_process, desc="Файлы"):
        file_path = os.path.join(MD_DIR, file)
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        
        # Извлекаем метаданные
        metadata, text_content = parse_markdown_metadata(content)
        
        # Разбиваем на чанки
        chunks = splitter_func(text_content, CHUNK_SIZE, OVERLAP, tokenizer, MAX_TOKENS)
        
        for i, chunk_data in enumerate(chunks):
            # Формируем текст чанка в зависимости от INCLUDE_HEADERS
            headers_text = "\n".join([f"{'#' * (j+1)} {h}" for j, h in enumerate(chunk_data["headers"])])
            
            if INCLUDE_HEADERS == "text":
                chunk_text = headers_text + "\n\n" + chunk_data["text"] if chunk_data["headers"] else chunk_data["text"]
            elif INCLUDE_HEADERS == "metadata":
                chunk_text = chunk_data["text"]
            else:  # both
                chunk_text = headers_text + "\n\n" + chunk_data["text"] if chunk_data["headers"] else chunk_data["text"]
            
            # Финальная проверка и обрезка длины токенов
            if MAX_TOKENS:
                tokens = encode_with_truncation(chunk_text, tokenizer, MAX_TOKENS)
                if len(tokens) > MAX_TOKENS:
                    chunk_text = tokenizer.decode(tokens[:MAX_TOKENS], skip_special_tokens=True)
            
            chunk_id = f"{file}_{i}"
            chunk_dict = {
                "id": chunk_id,
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source": metadata.get("source", file),
                "relative_path": metadata.get("relative_path", file),
            }
            
            # Добавляем заголовки в метаданные если нужно
            if INCLUDE_HEADERS in ["metadata", "both"]:
                chunk_dict["headers"] = chunk_data["headers"]
            elif chunk_data["headers"]:
                chunk_dict["headers"] = chunk_data["headers"]  # Всегда сохраняем для справки
            
            all_chunks.append(chunk_dict)
    
    print(f"Создано {len(all_chunks)} чанков.")
    
    # Генерация эмбеддингов
    if EMBEDDING_TYPE in ["dense", "hybrid"]:
        print(f"Генерация dense эмбеддингов (batch_size={BATCH_SIZE})...")
        if EMBEDDING_MODEL.startswith("text-embedding") or "openai" in EMBEDDING_MODEL.lower():
            # OpenAI эмбеддинги
            embeddings = []
            for chunk in tqdm(all_chunks, desc="OpenAI эмбеддинги"):
                emb = get_openai_embedding(chunk["text"], EMBEDDING_MODEL, OPENAI_API_KEY)
                embeddings.append(emb)
        else:
            # SentenceTransformer эмбеддинги
            embeddings = model.encode(
                [c["text"] for c in all_chunks],
                show_progress_bar=True,
                batch_size=BATCH_SIZE,
                convert_to_numpy=True
            )
        
        for i, emb in enumerate(embeddings):
            if isinstance(emb, list):
                all_chunks[i]["embedding"] = emb
            else:
                all_chunks[i]["embedding"] = emb.tolist()
    
    if EMBEDDING_TYPE in ["sparse", "hybrid"]:
        print("Построение BM25 индекса для sparse эмбеддингов...")
        bm25 = build_bm25_index(all_chunks)
        # Сохраняем BM25 индекс отдельно (можно сериализовать)
        # Для простоты сохраняем только токенизированный корпус
        tokenized_corpus = []
        for chunk in all_chunks:
            tokens = re.findall(r'\b\w+\b', chunk["text"].lower())
            tokenized_corpus.append(tokens)
        
        # Сохраняем информацию о BM25 для использования в retrieve
        bm25_info = {
            "tokenized_corpus": tokenized_corpus,
            "idf": bm25.idf.tolist() if hasattr(bm25.idf, 'tolist') else bm25.idf,
            "avgdl": bm25.avgdl
        }
        
        if EMBEDDING_TYPE == "sparse":
            # Для sparse только сохраняем BM25 информацию
            for i, chunk in enumerate(all_chunks):
                chunk["bm25_tokens"] = tokenized_corpus[i]
        else:
            # Для hybrid сохраняем и dense и sparse
            for i, chunk in enumerate(all_chunks):
                chunk["bm25_tokens"] = tokenized_corpus[i]
    
    # Сохранение чанков
    print("Сохранение чанков...")
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    # Сохранение метаданных
    metadata = {
        "config_hash": config_hash,
        "build_config": build_config,
        "build_time": datetime.now().isoformat(),
        "total_chunks": len(all_chunks),
        "total_files": len(files_to_process),
        "embedding_type": EMBEDDING_TYPE,
        "embedding_model": EMBEDDING_MODEL,
        "splitter_type": SPLITTER_TYPE
    }
    
    if EMBEDDING_TYPE in ["sparse", "hybrid"]:
        metadata["bm25_info"] = bm25_info
    
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Чанки и эмбеддинги построены. Всего чанков: {len(all_chunks)}")
    print(f"Конфигурация сохранена (hash: {config_hash})")


if __name__ == "__main__":
    main()
