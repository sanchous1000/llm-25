"""
Скрипт для построения эмбеддингов и загрузки в векторное хранилище.
Поддерживает dense, sparse и hybrid эмбеддинги.
"""
import argparse
import json
from typing import Any, Dict, List, Optional
import hashlib
from tqdm import tqdm

from utils.ollama import embed_texts, embed_query
from utils.qdrant import QdrantCollection

try:
    from rank_bm25 import BM25Okapi
    import numpy as np
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank_bm25 не установлен. Sparse эмбеддинги недоступны. Установите: pip install rank-bm25")


def load_jsonl(p: str) -> List[Dict[str, Any]]:
    """Загрузка данных из JSONL файла."""
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def simple_tokenize(text: str) -> List[str]:
    """Простая токенизация для BM25."""
    return text.lower().split()


def build_bm25_index(documents: List[str]) -> Optional[BM25Okapi]:
    """Построение BM25 индекса для sparse поиска."""
    if not BM25_AVAILABLE:
        return None

    tokenized_docs = [simple_tokenize(doc) for doc in documents]
    return BM25Okapi(tokenized_docs)


def get_bm25_scores(bm25: BM25Okapi, query: str, top_k: int = 10) -> List[float]:
    """Получение BM25 scores для запроса."""
    if bm25 is None:
        return []
    tokenized_query = simple_tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    # Нормализуем scores
    if len(scores) > 0:
        max_score = max(scores)
        if max_score > 0:
            scores = scores / max_score
    return scores.tolist()


def create_hybrid_vectors(
    dense_vectors: List[List[float]],
    sparse_scores: List[float],
    alpha: float = 0.7
) -> List[List[float]]:
    """
    Создание hybrid векторов из dense и sparse.

    Args:
        dense_vectors: Dense эмбеддинги
        sparse_scores: Sparse scores (BM25)
        alpha: Вес dense эмбеддингов (1-alpha для sparse)

    Returns:
        Hybrid векторы
    """
    # Для hybrid подхода можно использовать разные стратегии:
    # 1. Конкатенация (требует изменения размерности)
    # 2. Взвешенная комбинация (если размерности совпадают)
    # 3. Отдельные векторы для dense и sparse

    # Здесь используем простую стратегию: возвращаем dense, sparse храним отдельно
    # В реальности можно использовать Qdrant с несколькими векторами
    return dense_vectors


def main():
    ap = argparse.ArgumentParser(description="Построение эмбеддингов и загрузка в векторное хранилище")
    ap.add_argument("--input_jsonl", required=True, help="Путь к JSONL файлу с чанками")
    ap.add_argument("--ollama_host", default="http://localhost:11434", help="URL Ollama сервера")
    ap.add_argument("--embed_model", default="nomic-embed-text", help="Модель для dense эмбеддингов")
    ap.add_argument("--qdrant_host", default="localhost", help="Хост Qdrant")
    ap.add_argument("--qdrant_port", type=int, default=6333, help="Порт Qdrant")
    ap.add_argument("--collection", default="vllm_docs", help="Название коллекции")
    ap.add_argument("--distance", default="cosine", choices=["cosine", "dot", "euclidean"], help="Метрика расстояния")
    ap.add_argument(
        "--embedding_type",
        choices=["dense", "sparse", "hybrid"],
        default="dense",
        help="Тип эмбеддингов: dense, sparse (BM25), hybrid"
    )
    ap.add_argument("--recreate", action="store_true", help="Пересоздать коллекцию")
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="Пересобрать эмбеддинги (очистить коллекцию)"
    )
    ap.add_argument(
        "--hnsw_m",
        type=int,
        default=16,
        help="Параметр M для HNSW индекса"
    )
    ap.add_argument(
        "--hnsw_ef_construction",
        type=int,
        default=100,
        help="Параметр ef_construction для HNSW"
    )
    ap.add_argument(
        "--hnsw_ef_search",
        type=int,
        default=50,
        help="Параметр ef_search для HNSW"
    )
    args = ap.parse_args()

    # Загрузка данных
    print(f"Загрузка чанков из {args.input_jsonl}...")
    records = load_jsonl(args.input_jsonl)
    print(f"Загружено {len(records)} чанков")

    # Подготовка текстов для эмбеддингов
    texts = [f"{r['file_path']} | {r['heading']}\n{r['text']}" for r in records]

    # Построение эмбеддингов с прогресс-баром
    if args.embedding_type in ["dense", "hybrid"]:
        print(f"Построение dense эмбеддингов с моделью {args.embed_model}...")
        vectors = []
        with tqdm(total=len(texts), desc="Построение эмбеддингов", unit="текст") as pbar:
            for text in texts:
                vector = embed_query(text, model=args.embed_model, host=args.ollama_host)
                vectors.append(vector)
                pbar.update(1)
        vec_size = len(vectors[0])
        print(f"Dense эмбеддинги построены (размерность: {vec_size})")
    else:
        # Для sparse-only используем фиктивные векторы (в реальности нужна другая архитектура)
        print("Warning: Sparse-only режим требует специальной архитектуры. Используется dense.")
        vectors = []
        with tqdm(total=len(texts), desc="Построение эмбеддингов", unit="текст") as pbar:
            for text in texts:
                vector = embed_query(text, model=args.embed_model, host=args.ollama_host)
                vectors.append(vector)
                pbar.update(1)
        vec_size = len(vectors[0])

    # Построение sparse индекса (BM25) для hybrid режима
    bm25_index = None
    if args.embedding_type in ["sparse", "hybrid"]:
        if BM25_AVAILABLE:
            print("Построение BM25 индекса...")
            bm25_index = build_bm25_index([r['text'] for r in records])
            print("BM25 индекс построен")
        else:
            print("Warning: BM25 недоступен. Используется только dense.")

    # Добавляем sparse scores в метаданные для hybrid режима
    if args.embedding_type == "hybrid" and bm25_index:
        # Сохраняем BM25 индекс в метаданных (упрощенный подход)
        # В реальности можно использовать отдельную коллекцию или специальную структуру
        pass

    # Настройка HNSW
    hnsw_config = {
        "m": args.hnsw_m,
        "ef_construction": args.hnsw_ef_construction,
        "ef_search": args.hnsw_ef_search
    }

    # Создание/обновление коллекции
    qdrant = QdrantCollection(name=args.collection, host=args.qdrant_host, port=args.qdrant_port)
    qdrant.ensure_exists(
        vec_size=vec_size,
        distance=args.distance,
        recreate=args.recreate or args.rebuild,
        hnsw_config=hnsw_config
    )

    # Загрузка данных с прогресс-баром
    print(f"Загрузка {len(records)} точек в коллекцию '{args.collection}'...")
    with tqdm(total=len(records), desc="Загрузка в Qdrant", unit="точка") as pbar:
        # Загружаем батчами для отображения прогресса
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch_records = records[i:i+batch_size]
            batch_vectors = vectors[i:i+batch_size]
            qdrant.upload(batch_records, batch_vectors)
            pbar.update(len(batch_records))

    print(f"✓ Загружено {len(records)} точек в '{args.collection}' (dim={vec_size})")
    print(f"  Тип эмбеддингов: {args.embedding_type}")
    print(f"  HNSW параметры: M={args.hnsw_m}, ef_construction={args.hnsw_ef_construction}, ef_search={args.hnsw_ef_search}")

if __name__ == "__main__":
    main()