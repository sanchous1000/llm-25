import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from qdrant_client import QdrantClient
from tqdm import tqdm
from utils.ollama import embed_query
from utils.qdrant import QdrantCollection


@dataclass
class EvaluationQuery:
    question: str
    relevant_file_paths: List[str]
    description: str  # Описание того, что ищем
    relevant_doc_ids: List[str] = None
    relevant_pages: Dict[str, List[int]] = None  


def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Вычисление Recall@k"""
    if not relevant_ids:
        return 0.0
    retrieved_at_k = retrieved_ids[:k]
    relevant_retrieved = len(set(retrieved_at_k) & set(relevant_ids))
    return relevant_retrieved / len(relevant_ids)


def calculate_precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Вычисление Precision@k без потери порядка и учёта повторяющихся документов."""
    retrieved_at_k = retrieved_ids[:k]
    if not retrieved_at_k:
        return 0.0
    relevant_retrieved = sum(1 for id in retrieved_at_k if id in relevant_ids)
    return relevant_retrieved / len(retrieved_at_k)


def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Вычисление Mean Reciprocal Rank (MRR)"""
    if not relevant_ids:
        return 0.0

    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_query(
        query: EvaluationQuery,
        embed_model: str,
        ollama_host: str,
        qdrant: QdrantCollection,
        top_k: int,
        relevant_file_paths: List[str]
) -> Dict[str, Any]:
    """Оценка одного запроса"""

    # Получаем эмбеддинг запроса
    query_vec = embed_query(query.question, model=embed_model, host=ollama_host)

    # Поиск в Qdrant
    results = qdrant.search(query_vector=query_vec, top_k=top_k)

    normalized_relevant_paths = [p.replace('\\', '/') for p in relevant_file_paths]
    
    if query.relevant_pages:
        normalized_relevant_pages = {}
        for file_path, pages in query.relevant_pages.items():
            normalized_path = file_path.replace('\\', '/')
            normalized_relevant_pages[normalized_path] = pages
        
        retrieved_items = []  
        seen_items = set()
        for r in results:
            file_path = r.payload.get("file_path", "")
            normalized_path = file_path.replace('\\', '/')
            page_number = r.payload.get("page_number")
            
            if normalized_path in normalized_relevant_pages:
                if page_number is not None:
                    item_id = f"{normalized_path}:{page_number}"
                    if item_id not in seen_items:
                        retrieved_items.append(item_id)
                        seen_items.add(item_id)
                else:
                    if normalized_path not in seen_items:
                        retrieved_items.append(normalized_path)
                        seen_items.add(normalized_path)
            else:
                if normalized_path not in seen_items:
                    retrieved_items.append(normalized_path)
                    seen_items.add(normalized_path)
        
        relevant_items = []
        for file_path in normalized_relevant_paths:
            if file_path in normalized_relevant_pages:
                for page in normalized_relevant_pages[file_path]:
                    relevant_items.append(f"{file_path}:{page}")
            else:
                relevant_items.append(file_path)
    else:
        retrieved_items = []
        seen_paths = set()
        for r in results:
            file_path = r.payload.get("file_path", "")
            normalized_path = file_path.replace('\\', '/')
            if normalized_path not in seen_paths:
                retrieved_items.append(normalized_path)
                seen_paths.add(normalized_path)
        
        relevant_items = normalized_relevant_paths

    # Вычисляем метрики
    recall_5 = calculate_recall_at_k(retrieved_items, relevant_items, 5)
    recall_10 = calculate_recall_at_k(retrieved_items, relevant_items, 10)
    precision_5 = calculate_precision_at_k(retrieved_items, relevant_items, 5)
    precision_10 = calculate_precision_at_k(retrieved_items, relevant_items, 10)
    mrr = calculate_mrr(retrieved_items, relevant_items)

    # Собираем информацию о найденных страницах для отчета
    retrieved_info = []
    for r in results[:len(retrieved_items)]:
        file_path = r.payload.get("file_path", "")
        page_number = r.payload.get("page_number")
        info = file_path.replace('\\', '/')
        if page_number is not None:
            info += f" (page {page_number})"
        retrieved_info.append(info)

    return {
        "question": query.question,
        "description": query.description,
        "retrieved_file_paths": retrieved_items,
        "relevant_file_paths": relevant_items,
        "recall_at_5": recall_5,
        "recall_at_10": recall_10,
        "precision_at_5": precision_5,
        "precision_at_10": precision_10,
        "mrr": mrr,
        "retrieved_info": retrieved_info,
    }


def load_ground_truth(ground_truth_file: str) -> List[EvaluationQuery]:
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = []
    for item in data:
        # Нормализуем пути страниц если они указаны
        relevant_pages = None
        if "relevant_pages" in item and item["relevant_pages"]:
            relevant_pages = {}
            for file_path, pages in item["relevant_pages"].items():
                normalized_path = file_path.replace('\\', '/')
                # Убеждаемся, что pages - это список
                if isinstance(pages, int):
                    pages = [pages]
                relevant_pages[normalized_path] = pages
        
        query = EvaluationQuery(
            question=item["question"],
            relevant_file_paths=item["relevant_file_paths"],
            description=item.get("description", ""),
            relevant_pages=relevant_pages
        )
        queries.append(query)

    return queries


def main():
    ap = argparse.ArgumentParser(description="Оценка качества retrieval системы")
    ap.add_argument("--ollama_host", default="http://localhost:11434")
    ap.add_argument("--embed_model", default="nomic-embed-text")
    ap.add_argument("--qdrant_host", default="localhost")
    ap.add_argument("--qdrant_port", type=int, default=6333)
    ap.add_argument("--collection", default="vllm_docs")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--ground_truth_file", help="Путь к файлу с ground truth данными")
    args = ap.parse_args()

    assert args.ground_truth_file, "GT file not specified"
    queries = load_ground_truth(args.ground_truth_file)

    # Получаем все уникальные пути из коллекции для проверки
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    all_docs = client.scroll(collection_name=args.collection, limit=10000)[0]
    collection_paths = set()
    for doc in all_docs:
        file_path = doc.payload.get("file_path", "")
        if file_path:
            # Нормализуем путь
            normalized = file_path.replace('\\', '/')
            collection_paths.add(normalized)
    
    print(f"\nTotal unique file paths in collection: {len(collection_paths)}")
    
    # Проверяем наличие файлов из ground truth
    missing_files = []
    for query in queries:
        for file_path in query.relevant_file_paths:
            normalized_path = file_path.replace('\\', '/')
            if normalized_path not in collection_paths:
                if normalized_path not in missing_files:
                    missing_files.append(normalized_path)
                    print(f"Warning: {file_path} not found in the collection")
                    # Показываем похожие пути для отладки
                    filename = file_path.split('/')[-1]
                    similar = [p for p in collection_paths if filename in p or p.endswith(filename)]
                    if similar:
                        print(f"  Similar paths found: {similar[:3]}")
    
    if missing_files:
        print(f"\nTotal missing files: {len(missing_files)}")
        print("These files may not exist in the source data or were not indexed.")

    results = []
    qdrant = QdrantCollection(name=args.collection, host=args.qdrant_host, port=args.qdrant_port)

    progress_bar = tqdm(queries, desc="RAG Validation", total=len(queries))
    for i, query in enumerate(progress_bar):
        progress_bar.set_description(f"{i + 1}/{len(queries)}")
        result = evaluate_query(
            query, args.embed_model, args.ollama_host, qdrant, args.top_k, query.relevant_file_paths
        )
        results.append(result)

    avg_recall_5 = np.mean([r["recall_at_5"] for r in results])
    avg_recall_10 = np.mean([r["recall_at_10"] for r in results])
    avg_precision_5 = np.mean([r["precision_at_5"] for r in results])
    avg_precision_10 = np.mean([r["precision_at_10"] for r in results])
    avg_mrr = np.mean([r["mrr"] for r in results])

    print(f"\n=== RESULTS ===")
    print(f"Mean Recall@5:  {avg_recall_5:.3f}")
    print(f"Mean Recall@10: {avg_recall_10:.3f}")
    print(f"Mean Precision@5:  {avg_precision_5:.3f}")
    print(f"Mean Precision@10: {avg_precision_10:.3f}")
    print(f"Mean MRR: {avg_mrr:.3f}")


if __name__ == "__main__":
    main()