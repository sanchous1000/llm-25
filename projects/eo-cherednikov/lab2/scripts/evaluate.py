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
    relevant_file_paths: List[str]  # ID релевантных документов
    description: str  # Описание того, что ищем
    relevant_doc_ids: List[str] = None


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

def get_doc_id_to_path_mapping(qdrant_host: str, qdrant_port: int, collection: str) -> Dict[str, str]:
    """Получение соответствия между ID документа и путем к файлу"""
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    # Получаем все документы из коллекции
    all_docs = client.scroll(collection_name=collection, limit=10000)[0]

    mapping = {}
    for doc in all_docs:
        doc_id = doc.payload.get("id", "")
        file_path = doc.payload.get("file_path", "")
        if doc_id and file_path:
            mapping[doc_id] = file_path

    return mapping

def evaluate_query(
    query: EvaluationQuery,
    embed_model: str,
    ollama_host: str,
    qdrant: QdrantCollection,
    top_k: int
) -> Dict[str, Any]:
    """Оценка одного запроса"""

    # Получаем эмбеддинг запроса
    query_vec = embed_query(query.question, model=embed_model, host=ollama_host)

    # Поиск в Qdrant
    results = qdrant.search(query_vector=query_vec, top_k=top_k)

    # Извлекаем ID найденных документов
    retrieved_ids = [r.payload.get("id", "") for r in results]

    # Вычисляем метрики по ID документов
    recall_5 = calculate_recall_at_k(retrieved_ids, query.relevant_doc_ids, 5)
    recall_10 = calculate_recall_at_k(retrieved_ids, query.relevant_doc_ids, 10)
    precision_5 = calculate_precision_at_k(retrieved_ids, query.relevant_doc_ids, 5)
    precision_10 = calculate_precision_at_k(retrieved_ids, query.relevant_doc_ids, 10)
    mrr = calculate_mrr(retrieved_ids, query.relevant_doc_ids)

    return {
        "question": query.question,
        "description": query.description,
        "retrieved_ids": retrieved_ids,
        "retrieved_file_paths": [r.payload.get("file_path", "") for r in results],
        "relevant_ids": query.relevant_doc_ids,
        "recall_at_5": recall_5,
        "recall_at_10": recall_10,
        "precision_at_5": precision_5,
        "precision_at_10": precision_10,
        "mrr": mrr,
        "retrieved_headings": [r.payload.get("heading", "") for r in results],
    }



def load_ground_truth(ground_truth_file: str) -> List[EvaluationQuery]:
    """Загрузка ground truth данных из файла"""
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = []
    for item in data:
        query = EvaluationQuery(
            question=item["question"],
            relevant_file_paths=item["relevant_file_paths"],
            description=item.get("description", "")
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

    id_to_path_mapping = get_doc_id_to_path_mapping(args.qdrant_host, args.qdrant_port, args.collection)
    path_to_id_mapping = {path: doc_id for doc_id, path in id_to_path_mapping.items()}

    for query in tqdm(queries, desc="Forming GT"):
        file_path_ids = []
        for file_path in query.relevant_file_paths:
            if file_path in path_to_id_mapping:
                file_path_ids.append(path_to_id_mapping[file_path])
            else:
                print(f"Warning {file_path} not found in the collection")

        query.relevant_doc_ids = file_path_ids

    results = []
    qdrant = QdrantCollection(name=args.collection, host=args.qdrant_host, port=args.qdrant_port)

    progress_bar = tqdm(queries, desc="RAG Validation", total=len(queries))
    for i, query in enumerate(progress_bar):
        progress_bar.set_description(f"{i+1}/{len(queries)}")
        result = evaluate_query(
            query, args.embed_model, args.ollama_host, qdrant, args.top_k
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