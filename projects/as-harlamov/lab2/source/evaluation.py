from typing import List, Dict, Any, Set
import numpy as np


def recall_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_retrieved = len(top_k_retrieved & relevant_ids)
    return relevant_retrieved / len(relevant_ids)


def precision_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    if k == 0:
        return 0.0
    
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_retrieved = len(top_k_retrieved & relevant_ids)
    return relevant_retrieved / k


def mean_reciprocal_rank(retrieved_ids: List[int], relevant_ids: Set[int]) -> float:
    if not relevant_ids:
        return 0.0
    
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    
    return 0.0


def evaluate_retrieval(
    query_results: List[Dict[str, Any]],
    ground_truth: Set[int],
    k_values: List[int] = [5, 10]
) -> Dict[str, float]:
    retrieved_ids = [result["id"] for result in query_results]
    
    metrics = {}
    
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, ground_truth, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, ground_truth, k)
    
    metrics["mrr"] = mean_reciprocal_rank(retrieved_ids, ground_truth)
    
    return metrics


def evaluate_batch(
    test_queries: List[Dict[str, Any]],
    retrieval_function,
    k_values: List[int] = [5, 10]
) -> Dict[str, float]:
    all_metrics = {}
    for k in k_values:
        all_metrics[f"recall@{k}"] = []
        all_metrics[f"precision@{k}"] = []
    all_metrics["mrr"] = []
    
    for query_data in test_queries:
        query = query_data["query"]
        ground_truth_ids = set(query_data["relevant_chunk_ids"])
        
        results = retrieval_function(query)
        retrieved_ids = [result["id"] for result in results]
        
        for k in k_values:
            all_metrics[f"recall@{k}"].append(
                recall_at_k(retrieved_ids, ground_truth_ids, k)
            )
            all_metrics[f"precision@{k}"].append(
                precision_at_k(retrieved_ids, ground_truth_ids, k)
            )
        
        all_metrics["mrr"].append(
            mean_reciprocal_rank(retrieved_ids, ground_truth_ids)
        )
    
    avg_metrics = {
        metric: np.mean(values)
        for metric, values in all_metrics.items()
    }
    
    return avg_metrics

