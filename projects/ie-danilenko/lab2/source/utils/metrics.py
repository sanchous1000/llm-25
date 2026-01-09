#!/usr/bin/env python3
"""
Метрики для оценки качества retrieval.
"""

from typing import List


def calculate_recall_at_k(retrieved_indices: List[int], relevant_indices: List[int], k: int) -> float:
    """
    Вычисляет Recall@k.
    
    Args:
        retrieved_indices: Список индексов полученных результатов
        relevant_indices: Список индексов релевантных результатов
        k: Значение k для метрики
    
    Returns:
        Значение Recall@k
    """
    if not relevant_indices:
        return 0.0
    
    retrieved_at_k = set(retrieved_indices[:k])
    relevant_set = set(relevant_indices)
    
    intersection = retrieved_at_k & relevant_set
    return len(intersection) / len(relevant_set) if relevant_set else 0.0


def calculate_precision_at_k(retrieved_indices: List[int], relevant_indices: List[int], k: int) -> float:
    """
    Вычисляет Precision@k.
    
    Args:
        retrieved_indices: Список индексов полученных результатов
        relevant_indices: Список индексов релевантных результатов
        k: Значение k для метрики
    
    Returns:
        Значение Precision@k
    """
    if k == 0:
        return 0.0
    
    retrieved_at_k = set(retrieved_indices[:k])
    relevant_set = set(relevant_indices)
    
    intersection = retrieved_at_k & relevant_set
    return len(intersection) / k if k > 0 else 0.0


def calculate_mrr(retrieved_indices: List[int], relevant_indices: List[int]) -> float:
    """
    Вычисляет Mean Reciprocal Rank (MRR).
    
    Args:
        retrieved_indices: Список индексов полученных результатов
        relevant_indices: Список индексов релевантных результатов
    
    Returns:
        Значение MRR
    """
    if not relevant_indices:
        return 0.0
    
    relevant_set = set(relevant_indices)
    
    for rank, idx in enumerate(retrieved_indices, start=1):
        if idx in relevant_set:
            return 1.0 / rank
    
    return 0.0

