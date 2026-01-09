"""
Скрипт для оценки качества retrieval и QA.
Выполняет задание 5: метрики и оценка качества retrieval/QA.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
import argparse

from load_to_vector_store import FaissVectorStore
from utils import (
    get_device, tokenize,
    load_questions, load_ground_truth, load_chunks,
    calculate_recall_at_k, calculate_precision_at_k, calculate_mrr
)

class RetrievalEvaluator:
    """Класс для оценки качества retrieval."""
    
    def __init__(self, 
                 faiss_index_dir: str,
                 chunks_dir: str,
                 dense_model_name: str,
                 device: Optional[str] = None):
        """
        Инициализация оценщика.
        
        Args:
            faiss_index_dir: Директория с индексом Faiss
            chunks_dir: Директория с чанками
            dense_model_name: Название модели для dense эмбеддингов
            device: Устройство для вычислений
        """
        self.faiss_index_dir = Path(faiss_index_dir)
        self.chunks_dir = Path(chunks_dir)
        
        # Загружаем модель для эмбеддингов
        if device is None:
            device = get_device()
        self.device = device
        print(f"Загрузка модели эмбеддингов: {dense_model_name} на {device}")
        self.embedding_model = SentenceTransformer(dense_model_name, device=device)
        
        # Загружаем индекс и чанки
        self.vector_store = FaissVectorStore(str(self.faiss_index_dir))
        try:
            self.vector_store.load_index()
        except FileNotFoundError:
            print(f"Warning: Индекс не найден в {self.faiss_index_dir}, будет создан при первом поиске")
        
        # Загружаем sparse модель если есть
        self.sparse_model = None
        self.tokenized_corpus = None
        self._load_sparse_model()
    
    def _load_sparse_model(self):
        """Загружает sparse модель из чанков."""
        # Ищем последнюю версию чанков
        version_dirs = sorted(self.chunks_dir.glob("v_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if version_dirs:
            sparse_file = version_dirs[0] / "sparse_model.pkl"
            if sparse_file.exists():
                with open(sparse_file, 'rb') as f:
                    self.sparse_model = pickle.load(f)
                print("Sparse модель (BM25) загружена")
                
                # Загружаем токенизированный корпус
                chunks_file = version_dirs[0] / "chunks.json"
                if chunks_file.exists():
                    with open(chunks_file, 'r', encoding='utf-8') as f:
                        chunks = json.load(f)
                    texts = [chunk["text"] for chunk in chunks]
                    self.tokenized_corpus = [self._tokenize(text) for text in texts]
                    
                    # Сохраняем чанки в vector_store для доступа
                    if not self.vector_store.chunks:
                        self.vector_store.chunks = chunks
    
    def _tokenize(self, text: str) -> List[str]:
        """Токенизация для BM25."""
        return tokenize(text)
    
    def search_dense(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Поиск по dense эмбеддингам."""
        if self.vector_store.index is None:
            raise ValueError("Индекс Faiss не загружен. Запустите load_to_vector_store.py сначала.")
        
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)[0]
        results = self.vector_store.search(query_embedding, k=k)
        return results
    
    def search_sparse(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Поиск по sparse эмбеддингам (BM25)."""
        if not self.sparse_model:
            return []
        
        if not self.vector_store.chunks:
            # Загружаем чанки если их нет
            self._load_chunks()
        
        tokenized_query = self._tokenize(query)
        scores = self.sparse_model.get_scores(tokenized_query)
        
        # Топ-k индексов
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for idx in top_indices:
            if self.vector_store.chunks and idx < len(self.vector_store.chunks):
                results.append({
                    "index": idx,
                    "distance": float(scores[idx]),
                    "chunk": self.vector_store.chunks[idx]
                })
        
        return results
    
    def _load_chunks(self):
        """Загружает чанки из последней версии."""
        try:
            chunks_data = load_chunks(str(self.chunks_dir))
            self.vector_store.chunks = chunks_data
        except FileNotFoundError:
            pass
    
    def search_hybrid(self, query: str, k: int = 10, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Гибридный поиск (dense + sparse).
        
        Args:
            query: Запрос
            k: Количество результатов
            alpha: Вес dense поиска (0-1), sparse вес = 1-alpha
        """
        dense_results = self.search_dense(query, k=k*2)
        sparse_results = self.search_sparse(query, k=k*2)
        
        # Нормализуем scores
        if dense_results:
            max_dense = max(r["distance"] for r in dense_results) if dense_results else 1.0
            min_dense = min(r["distance"] for r in dense_results) if dense_results else 0.0
            dense_range = max_dense - min_dense if max_dense != min_dense else 1.0
        
        if sparse_results:
            max_sparse = max(r["distance"] for r in sparse_results) if sparse_results else 1.0
            min_sparse = min(r["distance"] for r in sparse_results) if sparse_results else 0.0
            sparse_range = max_sparse - min_sparse if max_sparse != min_sparse else 1.0
        
        # Объединяем результаты
        combined_scores = {}
        for result in dense_results:
            idx = result["index"]
            # Инвертируем distance (меньше = лучше) и нормализуем
            normalized_score = 1.0 - (result["distance"] - min_dense) / dense_range if dense_range > 0 else 0.5
            combined_scores[idx] = alpha * normalized_score
        
        for result in sparse_results:
            idx = result["index"]
            # Нормализуем BM25 score
            normalized_score = (result["distance"] - min_sparse) / sparse_range if sparse_range > 0 else 0.5
            if idx in combined_scores:
                combined_scores[idx] += (1 - alpha) * normalized_score
            else:
                combined_scores[idx] = (1 - alpha) * normalized_score
        
        # Сортируем по комбинированному score
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for idx, score in sorted_indices:
            if self.vector_store.chunks and idx < len(self.vector_store.chunks):
                results.append({
                    "index": idx,
                    "distance": 1.0 - score,  # Для совместимости с dense
                    "chunk": self.vector_store.chunks[idx]
                })
        
        return results




def evaluate_retrieval(evaluator: RetrievalEvaluator,
                      questions: List[Dict[str, Any]],
                      ground_truth: Dict[int, List[int]],
                      search_type: str = "dense",
                      k_values: List[int] = [5, 10]) -> Dict[str, Any]:
    """
    Оценивает качество retrieval.
    
    Args:
        evaluator: Оценщик retrieval
        questions: Список вопросов
        ground_truth: Релевантные чанки для каждого вопроса
        search_type: Тип поиска (dense, sparse, hybrid)
        k_values: Значения k для метрик
    
    Returns:
        Словарь с метриками
    """
    results = {
        "search_type": search_type,
        "num_questions": len(questions),
        "metrics": {}
    }
    
    all_recalls = {k: [] for k in k_values}
    all_precisions = {k: [] for k in k_values}
    all_mrrs = []
    
    for question in questions:
        q_id = question["id"]
        query = question["question"]
        relevant_indices = ground_truth.get(str(q_id), [])
        
        if not relevant_indices:
            continue  # Пропускаем вопросы без ground truth
        
        # Выполняем поиск
        if search_type == "dense":
            search_results = evaluator.search_dense(query, k=max(k_values))
        elif search_type == "sparse":
            search_results = evaluator.search_sparse(query, k=max(k_values))
        elif search_type == "hybrid":
            search_results = evaluator.search_hybrid(query, k=max(k_values))
        else:
            raise ValueError(f"Неизвестный тип поиска: {search_type}")
        
        retrieved_indices = [r["index"] for r in search_results]
        
        # Вычисляем метрики
        for k in k_values:
            recall = calculate_recall_at_k(retrieved_indices, relevant_indices, k)
            precision = calculate_precision_at_k(retrieved_indices, relevant_indices, k)
            all_recalls[k].append(recall)
            all_precisions[k].append(precision)
        
        mrr = calculate_mrr(retrieved_indices, relevant_indices)
        all_mrrs.append(mrr)
    
    # Вычисляем средние метрики
    for k in k_values:
        results["metrics"][f"Recall@{k}"] = {
            "mean": np.mean(all_recalls[k]),
            "std": np.std(all_recalls[k])
        }
        results["metrics"][f"Precision@{k}"] = {
            "mean": np.mean(all_precisions[k]),
            "std": np.std(all_precisions[k])
        }
    
    results["metrics"]["MRR"] = {
        "mean": np.mean(all_mrrs),
        "std": np.std(all_mrrs)
    }
    
    return results

if __name__ == "__main__":
    """Основная функция."""
    
    parser = argparse.ArgumentParser(description='Оценка качества retrieval')
    parser.add_argument('--faiss-index-dir', type=str, default='faiss_index',
                       help='Директория с индексом Faiss')
    parser.add_argument('--chunks-dir', type=str, default='chunks',
                       help='Директория с чанками')
    parser.add_argument('--questions-file', type=str, default='questions.md',
                       help='Файл с вопросами')
    parser.add_argument('--ground-truth-file', type=str, default='ground_truth.json',
                       help='Файл с ground truth (релевантные чанки)')
    parser.add_argument('--dense-model', type=str, default='intfloat/multilingual-e5-large',
                       help='Модель для dense эмбеддингов')
    parser.add_argument('--device', type=str, default=None,
                       help='Устройство для вычислений (mps/cuda/cpu)')
    parser.add_argument('--k-values', type=int, nargs='+', default=[5, 10],
                       help='Значения k для метрик')
    parser.add_argument('--search-types', type=str, nargs='+', 
                       choices=['dense', 'sparse', 'hybrid'], 
                       default=['dense', 'sparse', 'hybrid'],
                       help='Типы поиска для сравнения')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Файл для сохранения результатов')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Оценка качества retrieval")
    print("=" * 60)
    
    # Загружаем вопросы
    print(f"\nЗагрузка вопросов из {args.questions_file}...")
    questions = load_questions(args.questions_file)
    print(f"Загружено вопросов: {len(questions)}")
    
    # Загружаем ground truth
    print(f"\nЗагрузка ground truth из {args.ground_truth_file}...")
    ground_truth = load_ground_truth(args.ground_truth_file)
    print(f"Загружено ground truth для {len(ground_truth)} вопросов")
    
    if not ground_truth:
        print("Warning: Ground truth пуст. Создайте файл ground_truth.json с релевантными чанками.")
        print("Формат: {\"1\": [0, 5, 12], \"2\": [3, 7], ...}")
        exit()
    
    # Инициализируем оценщик
    device = get_device(args.device) if args.device else None
    evaluator = RetrievalEvaluator(
        args.faiss_index_dir,
        args.chunks_dir,
        args.dense_model,
        device=device
    )
    
    # Оцениваем для каждого типа поиска
    all_results = {
        "config": {
            "dense_model": args.dense_model,
            "device": device,
            "k_values": args.k_values
        },
        "evaluations": []
    }
    
    for search_type in args.search_types:
        print(f"\n{'='*60}")
        print(f"Оценка для типа поиска: {search_type}")
        print(f"{'='*60}")
        
        results = evaluate_retrieval(
            evaluator,
            questions,
            ground_truth,
            search_type=search_type,
            k_values=args.k_values
        )
        
        all_results["evaluations"].append(results)
        
        # Выводим результаты
        print(f"\nМетрики для {search_type}:")
        for metric_name, metric_value in results["metrics"].items():
            print(f"  {metric_name}: {metric_value['mean']:.4f} ± {metric_value['std']:.4f}")
    
    # Сохраняем результаты
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("Оценка завершена!")
    print(f"Результаты сохранены в: {output_path}")
    print("=" * 60)
    
    # Выводим сравнение через tabulate
    print("\nСравнение типов поиска:")
    
    # Формируем заголовки таблицы
    headers = ["Тип поиска"]
    for k in args.k_values:
        headers.extend([f"Recall@{k}", f"Precision@{k}"])
    headers.append("MRR")
    
    # Формируем данные для таблицы
    table_data = []
    for eval_result in all_results["evaluations"]:
        search_type = eval_result["search_type"]
        row = [search_type]
        for k in args.k_values:
            recall = eval_result["metrics"][f"Recall@{k}"]["mean"]
            precision = eval_result["metrics"][f"Precision@{k}"]["mean"]
            row.extend([f"{recall:.4f}", f"{precision:.4f}"])
        mrr = eval_result["metrics"]["MRR"]["mean"]
        row.append(f"{mrr:.4f}")
        table_data.append(row)
    
    # Выводим таблицу
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))