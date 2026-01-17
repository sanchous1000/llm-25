"""
Скрипт для оценки качества retrieval и QA.

Вычисляет метрики:
- Recall@k
- Precision@k
- MRR (Mean Reciprocal Rank)
"""
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from elasticsearch import Elasticsearch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

from config_utils import load_config
from embeddings import DenseEmbedder, format_text_for_e5
from es_utils import get_es_client


def load_questions(
    questions_file: Path,
    expected_chunks_map: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Загружает вопросы из файла.

    Формат файла: один вопрос на строку.
    Ожидаемые чанки задаются в JSON: {question: [chunk_id, ...]}.
    """
    questions = []
    with open(questions_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line and not line.startswith("#"):
                expected = []
                if expected_chunks_map and line in expected_chunks_map:
                    expected = expected_chunks_map[line]
                questions.append(
                    {
                        "id": i,
                        "question": line,
                        "expected_chunks": expected,
                    }
                )
    return questions


def search_chunks(
    es_client: Elasticsearch,
    index_name: str,
    query_embedding: list[float],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Ищет релевантные чанки через script_score (cosineSimilarity)."""
    script_query = {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_embedding},
                },
            }
        },
        "_source": ["chunk_id", "text", "source_path", "title", "header", "metadata"],
        "size": top_k,
    }

    response = es_client.search(index=index_name, body=script_query)

    results = []
    for hit in response["hits"]["hits"]:
        results.append(
            {
                "chunk_id": hit["_source"]["chunk_id"],
                "text": hit["_source"]["text"],
                "source_path": hit["_source"].get("source_path", ""),
                "title": hit["_source"].get("title", ""),
                "header": hit["_source"].get("header", ""),
                "score": hit["_score"],
                "metadata": hit["_source"].get("metadata", {}),
            }
        )

    return results


def load_expected_chunks(expected_file: Path) -> dict[str, list[str]]:
    """Загружает эталонные чанки из JSON файла."""
    with open(expected_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("expected_chunks_file must be a JSON object")
    return {str(k): list(v) for k, v in data.items()}


def calculate_recall_at_k(
    retrieved_chunks: list[str],
    relevant_chunks: list[str],
    k: int,
) -> float:
    """Вычисляет Recall@k.
    
    Args:
        retrieved_chunks: ID полученных чанков.
        relevant_chunks: ID релевантных чанков.
        k: Значение k.
    
    Returns:
        Recall@k (0.0 - 1.0).
    """
    if not relevant_chunks:
        return 0.0
    
    retrieved_k = set(retrieved_chunks[:k])
    relevant_set = set(relevant_chunks)
    
    intersection = retrieved_k & relevant_set
    return len(intersection) / len(relevant_set) if relevant_set else 0.0


def calculate_precision_at_k(
    retrieved_chunks: list[str],
    relevant_chunks: list[str],
    k: int,
) -> float:
    """Вычисляет Precision@k.
    
    Args:
        retrieved_chunks: ID полученных чанков.
        relevant_chunks: ID релевантных чанков.
        k: Значение k.
    
    Returns:
        Precision@k (0.0 - 1.0).
    """
    if not retrieved_chunks[:k]:
        return 0.0
    
    retrieved_k = set(retrieved_chunks[:k])
    relevant_set = set(relevant_chunks)
    
    intersection = retrieved_k & relevant_set
    return len(intersection) / k


def calculate_mrr(
    retrieved_chunks: list[str],
    relevant_chunks: list[str],
) -> float:
    """Вычисляет Mean Reciprocal Rank.
    
    Args:
        retrieved_chunks: ID полученных чанков.
        relevant_chunks: ID релевантных чанков.
    
    Returns:
        MRR (0.0 - 1.0).
    """
    if not relevant_chunks:
        return 0.0
    
    relevant_set = set(relevant_chunks)
    
    for rank, chunk_id in enumerate(retrieved_chunks, 1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    
    return 0.0


def evaluate(
    config: dict[str, Any],
    version: str | None = None,
) -> dict[str, Any]:
    """Выполняет оценку качества retrieval.
    
    Args:
        config: Конфигурация.
        version: Версия индекса (если не указана, используется последняя).
    
    Returns:
        Словарь с метриками.
    """
    script_dir = Path(__file__).parent
    lab2_dir = script_dir.parent
    
    # Определяем версию
    if not version:
        index_dir = lab2_dir / "data" / "index"
        # Сортируем по времени модификации (самая новая последняя)
        version_dirs = [d for d in index_dir.iterdir() if d.is_dir()]
        version_dirs.sort(key=lambda d: d.stat().st_mtime)
        versions = [d.name for d in version_dirs]
        if not versions:
            raise ValueError("No index versions found.")
        version = versions[-1]
        print(f"Using latest version: {version}")
    
    # Загружаем конфигурацию
    es_config = config.get("elasticsearch", {})
    eval_config = config.get("evaluation", {})
    rag_config = config.get("rag", {})
    embeddings_config = config.get("embeddings", {})
    
    # Подключаемся к Elasticsearch
    es_client, es_url = get_es_client(es_config)
    print(f"Connected to Elasticsearch: {es_url}")
    
    if not es_client.ping():
        raise ConnectionError("Cannot connect to Elasticsearch")
    
    index_name = es_config.get("index_name", "fastapi_docs")
    
    # Загружаем вопросы
    questions_file = lab2_dir / eval_config.get("questions_file", "source/questions.txt")
    expected_file = eval_config.get("expected_chunks_file")
    expected_map = None
    if expected_file:
        expected_map = load_expected_chunks(lab2_dir / expected_file)
    questions = load_questions(questions_file, expected_map)
    
    print(f"Loaded {len(questions)} questions")
    
    # Создаем embedder для запросов
    dense_config = embeddings_config.get("dense", {})
    model_name = dense_config.get("model", "intfloat/multilingual-e5-base")
    embedder = DenseEmbedder(
        model_name=model_name,
        device=dense_config.get("device", "cpu"),
    )
    
    # Форматируем вопросы для e5
    is_e5_model = "e5" in model_name.lower()
    
    # Вычисляем метрики
    k_values = eval_config.get("k_values", [5, 10])
    metrics = {}
    for k in k_values:
        metrics[f"recall@{k}"] = []
        metrics[f"precision@{k}"] = []
    metrics["mrr"] = []
    
    print("Evaluating retrieval quality...")
    
    evaluated = 0
    skipped = 0
    for question in tqdm(questions, desc="Processing questions"):
        # Создаем эмбеддинг запроса
        query_text = question["question"]
        if is_e5_model:
            query_text = format_text_for_e5(query_text, prefix="query: ")
        
        query_embedding = embedder.embed([query_text])[0].tolist()
        
        # Ищем релевантные чанки
        max_k = max(k_values)
        retrieved = search_chunks(es_client, index_name, query_embedding, top_k=max_k)
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        
        # Для демонстрации: считаем все релевантными (в реальности нужны эталонные ответы)
        # В будущем можно добавить файл с эталонными chunk_id для каждого вопроса
        expected = question.get("expected_chunks")
        if not expected:
            skipped += 1
            continue
        relevant_ids = expected
        evaluated += 1
        
        # Вычисляем метрики
        for k in k_values:
            metrics[f"recall@{k}"].append(
                calculate_recall_at_k(retrieved_ids, relevant_ids, k)
            )
            metrics[f"precision@{k}"].append(
                calculate_precision_at_k(retrieved_ids, relevant_ids, k)
            )
        
        metrics["mrr"].append(calculate_mrr(retrieved_ids, relevant_ids))
    
    # Вычисляем средние значения
    if evaluated == 0:
        raise ValueError(
            "No questions with expected_chunks found. "
            "Provide data/evaluation/expected_chunks.json and set "
            "evaluation.expected_chunks_file in config.yaml."
        )

    results = {
        "version": version,
        "num_questions": len(questions),
        "num_evaluated": evaluated,
        "num_skipped": skipped,
        "metrics": {},
    }
    
    for metric_name, values in metrics.items():
        results["metrics"][metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }
    
    return results


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument(
        "--version",
        type=str,
        help="Version of index to evaluate",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)",
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    lab2_dir = script_dir.parent
    
    config_path = lab2_dir / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_config(config_path)
    
    results = evaluate(config, args.version)
    
    # Выводим результаты
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Version: {results['version']}")
    print(f"Questions: {results['num_questions']}")
    print("\nMetrics:")
    for metric_name, metric_data in results["metrics"].items():
        print(f"  {metric_name}: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}")
    print("=" * 60)
    
    # Сохраняем результаты
    if args.output:
        output_path = lab2_dir / args.output
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
    else:
        # Сохраняем в data/evaluation/
        eval_dir = lab2_dir / "data" / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        output_path = eval_dir / f"results_{results['version']}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
