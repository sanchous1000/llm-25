import sys
sys.path.append('../lab2/scripts')

from langfuse import Langfuse
from evaluation_metrics import (
    load_evaluation_queries,
    match_chunk_with_query,
    get_relevant_chunk_ids,
    recall_at_k,
    precision_at_k,
    reciprocal_rank
)
from agentic import model, client, collection_name
import os
from dotenv import load_dotenv
import json
import yaml

load_dotenv()

# Load config
with open("scripts/config.yaml") as f:
    config = yaml.safe_load(f)

experiment_config = config["experiment"]

# Initialize Langfuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

# Config
DATASET_NAME = experiment_config["dataset_name"]
EXPERIMENT_NAME = experiment_config["experiment_name"]
K_VALUES = experiment_config["k_values"]
QUERIES_PATH = experiment_config["evaluation_queries_path"]


def run_experiment():
    """
    Запуск эксперимента по оценке RAG системы через Langfuse SDK
    """
    
    # Загрузка запросов
    queries = load_evaluation_queries(QUERIES_PATH)
    
    print(f"Running experiment: {EXPERIMENT_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Queries: {len(queries)}")
    
    all_results = []
    
    # Прогон по всем запросам
    for query_data in queries:
        query_text = query_data['query']
        query_id = query_data['id']
        
        print(f"\n[{query_id}] {query_text}")
        
        # Retrieval
        query_embedding = model.encode(query_text)
        retrieved_chunks = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            with_payload=True,
            limit=max(K_VALUES)
        ).points
        
        # Расчет метрик
        retrieved_ids, relevant_ids = get_relevant_chunk_ids(retrieved_chunks, query_data)
        
        metrics = {}
        for k in K_VALUES:
            metrics[f'recall@{k}'] = recall_at_k(retrieved_ids, relevant_ids, k)
            metrics[f'precision@{k}'] = precision_at_k(retrieved_ids, relevant_ids, k)
        metrics['reciprocal_rank'] = reciprocal_rank(retrieved_ids, relevant_ids)
        
        print(f"  Recall@5: {metrics['recall@5']:.2%}")
        print(f"  Recall@10: {metrics['recall@10']:.2%}")
        print(f"  Precision@5: {metrics['precision@5']:.2%}")
        print(f"  Precision@10: {metrics['precision@10']:.2%}")
        print(f"  MRR: {metrics['reciprocal_rank']:.3f}")
        
        # Создание trace для этого запроса
        trace = langfuse.trace(
            name=f"eval_query_{query_id}",
            metadata={"query_id": query_id, "experiment": EXPERIMENT_NAME}
        )
        
        # Span для retrieval
        retrieval_span = trace.span(
            name="retrieval",
            input={"query": query_text, "limit": max(K_VALUES)},
            output={
                "retrieved_count": len(retrieved_ids),
                "relevant_count": len(relevant_ids),
                "retrieved_ids": retrieved_ids[:max(K_VALUES)]
            }
        )
        
        # Логирование evaluations
        for metric_name, metric_value in metrics.items():
            langfuse.score(
                trace_id=trace.id,
                name=metric_name,
                value=metric_value
            )
        
        all_results.append({
            'query_id': query_id,
            'query': query_text,
            **metrics
        })
    
    # Агрегированные метрики
    aggregate = {}
    for k in K_VALUES:
        aggregate[f'recall@{k}'] = sum(r[f'recall@{k}'] for r in all_results) / len(all_results)
        aggregate[f'precision@{k}'] = sum(r[f'precision@{k}'] for r in all_results) / len(all_results)
    aggregate['MRR'] = sum(r['reciprocal_rank'] for r in all_results) / len(all_results)
    
    print("\n" + "="*50)
    print("AGGREGATE METRICS:")
    print("="*50)
    for metric_name, metric_value in aggregate.items():
        print(f"{metric_name}: {metric_value:.2%}" if 'recall' in metric_name or 'precision' in metric_name else f"{metric_name}: {metric_value:.3f}")
    
    # Сохранение результатов
    results = {
        'experiment_name': EXPERIMENT_NAME,
        'dataset_name': DATASET_NAME,
        'per_query': all_results,
        'aggregate': aggregate
    }
    
    with open('results/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    langfuse.flush()
    
    print(f"\n✓ Experiment completed!")
    print(f"View results in Langfuse UI: {os.getenv('LANGFUSE_HOST')}")
    print(f"Results saved to: results/experiment_results.json")


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    run_experiment()

