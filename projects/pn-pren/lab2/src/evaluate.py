import json
from pathlib import Path
from typing import List, Dict
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np

import config


def calculate_recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved_k = retrieved_ids[:k]
    hits = len(set(retrieved_k) & set(relevant_ids))
    return hits / len(relevant_ids) if relevant_ids else 0


def calculate_precision_at_k(retrieved_ids, relevant_ids, k):
    retrieved_k = retrieved_ids[:k]
    hits = len(set(retrieved_k) & set(relevant_ids))
    return hits / k if k > 0 else 0


def calculate_mrr(retrieved_ids, relevant_ids):
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def evaluate_retrieval(queries_file, collection_name=None, k_values=[5, 10]):
    if collection_name is None:
        collection_name = config.COLLECTION_NAME
    
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    
    results = {k: {'recall': [], 'precision': []} for k in k_values}
    mrr_scores = []
    
    for query_data in queries:
        query = query_data['query']
        relevant_chunks = query_data['relevant_chunks']
        
        query_vector = model.encode([query])[0].tolist()
        
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=max(k_values)
        ).points
        
        retrieved_ids = [hit.id for hit in search_results]
        
        for k in k_values:
            recall = calculate_recall_at_k(retrieved_ids, relevant_chunks, k)
            precision = calculate_precision_at_k(retrieved_ids, relevant_chunks, k)
            results[k]['recall'].append(recall)
            results[k]['precision'].append(precision)
        
        mrr = calculate_mrr(retrieved_ids, relevant_chunks)
        mrr_scores.append(mrr)
    
    print("Metrics:\n")
    
    for k in k_values:
        avg_recall = np.mean(results[k]['recall'])
        avg_precision = np.mean(results[k]['precision'])
        print(f"K={k}")
        print(f"  Recall@{k}: {avg_recall:.4f}")
        print(f"  Precision@{k}: {avg_precision:.4f}")
    
    avg_mrr = np.mean(mrr_scores)
    print(f"\nMRR: {avg_mrr:.4f}")
    
    metrics = {
        'avg_metrics': {
            f'recall@{k}': float(np.mean(results[k]['recall'])) for k in k_values
        },
        'mrr': float(avg_mrr),
        'queries': len(queries),
        'config': {
            'chunk_size': config.CHUNK_SIZE,
            'overlap': config.OVERLAP,
            'splitter_type': config.SPLITTER_TYPE,
            'embedding_model': config.EMBEDDING_MODEL
        }
    }
    
    for k in k_values:
        metrics['avg_metrics'][f'precision@{k}'] = float(np.mean(results[k]['precision']))
    
    output_file = Path('output/metrics.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to {output_file}")
    return metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', default='data/eval_queries.json')
    args = parser.parse_args()
    
    evaluate_retrieval(args.queries, config.COLLECTION_NAME)
