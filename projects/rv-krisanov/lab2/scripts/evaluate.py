import argparse
import yaml
import json
import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

sys.path.append('scripts')
from evaluation_metrics import load_evaluation_queries, get_relevant_chunk_ids, recall_at_k, precision_at_k, reciprocal_rank


def evaluate(config_path: str, output_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    agent_config = config['agent']
    collection_name = agent_config['collection_name']
    embedding_model_name = agent_config['embedding_model']
    k_values = [5, 10]
    
    print(f"=== Evaluation: {config_path} ===")
    print(f"Collection: {collection_name}")
    print(f"Embedding model: {embedding_model_name}")
    
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    model = SentenceTransformer(embedding_model_name)
    
    queries = load_evaluation_queries("evaluation_queries.yaml")
    
    results = []
    for query_data in queries:
        query_text = query_data['query']
        query_id = query_data['id']
        
        query_embedding = model.encode(query_text)
        retrieved_chunks = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            with_payload=True,
            limit=max(k_values)
        ).points
        
        retrieved_ids, relevant_ids = get_relevant_chunk_ids(retrieved_chunks, query_data)
        
        metrics = {'id': query_id, 'query': query_text}
        for k in k_values:
            metrics[f'recall@{k}'] = recall_at_k(retrieved_ids, relevant_ids, k)
            metrics[f'precision@{k}'] = precision_at_k(retrieved_ids, relevant_ids, k)
        metrics['reciprocal_rank'] = reciprocal_rank(retrieved_ids, relevant_ids)
        results.append(metrics)
    
    aggregate = {}
    for k in k_values:
        aggregate[f'recall@{k}'] = sum(r[f'recall@{k}'] for r in results) / len(results)
        aggregate[f'precision@{k}'] = sum(r[f'precision@{k}'] for r in results) / len(results)
    aggregate['MRR'] = sum(r['reciprocal_rank'] for r in results) / len(results)
    
    print("\nAGGREGATE METRICS:")
    for metric_name, metric_value in aggregate.items():
        if 'recall' in metric_name or 'precision' in metric_name:
            print(f"  {metric_name}: {metric_value:.2%}")
        else:
            print(f"  {metric_name}: {metric_value:.3f}")
    
    output = {
        'config': config_path,
        'collection_name': collection_name,
        'per_query': results,
        'aggregate': aggregate
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}\n")
    
    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    args = parser.parse_args()
    
    evaluate(args.config, args.output)

