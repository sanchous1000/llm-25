import yaml
import sys
sys.path.append('scripts')

from agentic import model, client, collection_name


def load_evaluation_queries(path: str) -> list[dict[str, any]]:
    return yaml.safe_load(open(path))['queries']


def match_chunk_with_query(chunk_metadata: dict, query_relevant_doc: dict) -> bool:
    if not chunk_metadata['document']['source'].endswith(query_relevant_doc['document']):
        return False
    chunk_headers = chunk_metadata['md_header']
    for expected_headers in query_relevant_doc['md_headers']:
        if all(chunk_headers.get(k) == v for k, v in expected_headers.items() if k != 'id'):
            return True
    return False


def get_relevant_chunk_ids(retrieved_chunks: list, query_data: dict) -> tuple[list[int], list[int]]:
    retrieved_ids = [p.id for p in retrieved_chunks]
    relevant_ids = [p.id for p in retrieved_chunks 
                    if any(match_chunk_with_query(p.payload['metadata'], doc) 
                           for doc in query_data['relevant_docs'])]
    return retrieved_ids, relevant_ids


def recall_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / len(relevant_ids) if relevant_ids else 0.0


def precision_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / k if k else 0.0


def reciprocal_rank(retrieved_ids: list[int], relevant_ids: list[int]) -> float:
    relevant_set = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids, 1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    queries: list[dict[str, any]], 
    k_values: list[int] = [5, 10]
) -> dict[str, any]:
    results = []
    for query_data in queries:
        query_embedding = model.encode(query_data['query'])
        retrieved_chunks = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            with_payload=True,
            limit=max(k_values)
        ).points
        
        retrieved_ids, relevant_ids = get_relevant_chunk_ids(retrieved_chunks, query_data)
        
        metrics = {'id': query_data['id'], 'query': query_data['query']}
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
    
    return {'per_query': results, 'aggregate': aggregate}
