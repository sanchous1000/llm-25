import json
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import IndexConfig

def evaluate_retrieval(config: IndexConfig, eval_file="data/eval_dataset.json"):
    client = QdrantClient(path=config.storage_path)
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    
    with open(eval_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)

    k = 5
    hits = 0
    mrr_sum = 0
    
    for item in qa_pairs:
        query = item['question']
        target = item['expected_source_substring']
        
        query_vec = embeddings.embed_query(f"query: {query}")
        
        try:
            search_result = client.search(
                collection_name=config.collection_name,
                query_vector=query_vec,
                limit=k
            )
        except AttributeError:
            search_result = client.query_points(
                collection_name=config.collection_name,
                query=query_vec,
                limit=k
            ).points
        
        found = False
        for rank, hit in enumerate(search_result):
            if target.lower() in hit.payload['text'].lower():
                hits += 1
                mrr_sum += 1 / (rank + 1)
                found = True
                break
    
    recall_at_k = hits / len(qa_pairs)
    mrr = mrr_sum / len(qa_pairs)
    
    print(f"Config: Chunk={config.chunk_size}, Overlap={config.chunk_overlap}")
    print(f"Recall@{k}: {recall_at_k:.2f}")
    print(f"MRR: {mrr:.2f}")
    return recall_at_k, mrr
