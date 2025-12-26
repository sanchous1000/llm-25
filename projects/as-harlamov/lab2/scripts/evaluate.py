#!/usr/bin/env python3
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

from config import Config
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from evaluation import evaluate_batch


def create_retrieval_function(config, embedding_generator, vector_store):
    def retrieve(query: str):
        if config.embeddings.type in ["dense", "hybrid"]:
            embeddings = embedding_generator.generate_dense_embeddings([query])
            query_embedding = embeddings[0].tolist()
            results = vector_store.search(query_embedding, top_k=10)
        else:
            embeddings = embedding_generator.generate_dense_embeddings([query])
            query_embedding = embeddings[0].tolist()
            results = vector_store.search(query_embedding, top_k=10)
        return results
    
    return retrieve


def main():
    config = Config()
    
    project_root = Path(__file__).parent.parent
    test_questions_file = project_root / config.evaluation.get("test_questions_file", "data/test_questions.json")
    if not test_questions_file.exists():
        print(f"Test questions file not found: {test_questions_file}")
        print("Please create test_questions.json with format:")
        print("""
[
  {
    "query": "question text",
    "relevant_chunk_ids": [0, 1, 2]
  }
]
        """)
        return
    
    with open(test_questions_file, "r", encoding="utf-8") as f:
        test_queries = json.load(f)
    
    print(f"Loaded {len(test_queries)} test queries")
    
    embedding_generator = EmbeddingGenerator(config)
    vector_store = VectorStore(config, embedding_generator)
    
    retrieve = create_retrieval_function(config, embedding_generator, vector_store)
    
    k_values = config.evaluation.get("k_values", [5, 10])
    print(f"\nEvaluating with k_values: {k_values}")
    print(f"Configuration:")
    print(f"  Chunking: {config.chunking.strategy}, size={config.chunking.chunk_size}")
    print(f"  Embeddings: {config.embeddings.type}, model={config.embeddings.model}")
    
    metrics = evaluate_batch(test_queries, retrieve, k_values)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:.4f}")
    print("="*50)
    
    results_file = project_root / "data/evaluation_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "config": {
            "chunking": {
                "strategy": config.chunking.strategy,
                "chunk_size": config.chunking.chunk_size,
                "chunk_overlap": config.chunking.chunk_overlap,
            },
            "embeddings": {
                "type": config.embeddings.type,
                "model": config.embeddings.model,
            },
        },
        "metrics": metrics,
    }
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()

