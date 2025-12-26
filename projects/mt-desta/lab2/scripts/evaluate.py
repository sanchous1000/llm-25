import yaml
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from rag_pipeline import RAGAgent

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
EVAL_QUESTIONS_PATH = Path(__file__).parent.parent / "evaluation_questions.json"

def load_evaluation_questions():
    """Load evaluation questions with ground truth from JSON file."""
    if not EVAL_QUESTIONS_PATH.exists():
        print(f"Warning: {EVAL_QUESTIONS_PATH} not found. Using default questions.")
        return [
            {
                "question": "What is the main topic of the documents?",
                "ground_truth_chunks": [],
                "expected_topics": []
            }
        ]
    with open(EVAL_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("questions", [])

def calculate_recall_at_k(retrieved_chunks: List[str], ground_truth_chunks: List[str], k: int) -> float:
    """Calculate Recall@k: fraction of ground truth chunks found in top-k retrieved chunks."""
    if not ground_truth_chunks:
        return 0.0
    
    retrieved_top_k = retrieved_chunks[:k]
    # Check if any ground truth chunk identifier appears in retrieved chunks
    found = 0
    for gt_chunk in ground_truth_chunks:
        # Check if ground truth identifier appears in any retrieved chunk
        for ret_chunk in retrieved_top_k:
            if gt_chunk.lower() in ret_chunk.lower():
                found += 1
                break
    
    return found / len(ground_truth_chunks) if ground_truth_chunks else 0.0

def calculate_precision_at_k(retrieved_chunks: List[str], ground_truth_chunks: List[str], k: int) -> float:
    """Calculate Precision@k: fraction of top-k retrieved chunks that are relevant."""
    if k == 0:
        return 0.0
    
    retrieved_top_k = retrieved_chunks[:k]
    if not retrieved_top_k:
        return 0.0
    
    # Check if retrieved chunks contain ground truth identifiers
    relevant = 0
    for ret_chunk in retrieved_top_k:
        for gt_chunk in ground_truth_chunks:
            if gt_chunk.lower() in ret_chunk.lower():
                relevant += 1
                break
    
    return relevant / len(retrieved_top_k)

def calculate_mrr(retrieved_chunks: List[str], ground_truth_chunks: List[str]) -> float:
    """Calculate Mean Reciprocal Rank: 1/rank of first relevant chunk."""
    if not ground_truth_chunks:
        return 0.0
    
    for rank, ret_chunk in enumerate(retrieved_chunks, start=1):
        for gt_chunk in ground_truth_chunks:
            if gt_chunk.lower() in ret_chunk.lower():
                return 1.0 / rank
    
    return 0.0

def evaluate_retrieval(agent: RAGAgent, questions: List[Dict[str, Any]], k_values: List[int] = [5, 10]) -> Dict[str, Any]:
    """Evaluate retrieval performance with Recall@k, Precision@k, and MRR."""
    results = []
    all_recall_at_k = {k: [] for k in k_values}
    all_precision_at_k = {k: [] for k in k_values}
    all_mrr = []
    
    print("Running evaluation...")
    for i, item in enumerate(questions, 1):
        question = item["question"]
        ground_truth_chunks = item.get("ground_truth_chunks", [])
        expected_topics = item.get("expected_topics", [])
        
        print(f"\n[{i}/{len(questions)}] Processing: {question}")
        
        # Retrieve documents
        retrieved_docs = agent.retrieve(question, k=max(k_values))
        retrieved_chunks = [doc.page_content for doc in retrieved_docs]
        
        # Calculate metrics for each k
        metrics = {}
        for k in k_values:
            recall = calculate_recall_at_k(retrieved_chunks, ground_truth_chunks, k)
            precision = calculate_precision_at_k(retrieved_chunks, ground_truth_chunks, k)
            metrics[f"recall@{k}"] = recall
            metrics[f"precision@{k}"] = precision
            all_recall_at_k[k].append(recall)
            all_precision_at_k[k].append(precision)
        
        mrr = calculate_mrr(retrieved_chunks, ground_truth_chunks)
        metrics["mrr"] = mrr
        all_mrr.append(mrr)
        
        # Get full query result for answer quality
        query_result = agent.query(question)
        
        result = {
            "question": question,
            "ground_truth_chunks": ground_truth_chunks,
            "expected_topics": expected_topics,
            "metrics": metrics,
            "retrieved_sources": [doc.metadata.get("source", "Unknown") for doc in retrieved_docs],
            "answer": query_result["answer"],
            "sources": query_result["sources"]
        }
        results.append(result)
        
        print(f"  Recall@5: {metrics['recall@5']:.3f}, Precision@5: {metrics['precision@5']:.3f}, MRR: {mrr:.3f}")
    
    # Calculate average metrics
    avg_metrics = {}
    for k in k_values:
        avg_metrics[f"avg_recall@{k}"] = sum(all_recall_at_k[k]) / len(all_recall_at_k[k]) if all_recall_at_k[k] else 0.0
        avg_metrics[f"avg_precision@{k}"] = sum(all_precision_at_k[k]) / len(all_precision_at_k[k]) if all_precision_at_k[k] else 0.0
    avg_metrics["avg_mrr"] = sum(all_mrr) / len(all_mrr) if all_mrr else 0.0
    
    return {
        "configuration": load_config(),
        "summary": avg_metrics,
        "detailed_results": results
    }

def load_config():
    """Load current configuration."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def evaluate():
    """Main evaluation function."""
    questions = load_evaluation_questions()
    agent = RAGAgent()
    
    results = evaluate_retrieval(agent, questions, k_values=[5, 10])
    
    # Save results
    output_path = Path(__file__).parent.parent / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Configuration:")
    config = results["configuration"]
    print(f"  Splitter: {config['splitter']['type']}")
    print(f"  Chunk size: {config['data']['chunk_size']}")
    print(f"  Chunk overlap: {config['data']['chunk_overlap']}")
    print(f"  Vectorization: {config['vectorization']['type']}")
    print(f"  Embedding model: {config['embedding']['model_name']}")
    print(f"\nAverage Metrics:")
    summary = results["summary"]
    for k in [5, 10]:
        print(f"  Recall@{k}: {summary[f'avg_recall@{k}']:.3f}")
        print(f"  Precision@{k}: {summary[f'avg_precision@{k}']:.3f}")
    print(f"  MRR: {summary['avg_mrr']:.3f}")
    print("="*60)
    print(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG system with retrieval metrics")
    parser.add_argument("--questions", type=str, help="Path to evaluation questions JSON file")
    args = parser.parse_args()
    
    if args.questions:
        # Update the module-level variable (no global needed at module level)
        EVAL_QUESTIONS_PATH = Path(args.questions)
    
    evaluate()
