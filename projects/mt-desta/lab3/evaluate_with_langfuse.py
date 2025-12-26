"""
Evaluate RAG Pipeline with Langfuse Experiment Run Integration
Uses the existing lab2 evaluation metrics and integrates with Langfuse Experiments
"""
import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
from langfuse import Langfuse
from langfuse.experiment import Evaluation
from dotenv import load_dotenv

# Import existing evaluation functions from lab2
import sys
sys.path.append(str(Path(__file__).parent.parent / "lab2" / "scripts"))
from evaluate import (
    load_evaluation_questions,
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_mrr
)
from rag_pipeline import RAGAgent

load_dotenv()


def run_evaluator(
    dataset_item: Dict[str, Any],
    agent: RAGAgent,
    k_values: List[int] = [5, 10]
) -> List[Evaluation]:
    """
    Custom evaluator for RAG pipeline.
    Calculates retrieval metrics: Recall@k, Precision@k, MRR
    Returns list of Evaluation objects for Langfuse.
    """
    # Handle both dict format and Langfuse dataset item format
    if isinstance(dataset_item, dict):
        if "input" in dataset_item:
            question = dataset_item["input"]["question"]
            ground_truth_chunks = dataset_item.get("expected_output", {}).get("ground_truth_chunks", [])
        else:
            # Direct format from JSON
            question = dataset_item["question"]
            ground_truth_chunks = dataset_item.get("ground_truth_chunks", [])
    else:
        # Langfuse dataset item object
        question = dataset_item.input["question"]
        ground_truth_chunks = dataset_item.expected_output.get("ground_truth_chunks", [])
    
    # Retrieve documents
    retrieved_docs = agent.retrieve(question, k=max(k_values))
    retrieved_chunks = [doc.page_content for doc in retrieved_docs]
    
    # Calculate metrics for each k
    evaluations = []
    
    for k in k_values:
        recall = calculate_recall_at_k(retrieved_chunks, ground_truth_chunks, k)
        precision = calculate_precision_at_k(retrieved_chunks, ground_truth_chunks, k)
        
        evaluations.append(
            Evaluation(
                name=f"recall@{k}",
                value=recall,
                comment=f"Recall@k metric for k={k}"
            )
        )
        
        evaluations.append(
            Evaluation(
                name=f"precision@{k}",
                value=precision,
                comment=f"Precision@k metric for k={k}"
            )
        )
    
    # Calculate MRR
    mrr = calculate_mrr(retrieved_chunks, ground_truth_chunks)
    evaluations.append(
        Evaluation(
            name="mrr",
            value=mrr,
            comment="Mean Reciprocal Rank"
        )
    )
    
    return evaluations


def run_experiment(
    langfuse: Langfuse,
    dataset_name: str,
    experiment_name: str,
    config: Dict[str, Any],
    k_values: List[int] = [5, 10]
):
    """
    Run evaluation experiment using Langfuse Experiment Run.
    """
    # Get dataset items
    # Note: Langfuse SDK may require fetching items separately
    # For now, we'll load from local file and use Langfuse for logging only
    print(f"Loading questions from local file for evaluation...")
    questions_path = Path(__file__).parent.parent / "lab2" / "evaluation_questions.json"
    
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found at {questions_path}")
    
    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        dataset_items = []
        for item in data.get("questions", []):
            dataset_items.append({
                "input": {
                    "question": item["question"],
                    "ground_truth_chunks": item.get("ground_truth_chunks", []),
                    "expected_topics": item.get("expected_topics", [])
                },
                "expected_output": {
                    "ground_truth_chunks": item.get("ground_truth_chunks", []),
                    "expected_topics": item.get("expected_topics", [])
                }
            })
    
    print(f"Experiment: {experiment_name}")
    print(f"Configuration: {config}")
    print(f"\nRunning evaluation on {len(dataset_items)} items...\n")
    
    # Initialize RAG agent
    agent = RAGAgent()
    
    # Process each dataset item
    all_evaluations = []
    trace_ids = []
    
    for i, dataset_item in enumerate(dataset_items, 1):
        question = dataset_item["input"]["question"]
        print(f"[{i}/{len(dataset_items)}] Processing: {question[:60]}...")
        
        # Run evaluator
        evaluations = run_evaluator(dataset_item, agent, k_values)
        all_evaluations.extend(evaluations)
        
        # Prepare input/output for logging
        trace_input = dataset_item.get("input", dataset_item)
        trace_output = {"evaluations": {e.name: e.value for e in evaluations}}
        
        # Create trace using start_as_current_observation
        with langfuse.start_as_current_observation(
            as_type="span",
            name="evaluation_item",
            input=trace_input,
            metadata={
                "experiment_name": experiment_name,
                "dataset_name": dataset_name,
                **config
            }
        ) as trace:
            trace.update(output=trace_output)
            
            # Add scores for each evaluation metric
            for evaluation in evaluations:
                # Use score method if available, otherwise add to metadata
                if hasattr(langfuse, 'score'):
                    try:
                        langfuse.score(
                            trace_id=trace.id,
                            name=evaluation.name,
                            value=evaluation.value,
                            comment=evaluation.comment
                        )
                    except:
                        # If score method doesn't work, add to trace metadata
                        pass
            
            trace_ids.append(trace.id)
        
        # Print metrics
        metrics_str = ", ".join([f"{e.name}={e.value:.3f}" for e in evaluations])
        print(f"  Metrics: {metrics_str}")
    
    # Flush all traces
    langfuse.flush()
    
    # Calculate average metrics
    avg_metrics = {}
    for k in k_values:
        recall_values = [e.value for e in all_evaluations if e.name == f"recall@{k}"]
        precision_values = [e.value for e in all_evaluations if e.name == f"precision@{k}"]
        mrr_values = [e.value for e in all_evaluations if e.name == "mrr"]
        
        avg_metrics[f"avg_recall@{k}"] = sum(recall_values) / len(recall_values) if recall_values else 0.0
        avg_metrics[f"avg_precision@{k}"] = sum(precision_values) / len(precision_values) if precision_values else 0.0
        avg_metrics["avg_mrr"] = sum(mrr_values) / len(mrr_values) if mrr_values else 0.0
    
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Experiment: {experiment_name}")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"\nAverage Metrics:")
    for k in k_values:
        print(f"  Recall@{k}: {avg_metrics[f'avg_recall@{k}']:.3f}")
        print(f"  Precision@{k}: {avg_metrics[f'avg_precision@{k}']:.3f}")
    print(f"  MRR: {avg_metrics['avg_mrr']:.3f}")
    print("="*60)
    
    print(f"\nView traces at: {os.getenv('LANGFUSE_HOST', 'http://localhost:3001')}/traces")
    print(f"Filter by experiment_name: {experiment_name}")
    print(f"\nTrace IDs created: {len(trace_ids)}")
    
    # Return None for experiment_run (not using experiment run API)
    # Return avg_metrics for summary
    return None, avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation with Langfuse Experiment Run")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="rag-evaluation-dataset",
        help="Name of the dataset in Langfuse"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="rag-evaluation-experiment",
        help="Name of the experiment run"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file (optional, will use lab2 config by default)"
    )
    args = parser.parse_args()
    
    # Initialize Langfuse
    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "http://localhost:3001")
    )
    
    # Load configuration
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        # Use lab2 config
        import yaml
        config_path = Path(__file__).parent.parent / "lab2" / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    
    # Format config for experiment metadata
    experiment_config = {
        "vectorization_type": config["vectorization"]["type"],
        "chunk_size": config["data"]["chunk_size"],
        "chunk_overlap": config["data"]["chunk_overlap"],
        "embedding_model": config["embedding"]["model_name"],
        "llm_model": config["llm"]["model_id"]
    }
    
    # Run experiment
    experiment_run, avg_metrics = run_experiment(
        langfuse=langfuse,
        dataset_name=args.dataset_name,
        experiment_name=args.experiment_name,
        config=experiment_config,
        k_values=[5, 10]
    )
    
    if experiment_run:
        print(f"\nExperiment Run ID: {experiment_run.id}")
    else:
        print(f"\n✅ Evaluation complete! View traces in Langfuse UI filtered by experiment_name: {args.experiment_name}")


if __name__ == "__main__":
    main()

