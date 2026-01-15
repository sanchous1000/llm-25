import os
import json
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from langfuse import Langfuse

from config import Config, load_config
from rag_traced import TracedRAGPipeline, RAGResult


def calculate_precision(retrieved_books: list[str], expected_books: list[str], k: int) -> float:
    relevant = sum(1 for book in retrieved_books[:k] if book in expected_books)
    return relevant / k if k > 0 else 0


def calculate_recall(retrieved_books: list[str], expected_books: list[str]) -> float:
    found = set(retrieved_books) & set(expected_books)
    return len(found) / len(expected_books) if expected_books else 0


def calculate_mrr(retrieved_books: list[str], expected_books: list[str]) -> float:
    for i, book in enumerate(retrieved_books):
        if book in expected_books:
            return 1 / (i + 1)
    return 0


def calculate_keyword_coverage(answer: str, keywords: list[str]) -> float:
    answer_lower = answer.lower()
    found = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return found / len(keywords) if keywords else 0


class ExperimentRunner:
    def __init__(self, config: Config = None):
        if config is None:
            config = load_config()
        self.config = config
        
        # Initialize Langfuse client directly (not singleton)
        self.langfuse = Langfuse(
            public_key=config.langfuse_public_key,
            secret_key=config.langfuse_secret_key,
            host=config.langfuse_host,
        )
        print(f"[DEBUG] Langfuse client initialized")
        
        self.pipeline = TracedRAGPipeline(config)
    
    def run_experiment(self, experiment_name: str, top_k: int = 5):
        run_name = experiment_name
        
        print(f"Running experiment: {run_name}")
        print(f"Dataset: {self.config.dataset_name}")
        print(f"Parameters: top_k={top_k}")
        print("-" * 50)
        
        # Get dataset from Langfuse
        dataset = self.langfuse.get_dataset(self.config.dataset_name)
        
        results = []
        
        # Manually iterate over items and use item.run() context manager
        for idx, item in enumerate(dataset.items):
            print(f"\nProcessing item {idx + 1}/{len(dataset.items)}")
            question = item.input.get("question", "")
            print(f"  Question: {question[:50]}...")
            
            # Use item.run() to create a linked trace
            with item.run(run_name=run_name) as run:
                # Execute the RAG pipeline
                rag_result: RAGResult = self.pipeline.query(question, top_k=top_k)
                
                # Set output on the run
                output = {
                    "answer": rag_result.answer,
                    "sources": [
                        {"book": s["book"], "section": s["section"], "score": s["score"]}
                        for s in rag_result.sources
                    ],
                    "retrieval_time": rag_result.retrieval_time,
                    "generation_time": rag_result.generation_time
                }
                run.output = output

                run.update_trace(
                    input={"question": question},
                    output=output,
                    metadata={"top_k": top_k, "experiment": run_name}
                )
                
                # Calculate metrics
                retrieved_books = [s["book"] for s in rag_result.sources]
                expected_output = item.expected_output or {}
                expected_books = expected_output.get("expected_books", [])
                keywords = expected_output.get("keywords", [])
                
                precision = calculate_precision(retrieved_books, expected_books, top_k)
                recall = calculate_recall(retrieved_books, expected_books)
                mrr = calculate_mrr(retrieved_books, expected_books)
                keyword_cov = calculate_keyword_coverage(rag_result.answer, keywords)
                
                # Score the run
                run.score(name="precision", value=precision)
                run.score(name="recall", value=recall)
                run.score(name="mrr", value=mrr)
                run.score(name="keyword_coverage", value=keyword_cov)
                
                print(f"  Metrics: P={precision:.2f}, R={recall:.2f}, MRR={mrr:.2f}, KW={keyword_cov:.2f}")
                
                results.append({
                    "question": question,
                    "answer": rag_result.answer,
                    "sources": output["sources"],
                    "metrics": {
                        "precision": precision,
                        "recall": recall,
                        "mrr": mrr,
                        "keyword_coverage": keyword_cov
                    }
                })
        
        # Flush to ensure all data is sent
        self.langfuse.flush()
        print(f"\n[DEBUG] Flushed Langfuse client")
        
        # Save results locally
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f"{run_name}.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {results_file}")
        
        return results


def run_comparison_experiments(config: Config = None):
    if config is None:
        config = load_config()
    
    runner = ExperimentRunner(config)
    runner.run_experiment("ydkjs-experiment", top_k=5)


if __name__ == "__main__":
    run_comparison_experiments()
