"""
Experiment runner for RAG evaluation with Langfuse.
Implements Experiment Run with custom evaluators and retrieval metrics.
"""
import json
import time
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

from langfuse import Langfuse
from langfuse.api.resources.commons.types import DatasetItem
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama
import numpy as np

import config


class RAGExperimentRunner:
    """
    Runs evaluation experiments on RAG pipeline with Langfuse logging.
    """
    
    def __init__(self):
        # Initialize Langfuse
        self.langfuse = Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST
        )
        
        # Initialize RAG components
        self.qdrant_client = QdrantClient(
            host=config.QDRANT_HOST, 
            port=config.QDRANT_PORT
        )
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        self.ollama_base = f"http://{config.OLLAMA_HOST}:{config.OLLAMA_PORT}"
        
        print(f"Experiment Runner initialized")
        print(f"Langfuse: {config.LANGFUSE_HOST}")
        print(f"Qdrant: {config.QDRANT_HOST}:{config.QDRANT_PORT}")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve relevant documents."""
        top_k = top_k or config.TOP_K
        
        query_vector = self.embedder.encode([query])[0].tolist()
        
        results = self.qdrant_client.query_points(
            collection_name=config.COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        ).points
        
        contexts = []
        for hit in results:
            contexts.append({
                'text': hit.payload.get('text', ''),
                'source': hit.payload.get('original_file', 'unknown'),
                'chunk_id': hit.payload.get('chunk_id', hit.id),
                'score': hit.score,
                'id': hit.id
            })
        
        return contexts
    
    def generate_answer(self, query: str, contexts: List[Dict]) -> str:
        """Generate answer using LLM."""
        context_text = "\n\n---\n\n".join([
            f"[{ctx['source']}, chunk {ctx['chunk_id']}]\n{ctx['text']}"
            for ctx in contexts
        ])
        
        prompt = f"""You are a helpful assistant. Answer the question based on the provided context.
Always cite your sources by mentioning the document name and chunk number.

Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            response = ollama.chat(
                model=config.LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'base_url': self.ollama_base}
            )
            return response['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"
    
    # ==================== Retrieval Metrics ====================
    
    @staticmethod
    def calculate_recall_at_k(retrieved_ids: List, relevant_ids: List, k: int) -> float:
        """Calculate Recall@k metric."""
        if not relevant_ids:
            return 0.0
        retrieved_k = retrieved_ids[:k]
        hits = len(set(retrieved_k) & set(relevant_ids))
        return hits / len(relevant_ids)
    
    @staticmethod
    def calculate_precision_at_k(retrieved_ids: List, relevant_ids: List, k: int) -> float:
        """Calculate Precision@k metric."""
        if k == 0:
            return 0.0
        retrieved_k = retrieved_ids[:k]
        hits = len(set(retrieved_k) & set(relevant_ids))
        return hits / k
    
    @staticmethod
    def calculate_mrr(retrieved_ids: List, relevant_ids: List) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def calculate_answer_similarity(generated: str, expected: str, embedder) -> float:
        """Calculate semantic similarity between generated and expected answers."""
        if not expected or not generated:
            return 0.0
        
        embeddings = embedder.encode([generated, expected])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)
    
    # ==================== Experiment Run ====================
    
    def run_experiment(
        self,
        dataset_name: str,
        experiment_name: str = None,
        top_k: int = None,
        k_values: List[int] = [3, 5, 10],
        max_items: int = None
    ) -> Dict:
        """
        Run evaluation experiment on a Langfuse dataset.
        
        Args:
            dataset_name: Name of the dataset in Langfuse
            experiment_name: Name for this experiment run
            top_k: Number of documents to retrieve
            k_values: K values for Recall@k and Precision@k
            max_items: Maximum number of items to evaluate (for testing)
        
        Returns:
            Experiment results and metrics
        """
        top_k = top_k or config.TOP_K
        experiment_name = experiment_name or f"rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"Running Experiment: {experiment_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Top-K: {top_k}")
        print(f"{'='*60}\n")
        
        # Get dataset from Langfuse
        dataset = self.langfuse.get_dataset(dataset_name)
        items = dataset.items
        
        if max_items:
            items = items[:max_items]
        
        print(f"Evaluating {len(items)} items...\n")
        
        # Store metrics for aggregation
        all_metrics = {k: {'recall': [], 'precision': []} for k in k_values}
        mrr_scores = []
        answer_similarities = []
        
        # Process each dataset item
        for i, item in enumerate(items):
            print(f"Processing item {i+1}/{len(items)}: {item.input.get('query', '')[:50]}...")
            
            query = item.input.get('query', '')
            expected_output = item.expected_output or {}
            expected_answer = expected_output.get('answer', '')
            relevant_chunks = expected_output.get('relevant_chunks', [])
            
            start_time = time.time()
            
            # Create trace for this evaluation
            trace = self.langfuse.trace(
                name=f"eval_{experiment_name}",
                input=item.input,
                metadata={
                    "experiment": experiment_name,
                    "dataset": dataset_name,
                    "item_index": i,
                    "top_k": top_k
                }
            )
            
            # Retrieval step with span
            retrieval_span = self.langfuse.span(
                name="retrieval",
                trace_id=trace.id,
                input={"query": query, "top_k": top_k}
            )
            
            contexts = self.retrieve(query, top_k=max(k_values))
            retrieved_ids = [ctx.get('id', ctx.get('chunk_id')) for ctx in contexts]
            
            retrieval_span.end(
                output={
                    "num_results": len(contexts),
                    "retrieved_ids": retrieved_ids[:10]
                }
            )
            
            # Generation step with span
            generation_span = self.langfuse.generation(
                name="llm_generation",
                trace_id=trace.id,
                model=config.LLM_MODEL,
                input={"query": query, "num_contexts": len(contexts[:top_k])}
            )
            
            answer = self.generate_answer(query, contexts[:top_k])
            
            generation_span.end(
                output=answer,
                metadata={"answer_length": len(answer)}
            )
            
            duration = time.time() - start_time
            
            # Calculate metrics
            item_metrics = {}
            
            # Retrieval metrics (if we have relevant_chunks)
            if relevant_chunks:
                for k in k_values:
                    recall = self.calculate_recall_at_k(retrieved_ids, relevant_chunks, k)
                    precision = self.calculate_precision_at_k(retrieved_ids, relevant_chunks, k)
                    all_metrics[k]['recall'].append(recall)
                    all_metrics[k]['precision'].append(precision)
                    item_metrics[f'recall@{k}'] = recall
                    item_metrics[f'precision@{k}'] = precision
                
                mrr = self.calculate_mrr(retrieved_ids, relevant_chunks)
                mrr_scores.append(mrr)
                item_metrics['mrr'] = mrr
            
            # Answer similarity (if we have expected answer)
            if expected_answer:
                similarity = self.calculate_answer_similarity(
                    answer, expected_answer, self.embedder
                )
                answer_similarities.append(similarity)
                item_metrics['answer_similarity'] = similarity
            
            item_metrics['duration_ms'] = round(duration * 1000, 2)
            
            # Update trace with output
            trace.update(
                output={
                    "answer": answer,
                    "num_contexts": len(contexts),
                    "metrics": item_metrics
                }
            )
            
            # Log scores to Langfuse
            for metric_name, metric_value in item_metrics.items():
                if isinstance(metric_value, (int, float)):
                    self.langfuse.score(
                        trace_id=trace.id,
                        name=metric_name,
                        value=float(metric_value)
                    )
            
            # Link to dataset run
            item.link(
                trace_or_observation=trace,
                run_name=experiment_name
            )
            
            self.langfuse.flush()
        
        # Calculate aggregate metrics
        aggregate_metrics = {
            "experiment_name": experiment_name,
            "dataset_name": dataset_name,
            "num_items": len(items),
            "config": {
                "top_k": top_k,
                "k_values": k_values,
                "llm_model": config.LLM_MODEL,
                "embedding_model": config.EMBEDDING_MODEL
            }
        }
        
        # Retrieval metrics
        if any(all_metrics[k]['recall'] for k in k_values):
            aggregate_metrics["retrieval_metrics"] = {}
            for k in k_values:
                if all_metrics[k]['recall']:
                    aggregate_metrics["retrieval_metrics"][f"recall@{k}"] = round(
                        np.mean(all_metrics[k]['recall']), 4
                    )
                    aggregate_metrics["retrieval_metrics"][f"precision@{k}"] = round(
                        np.mean(all_metrics[k]['precision']), 4
                    )
            
            if mrr_scores:
                aggregate_metrics["retrieval_metrics"]["mrr"] = round(np.mean(mrr_scores), 4)
        
        # Answer metrics
        if answer_similarities:
            aggregate_metrics["answer_metrics"] = {
                "mean_similarity": round(np.mean(answer_similarities), 4),
                "min_similarity": round(min(answer_similarities), 4),
                "max_similarity": round(max(answer_similarities), 4)
            }
        
        # Print results
        print(f"\n{'='*60}")
        print("EXPERIMENT RESULTS")
        print(f"{'='*60}")
        print(json.dumps(aggregate_metrics, indent=2))
        
        # Save results
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"{experiment_name}_results.json"
        with open(output_file, 'w') as f:
            json.dump(aggregate_metrics, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        print(f"View in Langfuse: {config.LANGFUSE_HOST}/project/datasets")
        
        return aggregate_metrics
    
    def close(self):
        """Flush and close connections."""
        self.langfuse.flush()
        self.langfuse.shutdown()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RAG evaluation experiment')
    parser.add_argument('--dataset', type=str, default=config.DATASET_NAME,
                       help='Langfuse dataset name')
    parser.add_argument('--experiment', type=str,
                       help='Experiment name (default: auto-generated)')
    parser.add_argument('--top-k', type=int, default=config.TOP_K,
                       help='Number of documents to retrieve')
    parser.add_argument('--max-items', type=int,
                       help='Max items to evaluate (for testing)')
    args = parser.parse_args()
    
    runner = RAGExperimentRunner()
    
    try:
        runner.run_experiment(
            dataset_name=args.dataset,
            experiment_name=args.experiment,
            top_k=args.top_k,
            max_items=args.max_items
        )
    finally:
        runner.close()


if __name__ == '__main__':
    main()
