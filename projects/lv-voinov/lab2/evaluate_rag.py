import json
import argparse
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd

class RAGEvaluator:
    def __init__(self, store_path: str, model_name: str, k_list=[5, 10]):
        self.store_path = Path(store_path)
        self.k_list = k_list
        self.index = None
        self.id_map = {}
        self.meta_map = {}
        self.model = SentenceTransformer(model_name)

        self.load_store()

    def load_store(self):
        print(f"Загрузка хранилища из {self.store_path}")
        
        index_path = self.store_path / "faiss.index"
        self.index = faiss.read_index(str(index_path))
        
        with open(self.store_path / "id_mapping.json", 'r', encoding='utf-8') as f:
            self.id_map = {int(k): v for k, v in json.load(f).items()}

        with open(self.store_path / "metadata.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                faiss_id = record['faiss_id']
                self.meta_map[faiss_id] = record['payload']
        
        print(f"Индекс загружен. Векторов: {self.index.ntotal}")

    def search(self, query: str, k: int):
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(query_embedding, k)
        
        retrieved_ids = indices[0]
        
        retrieved_files = []
        for faiss_id in retrieved_ids:
            if faiss_id != -1 and faiss_id in self.meta_map:
                source_path = self.meta_map[faiss_id]['metadata'].get('source_path', 'unknown')
                source_path = source_path.replace("\\", "/")
                
                retrieved_files.append(source_path)
            else:
                retrieved_files.append(None)
                
        return retrieved_files

    def calculate_metrics(self, retrieved_files, relevant_files):
        results = {}
        relevant_set = set(relevant_files)
        
        if not relevant_set:
            return {}

        for k in self.k_list:
            top_k_files = retrieved_files[:k]
            top_k_set = set(top_k_files)
            
            hits = len(top_k_set.intersection(relevant_set))
            
            results[f'precision@{k}'] = hits / k
            
            results[f'recall@{k}'] = hits / len(relevant_files)

        mrr = 0.0
        for i, file in enumerate(retrieved_files):
            if file in relevant_set:
                mrr = 1.0 / (i + 1)
                break
        results['mrr'] = mrr
        
        return results

    def run_evaluation(self, golden_set_path):
        with open(golden_set_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)

        all_metrics = []
        
        for item in tqdm(questions, desc="Оценка запросов"):
            query = item['query']
            relevant = [r.replace("\\", "/") for r in item.get('relevant_files', [])]
            
            retrieved_files = self.search(query, max(self.k_list))
            metrics = self.calculate_metrics(retrieved_files, relevant)
            
            all_metrics.append(metrics)

        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m.get(key, 0) for m in all_metrics]
            avg_metrics[key] = np.mean(values)
            
        return avg_metrics, all_metrics

def main():
    parser = argparse.ArgumentParser(description="Оценка RAG системы")
    parser.add_argument("--store_dir", required=True, help="Папка с индексом Faiss")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Модель эмбеддингов")
    parser.add_argument("--golden_set", default="evaluation_set.json", help="Файл с вопросами")
    parser.add_argument("--k", type=int, nargs='+', default=[5, 10], help="Список значений K")
    
    args = parser.parse_args()
    
    evaluator = RAGEvaluator(
        store_path=args.store_dir,
        model_name=args.model,
        k_list=args.k
    )
    
    avg_metrics, detailed = evaluator.run_evaluation(args.golden_set)
    
    print(f"Конфигурация: {args.store_dir}")
    for k in args.k:
        print(f"Precision@{k}: {avg_metrics.get(f'precision@{k}', 0):.4f}")
        print(f"Recall@{k}: {avg_metrics.get(f'recall@{k}', 0):.4f}")
    print(f"MRR: {avg_metrics.get('mrr', 0):.4f}")

if __name__ == "__main__":
    main()