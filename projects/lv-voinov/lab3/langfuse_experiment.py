import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Literal
from dotenv import load_dotenv
import argparse
import ollama
from langfuse import Langfuse, propagate_attributes

import rag_engine

load_dotenv()

class LangfuseExperiment:
    def __init__(self, store_dir: str):
        self.langfuse = Langfuse()
        self.store_dir = store_dir
        
        print("Инициализация RAG движка для эксперимента")
        self.rag = rag_engine.RAGEngine(
            store_dir=store_dir,
            embedding_model_name="BAAI/bge-m3",
            llm_model_name="qwen3:4b-instruct"
        )
        self.dataset_name = "GPU Glossary QA Evaluation"

    def load_and_upload_dataset(self, eval_set_path: str):
        print(f"Загрузка датасета из {eval_set_path}")
        
        with open(eval_set_path, 'r', encoding='utf-8') as f:
            local_data = json.load(f)
        
        langfuse_items = []
        for item in local_data:
            langfuse_items.append({
                "input": item['query'],
                "expected_output": item.get('relevant_files', [])
            })
        
        dataset = self.langfuse.get_dataset(self.dataset_name)
        if dataset is None:
            print(f"Создание нового датасета '{self.dataset_name}'...")
            self.langfuse.create_dataset(name=self.dataset_name, items=langfuse_items)
        else:
            print(f"Датасет '{self.dataset_name}' уже существует.")
        
        print("Датасет готов к использованию")

    def compute_metrics(self, retrieved_docs: List[Dict], expected_files: List[str], k: int = 5):
        expected_set = set([f.replace("\\", "/") for f in expected_files])
        
        if not expected_set:
            return {"precision": 0, "recall": 0, "mrr": 0}

        retrieved_paths = [d['meta'].get('source_path', '').replace("\\", "/") for d in retrieved_docs[:k]]

        mrr = 0.0
        first_hit_rank = -1
        for i, path in enumerate(retrieved_paths):
            if path in expected_set:
                first_hit_rank = i + 1
                break
        
        if first_hit_rank != -1:
            mrr = 1.0 / first_hit_rank

        retrieved_unique_set = set(retrieved_paths)
        
        found_set = retrieved_unique_set & expected_set
        unique_hits = len(found_set)

        recall = unique_hits / len(expected_set)
        
        precision = unique_hits / k

        return {
            "precision": precision,
            "recall": recall,
            "mrr": mrr
        }

    def custom_rag_evaluator(self, item):
        query = item.input
        expected_data = item.expected_output
        expected_files = expected_data.get("relevant_files", [])
        
        with self.langfuse.start_as_current_observation(
            as_type="span",
            name="RAG-Eval-Run",
            input=query
        ) as root_trace:
            
            with self.langfuse.start_as_current_observation(
                as_type="span",
                name="Document-Retrieval",
                metadata={"k": 5}
            ) as retrieval_span:
                
                retrieved_docs = self.rag._retrieve(query, k=5)
                
                retrieval_span.update(
                    output={"count": len(retrieved_docs)},
                    metadata={
                        "k": 5,
                        "retrieved_ids": [doc["id"] for doc in retrieved_docs],
                        "retrieved_scores": [doc.get("score") for doc in retrieved_docs],
                        "chunk_contents": [doc.get("text")[:100] for doc in retrieved_docs]
                    }
                )
            
            if not retrieved_docs:
                root_trace.update(
                    output="No documents found",
                    metadata={"status": "no_docs"}
                )
                return {"precision": 0, "recall": 0, "mrr": 0}
            
            prompt = self.rag._build_prompt(query, retrieved_docs)
            
            with self.langfuse.start_as_current_observation(
                as_type="span",
                name="RAG-Query",
                input=query,
                metadata={"k": 5, "model": self.rag.llm_model}
            ) as trace:
                
                with propagate_attributes(user_id="eval-user", session_id="eval-session"):
                    
                    with self.langfuse.start_as_current_observation(
                        as_type="generation",
                        name="LLM-Answer",
                        model=self.rag.llm_model,
                        input=prompt,
                        metadata={"retrieved_docs_count": len(retrieved_docs)}
                    ) as generation:
                        
                        try:
                            response = ollama.chat(
                                model=self.rag.llm_model, 
                                messages=[
                                    {
                                        'role': 'system', 
                                        'content': 'You are a strict factual assistant. You answer only based on the provided context and cite your sources.'
                                    }, 
                                    {
                                        'role': 'user', 
                                        'content': prompt
                                    }
                                ]
                            )
                            llm_answer = response['message']['content']
                            
                        except Exception as e:
                            llm_answer = f"Error generating answer: {e}"
                        
                        root_trace.update(output=llm_answer)
                        trace.update(output=llm_answer)
                        generation.update(output=llm_answer)

            metrics = self.compute_metrics(retrieved_docs, expected_files, k=5)
            
            for metric_name, metric_value in metrics.items():
                root_trace.score(
                    name=metric_name,
                    value=metric_value
                )

            return metrics

    def run_experiment(self, local_dataset_path: str):
        self.load_and_upload_dataset(local_dataset_path)
        
        dataset = self.langfuse.get_dataset(self.dataset_name)
        if not dataset:
            print("Ошибка получения датасета")
            return
        
        dataset.run_experiment(
            name="Experiment_v1",
            task=self.custom_rag_evaluator
        )
        
        print("Эксперимент завершен. Логи отправлены в Langfuse")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск эксперимента RAG с Langfuse")
    parser.add_argument("--store_dir", default="./store_small", help="Папка с Faiss индексом")
    parser.add_argument("--eval_set", default="evaluation_set.json", help="Локальный файл с вопросами")
    
    args = parser.parse_args()
    
    if not os.getenv("LANGFUSE_SECRET_KEY"):
        print("ОШИБКА: Переменные окружения LANGFUSE_PUBLIC_KEY и LANGFUSE_SECRET_KEY не найдены.")
        print("Создайте файл .env или установите их в системе.")
        exit(1)

    experiment = LangfuseExperiment(store_dir=args.store_dir)
    experiment.run_experiment(local_dataset_path=args.eval_set)