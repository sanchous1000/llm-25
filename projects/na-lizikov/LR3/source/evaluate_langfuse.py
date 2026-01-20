"""
Скрипт для оценки RAG-системы через Langfuse Experiment Run (V2 API)
"""
import json
import os
import sys
from dotenv import load_dotenv
from langfuse import Langfuse
from typing import List, Dict, Any

# Загружаем переменные окружения
load_dotenv()

# Добавляем путь к LR2 для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "LR2", "scripts"))
from retrieve import hybrid_search

# Настройки
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
)

DATASET_NAME = "python_docs_qa"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "mistral")


def normalize_path(path):
    """Нормализует путь для сравнения"""
    if not path:
        return ""
    return path.replace("/", "\\")


def calculate_metrics(retrieved_paths, expected_paths, k_values=[5, 10]):
    """Вычисляет метрики Recall@k, Precision@k, MRR"""
    metrics = {}
    
    expected_set = set(normalize_path(p) for p in expected_paths)
    retrieved_normalized = [normalize_path(p) for p in retrieved_paths]
    
    for k in k_values:
        top_k_paths = retrieved_normalized[:k]
        found_relevant = [p for p in top_k_paths if p in expected_set]
        
        # Recall@k
        recall = len(found_relevant) / len(expected_set) if expected_set else 0.0
        
        # Precision@k
        precision = len(found_relevant) / k if k > 0 else 0.0
        
        # MRR
        mrr = 0.0
        for rank, path in enumerate(top_k_paths, start=1):
            if path in expected_set:
                mrr = 1.0 / rank
                break
        
        metrics[k] = {
            "recall": recall,
            "precision": precision,
            "mrr": mrr,
            "found_relevant": len(found_relevant),
            "total_relevant": len(expected_set)
        }
    
    return metrics


def call_ollama(prompt, trace_id=None):
    """Вызов Ollama для генерации ответа (V2 API)"""
    import requests
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        answer = result["response"]
        
        # Логируем генерацию через V2 API
        if trace_id:
            langfuse.generation(
                trace_id=trace_id,
                name="llm_generation",
                model=MODEL,
                input=prompt,
                output=answer,
                metadata={"model": MODEL}
            )
        
        return answer
    except Exception as e:
        error_msg = f"Ошибка: {e}"
        if trace_id:
            langfuse.generation(
                trace_id=trace_id,
                name="llm_generation_error",
                model=MODEL,
                input=prompt,
                output=error_msg,
                metadata={"error": True}
            )
        return error_msg


def run_evaluator(dataset_item, top_k=10, trace_id=None) -> Dict[str, Any]:
    """
    Кастомный evaluator для RAG-пайплайна (V2 API)
    Принимает элемент датасета и возвращает словарь с метриками и output
    """
    question = dataset_item.input["question"]
    expected_output = dataset_item.expected_output
    expected_chunks = expected_output.get("relevant_chunks", [])
    
    # 1. Retrieval: извлечение релевантных документов
    hits = hybrid_search(question, top_k)
    
    retrieved_paths = [
        hit["_source"].get("relative_path", "") for hit in hits
    ]
    
    # Логируем retrieval через span (V2 API)
    if trace_id:
        retrieval_span = langfuse.span(
            trace_id=trace_id,
            name="retrieval",
            input=question,
            metadata={"top_k": top_k, "num_hits": len(hits)},
            output={"retrieved_paths": retrieved_paths}
        )
    
    # 2. Вычисление retrieval-метрик
    k_values = [5, 10]
    metrics = calculate_metrics(retrieved_paths, expected_chunks, k_values)
    
    # 3. Генерация ответа LLM
    if hits:
        contexts = []
        for h in hits[:5]:
            src = h["_source"]
            text = src.get("text", "")[:500]
            contexts.append(f"{text}...")
        
        prompt = f"""Ответь на вопрос используя контекст:
{chr(10).join(contexts)}

Вопрос: {question}
Ответ:"""
        
        answer = call_ollama(prompt, trace_id=trace_id)
    else:
        answer = "Не найдено релевантной информации"
    
    # 4. Формирование результата с метриками
    result = {
        "output": answer,
        "metrics": metrics,
        "retrieved_paths": retrieved_paths,
        "expected_paths": expected_chunks
    }
    
    return result


def run_experiment(experiment_name="rag_evaluation_v1", top_k=10):
    """
    Запускает эксперимент оценки на датасете через Langfuse
    """
    print(f"Запуск эксперимента: {experiment_name}")
    print(f"Датасет: {DATASET_NAME}")
    print(f"Top-K для retrieval: {top_k}\n")
    
    # В V2 API получаем элементы датасета через get_dataset_item
    QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "LR2", "data", "questions.json")
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions_data = json.load(f)
    
    # Преобразуем в формат элементов датасета
    class DatasetItem:
        def __init__(self, input_data, expected_output_data, item_id, dataset_item_id=None):
            self.input = input_data
            self.expected_output = expected_output_data
            self.id = item_id
            self.dataset_item_id = dataset_item_id  # Реальный ID из Langfuse
    
    items = []
    for idx, item_data in enumerate(questions_data, 1):
        question = item_data["question"]
        expected_output = {
            "relevant_chunks": item_data.get("relevant_chunks", []),
            "description": f"Ожидаемые релевантные чанки для вопроса: {question[:100]}..."
        }
        
        # Пытаемся получить реальный ID элемента датасета из Langfuse
        dataset_item_id = None
        
        items.append(DatasetItem(
            input_data={"question": question},
            expected_output_data=expected_output,
            item_id=f"item_{idx}",
            dataset_item_id=dataset_item_id
        ))
    
    print(f"Найдено {len(items)} элементов в датасете\n")
    
    # Запускаем оценку для каждого элемента
    for idx, item in enumerate(items, 1):
        print(f"Обработка {idx}/{len(items)}: {item.input['question'][:60]}...")
        
        try:
            # Создаем trace через правильный API V2
            # Связываем trace с dataset через metadata
            trace = langfuse.trace(
                name="rag_evaluation",
                user_id=f"experiment_{experiment_name}",
                input=item.input,
                metadata={
                    "experiment_name": experiment_name,
                    "top_k": top_k,
                    "model": MODEL,
                    "question": item.input["question"],
                    "dataset_name": DATASET_NAME,
                    "dataset_item_id": item.id,
                    "evaluation": True
                }
            )
            
            # Получаем trace_id
            trace_id = trace.id if hasattr(trace, 'id') else None
            if not trace_id:
                try:
                    trace_id = trace.get_trace_id() if hasattr(trace, 'get_trace_id') else None
                except:
                    pass
            
            # Запускаем evaluator с trace_id
            result = run_evaluator(item, top_k, trace_id=trace_id)
            
            # Вычисляем метрики из результата
            metrics = result["metrics"]
            k_values = [5, 10]
            
            # Создаем scores для каждой метрики
            scores = []
            for k in k_values:
                m = metrics[k]
                scores.append({
                    "name": f"recall@{k}",
                    "value": m["recall"],
                    "comment": f"Найдено {m['found_relevant']} из {m['total_relevant']} релевантных"
                })
                scores.append({
                    "name": f"precision@{k}",
                    "value": m["precision"],
                    "comment": f"Точность среди top-{k}"
                })
                scores.append({
                    "name": f"mrr@{k}",
                    "value": m["mrr"],
                    "comment": f"MRR для k={k}"
                })
            
            # Обновляем trace с output
            trace.update(
                output=result["output"],
                metadata={
                    "metrics": metrics,
                    "num_retrieved": len(result["retrieved_paths"]),
                    "num_expected": len(result["expected_paths"])
                }
            )
            
            # Добавляем метрики как scores (V2 API)
            scores_added = 0
            for score_data in scores:
                try:
                    # В V2 используем langfuse.score() напрямую с trace_id
                    langfuse.score(
                        trace_id=trace_id,
                        name=score_data["name"],
                        value=score_data["value"],
                        comment=score_data.get("comment")
                    )
                    scores_added += 1
                except Exception as score_err:
                    # Если не получается через score, пробуем через trace
                    try:
                        if hasattr(trace, 'score'):
                            trace.score(
                                name=score_data["name"],
                                value=score_data["value"],
                                comment=score_data.get("comment")
                            )
                            scores_added += 1
                        else:
                            # Сохраняем в metadata как fallback
                            pass
                    except Exception:
                        pass
            
            if scores_added > 0:
                print(f"  ✓ Добавлено {scores_added} метрик как scores")
            else:
                print(f"  ⚠ Метрики сохранены в trace metadata (scores недоступны)")
            
            # Обновляем metadata trace с полными метриками для анализа
            try:
                trace.update(metadata={
                    "experiment_name": experiment_name,
                    "top_k": top_k,
                    "model": MODEL,
                    "question": item.input["question"],
                    "dataset_name": DATASET_NAME,
                    "dataset_item_id": item.id,
                    "evaluation": True,
                    "metrics": metrics,
                    "num_retrieved": len(result["retrieved_paths"]),
                    "num_expected": len(result["expected_paths"]),
                    "scores": {s["name"]: s["value"] for s in scores}
                })
            except Exception:
                pass
            
            # Отправляем данные в Langfuse
            try:
                langfuse.flush()
            except Exception:
                pass
            
            print(f"  ✓ Метрики вычислены и сохранены")
            
        except Exception as e:
            print(f"  ✗ Ошибка: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nЭксперимент '{experiment_name}' завершен!")
    print(f"Результаты доступны в интерфейсе Langfuse: {os.getenv('LANGFUSE_HOST', 'http://localhost:3000')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Оценка RAG-системы через Langfuse")
    parser.add_argument("--experiment-name", default="rag_evaluation_v1", help="Имя эксперимента")
    parser.add_argument("--top-k", type=int, default=10, help="Количество результатов для retrieval")
    args = parser.parse_args()
    
    run_experiment(args.experiment_name, args.top_k)
