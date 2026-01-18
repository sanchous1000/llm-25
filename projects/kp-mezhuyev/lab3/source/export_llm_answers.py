"""
Экспорт ответов LLM из Langfuse.
Извлекает traces с ответами и метриками, сохраняет в JSON.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Langfuse  # type: ignore


def extract_answers_from_dataset(
    langfuse: Langfuse,
    dataset_name: str,
) -> list[dict]:
    """Извлечь ответы LLM из датасета."""
    print(f"\nИзвлечение данных из датасета: {dataset_name}")
    
    dataset = langfuse.get_dataset(dataset_name)
    dataset_items = list(dataset.items)
    
    print(f"Найдено элементов: {len(dataset_items)}")
    
    results = []
    
    for i, item in enumerate(dataset_items, 1):
        question = item.input["question"]
        expected_chunk_ids = item.expected_output["expected_chunk_ids"]
        
        print(f"  [{i}/{len(dataset_items)}] {question[:60]}...")
        
        item_data = {
            "question": question,
            "expected_chunk_ids": expected_chunk_ids,
            "num_expected_chunks": len(expected_chunk_ids),
            "answer": None,
            "retrieved_chunk_ids": [],
            "metrics": {},
            "sources": [],
        }
        
        results.append(item_data)
    
    print("\nПолучение traces из Langfuse...")
    
    try:
        traces_response = langfuse.client.trace.list(
            name=f"rag_evaluation_{dataset_name}",
            limit=100,
        )
        
        traces = traces_response.data if hasattr(traces_response, 'data') else []
        print(f"Найдено traces: {len(traces)}")
        
        for trace in traces:
            trace_input = trace.input if hasattr(trace, 'input') else {}
            trace_output = trace.output if hasattr(trace, 'output') else {}
            
            if not trace_input or not isinstance(trace_input, dict):
                continue
            
            trace_question = trace_input.get("question", "")
            
            for item_data in results:
                if item_data["question"] == trace_question:
                    if isinstance(trace_output, dict):
                        item_data["retrieved_chunk_ids"] = trace_output.get("retrieved_chunk_ids", [])
                        item_data["metrics"] = trace_output.get("metrics", {})
                        if trace_output.get("answer"):
                            item_data["answer"] = trace_output["answer"]
                    
                    # Получаем из observations (spans)
                    if hasattr(trace, 'id'):
                        try:
                            full_trace = langfuse.client.trace.get(trace.id)
                            
                            if hasattr(full_trace, 'observations'):
                                for obs in full_trace.observations:
                                    obs_name = getattr(obs, 'name', '')
                                    obs_output = getattr(obs, 'output', None)
                                    
                                    if obs_name == "llm_generation" and obs_output:
                                        if isinstance(obs_output, dict):
                                            answer = obs_output.get("answer", "")
                                            if answer:
                                                item_data["answer"] = answer
                                    
                                    elif obs_name == "retrieval" and obs_output:
                                        if isinstance(obs_output, dict):
                                            sources = obs_output.get("sources", [])
                                            if sources:
                                                item_data["sources"] = sources
                        except Exception:
                            pass
                    
                    break
        
    except Exception as e:
        print(f"Ошибка при получении traces: {e}")
    
    return results


def main():
    """Главная функция."""
    lab3_dir = Path(__file__).parent.parent
    env_path = lab3_dir / "source" / ".env"
    
    if not env_path.exists():
        print(f"Файл .env не найден: {env_path}")
        sys.exit(1)
    
    load_dotenv(env_path)
    
    try:
        langfuse = Langfuse()
        print("Подключено к Langfuse")
    except Exception as e:
        print(f"Ошибка подключения: {e}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("ЭКСПОРТ ОТВЕТОВ LLM ИЗ LANGFUSE")
    print("="*60)
    
    datasets = [
        {
            "name": "fastapi_rag_baseline_recursive_1024",
            "output_file": "llm_answers_baseline_recursive_1024.json",
            "description": "Baseline (Recursive, 1024 tokens)",
        },
        {
            "name": "fastapi_rag_markdown_512",
            "output_file": "llm_answers_markdown_512.json",
            "description": "Markdown splitter (512 tokens)",
        },
    ]
    
    all_answers = {}
    
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Датасет: {ds['description']}")
        print("="*60)
        
        try:
            answers = extract_answers_from_dataset(langfuse, ds["name"])
            
            output_path = lab3_dir / "data" / ds["output_file"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(answers, f, indent=2, ensure_ascii=False)
            
            print(f"\nСохранено {len(answers)} ответов: {output_path}")
            all_answers[ds["name"]] = answers
            
        except Exception as e:
            print(f"\nОшибка: {e}")
            continue
    
    print("\n" + "="*60)
    print("СОЗДАНИЕ СВОДНОГО ФАЙЛА")
    print("="*60)
    
    comparison = {
        "datasets": [ds["name"] for ds in datasets],
        "questions": [],
    }
    
    if all_answers:
        first_dataset = list(all_answers.values())[0]
        
        for item in first_dataset:
            question_comparison = {
                "question": item["question"],
                "expected_chunk_ids": item["expected_chunk_ids"],
                "configurations": {},
            }
            
            for ds_name, answers_list in all_answers.items():
                for answer_item in answers_list:
                    if answer_item["question"] == item["question"]:
                        question_comparison["configurations"][ds_name] = {
                            "answer": answer_item.get("answer", ""),
                            "retrieved_chunk_ids": answer_item.get("retrieved_chunk_ids", []),
                            "metrics": answer_item.get("metrics", {}),
                        }
                        break
            
            comparison["questions"].append(question_comparison)
    
    comparison_path = lab3_dir / "data" / "llm_answers_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\nСводное сравнение: {comparison_path}")
    
    print("\n" + "="*60)
    print("Созданные файлы:")
    for ds in datasets:
        print(f"  - data/{ds['output_file']}")
    print(f"  - data/llm_answers_comparison.json")
    print("\nПросмотр в Langfuse UI: http://localhost:3000/traces")


if __name__ == "__main__":
    main()
