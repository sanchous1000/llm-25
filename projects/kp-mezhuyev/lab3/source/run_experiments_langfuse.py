"""
Запуск экспериментов RAG через Langfuse для сравнения двух конфигураций.

Использует датасеты, созданные из lab2, и функции из lab2/source/evaluate.py
для вычисления метрик retrieval с полным логированием в Langfuse.
"""

import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langfuse import Langfuse
from tqdm import tqdm

# Добавляем lab2 в путь для импорта модулей
lab3_dir = Path(__file__).parent.parent
lab2_dir = lab3_dir.parent / "lab2"
lab2_source = lab2_dir / "source"

if not lab2_source.exists():
    raise FileNotFoundError(
        f"Не найдена папка lab2/source: {lab2_source}\n"
        f"Убедитесь, что lab2 находится рядом с lab3."
    )

sys.path.insert(0, str(lab2_source))

# Импортируем функции из lab2
# type: ignore комментарии для подавления предупреждений IDE
try:
    from config_utils import load_config  # type: ignore
    from embeddings import DenseEmbedder, format_text_for_e5  # type: ignore
    from es_utils import get_es_client  # type: ignore
    from rag_pipeline import rag_query, build_prompt, call_llm  # type: ignore
    from evaluate import (  # type: ignore
        search_chunks,
        calculate_recall_at_k,
        calculate_precision_at_k,
        calculate_mrr,
    )
except ImportError as e:
    print(f"Ошибка импорта модулей из lab2: {e}")
    print(f"Путь к lab2/source: {lab2_source}")
    print("\nУбедитесь, что:")
    print("1. Папка lab2 существует")
    print("2. В lab2/source есть все необходимые модули")
    print("3. Структура проекта:")
    print("   Third_semester/LLM/")
    print("   ├── lab2/")
    print("   │   └── source/")
    print("   │       ├── config_utils.py")
    print("   │       ├── embeddings.py")
    print("   │       ├── es_utils.py")
    print("   │       └── evaluate.py")
    print("   └── lab3/")
    print("       └── source/")
    print("           └── run_experiments_langfuse.py")
    raise


def run_experiment(
    dataset_name: str,
    index_name: str,
    config_path: Path,
    k: int = 5,
) -> dict[str, Any]:
    """Запустить эксперимент для одной конфигурации.
    
    Args:
        dataset_name: Имя датасета в Langfuse
        index_name: Имя индекса в Elasticsearch
        config_path: Путь к config.yaml
        k: Количество топ результатов
    
    Returns:
        dict: Результаты эксперимента с метриками
    """
    # Загрузка .env
    env_path = lab3_dir / "source" / ".env"
    
    if not env_path.exists():
        raise FileNotFoundError(f"Файл .env не найден: {env_path}")
    
    load_dotenv(env_path)
    
    # Инициализация Langfuse
    langfuse = Langfuse()
    
    # Загрузка конфигурации RAG
    config = load_config(config_path)
    
    # Переопределяем индекс
    es_config = config.get("elasticsearch", {})
    es_config["index_name"] = index_name
    config["elasticsearch"] = es_config
    
    # Подключаемся к Elasticsearch
    es_client, es_url = get_es_client(es_config)
    print(f"Подключено к Elasticsearch: {es_url}")
    
    if not es_client.ping():
        raise ConnectionError("Cannot connect to Elasticsearch")
    
    # Создаем embedder для запросов
    embeddings_config = config.get("embeddings", {})
    dense_config = embeddings_config.get("dense", {})
    model_name = dense_config.get("model", "intfloat/multilingual-e5-base")
    
    embedder = DenseEmbedder(
        model_name=model_name,
        device=dense_config.get("device", "cpu"),
    )
    
    is_e5_model = "e5" in model_name.lower()
    
    # Получаем датасет из Langfuse
    dataset = langfuse.get_dataset(dataset_name)
    
    print(f"\n{'='*60}")
    print(f"ЭКСПЕРИМЕНТ: {dataset_name}")
    print(f"Индекс: {index_name}")
    print(f"Модель: {model_name}")
    print("="*60)
    
    # Получаем все элементы датасета
    dataset_items = list(dataset.items)
    print(f"Элементов датасета: {len(dataset_items)}")
    
    # Результаты
    all_metrics = {
        f"recall@{k}": [],
        f"precision@{k}": [],
        "mrr": [],
    }
    
    # Запускаем оценку по каждому элементу
    print(f"\nОценка retrieval качества (k={k})...")
    
    for i, item in enumerate(tqdm(dataset_items, desc="Вопросы"), 1):
        question = item.input["question"]
        expected_chunk_ids = item.expected_output["expected_chunk_ids"]
        
        # Создаем trace в Langfuse
        trace = langfuse.trace(
            name=f"rag_evaluation_{dataset_name}",
            input={"question": question},
            metadata={
                "dataset_name": dataset_name,
                "index_name": index_name,
                "model_name": model_name,
                "question_index": i,
                "num_expected_chunks": len(expected_chunk_ids),
            },
        )
        
        try:
            # Создаем span для retrieval
            retrieval_span = trace.span(
                name="retrieval",
                input={"question": question, "top_k": k},
            )
            
            # Векторизуем вопрос
            query_text = format_text_for_e5(question, prefix="query: ") if is_e5_model else question
            query_embedding = embedder.embed([query_text])[0].tolist()
            
            # Поиск через Elasticsearch (используем функцию из evaluate.py)
            retrieved_results = search_chunks(
                es_client=es_client,
                index_name=index_name,
                query_embedding=query_embedding,
                top_k=k,
            )
            
            # Извлекаем chunk_ids
            retrieved_chunk_ids = [result["chunk_id"] for result in retrieved_results]
            
            # Завершаем span retrieval
            retrieval_span.end(
                output={
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "num_retrieved": len(retrieved_chunk_ids),
                    "sources": [
                        {
                            "chunk_id": r["chunk_id"],
                            "source_path": r["source_path"],
                            "title": r.get("title", ""),
                            "score": r["score"],
                        }
                        for r in retrieved_results
                    ],
                }
            )
            
            # Создаем span для генерации ответа LLM
            llm_span = trace.span(
                name="llm_generation",
                input={"question": question, "num_context_chunks": len(retrieved_results)},
            )
            
            # Собираем промпт и вызываем LLM (используем функции из rag_pipeline.py)
            prompt = build_prompt(question, retrieved_results)
            llm_answer = call_llm(prompt, config)
            
            # Завершаем span LLM
            llm_span.end(
                output={
                    "answer": llm_answer,
                    "answer_length": len(llm_answer),
                }
            )
            
            # Создаем span для вычисления метрик
            metrics_span = trace.span(
                name="metrics_calculation",
                input={
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "expected_chunk_ids": expected_chunk_ids,
                    "k": k,
                },
            )
            
            # Вычисляем метрики (используем функции из evaluate.py)
            recall = calculate_recall_at_k(
                retrieved_chunks=retrieved_chunk_ids,
                relevant_chunks=expected_chunk_ids,
                k=k,
            )
            
            precision = calculate_precision_at_k(
                retrieved_chunks=retrieved_chunk_ids,
                relevant_chunks=expected_chunk_ids,
                k=k,
            )
            
            mrr = calculate_mrr(
                retrieved_chunks=retrieved_chunk_ids,
                relevant_chunks=expected_chunk_ids,
            )
            
            metrics = {
                f"recall@{k}": recall,
                f"precision@{k}": precision,
                "mrr": mrr,
            }
            
            # Завершаем span метрик
            metrics_span.end(output=metrics)
            
            # Сохраняем метрики
            for metric_name, value in metrics.items():
                all_metrics[metric_name].append(value)
            
            # Логируем в trace output
            trace.update(
                output={
                    "answer": llm_answer,
                    "answer_length": len(llm_answer),
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "num_retrieved": len(retrieved_chunk_ids),
                    "metrics": metrics,
                }
            )
            
            # Логируем метрики как scores в Langfuse
            for metric_name, value in metrics.items():
                langfuse.score(
                    trace_id=trace.id,
                    name=metric_name,
                    value=value,
                    data_type="NUMERIC",
                )
            
            # Связываем trace с dataset item
            item.link(trace, f"evaluation_run_{dataset_name}")
            
        except Exception as e:
            print(f"Ошибка на вопросе {i}: {e}")
            trace.update(
                level="ERROR",
                status_message=str(e),
            )
            continue
    
    # Вычисляем средние метрики
    results = {}
    for metric_name, values in all_metrics.items():
        if values:
            results[metric_name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "std": (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5,
                "count": len(values),
            }
        else:
            results[metric_name] = {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
                "count": 0,
            }
    
    return results


def main():
    """Главная функция."""
    print("="*60)
    print("ЗАПУСК ЭКСПЕРИМЕНТОВ ЧЕРЕЗ LANGFUSE")
    print("="*60)
    
    # Конфигурации экспериментов
    experiments = [
        {
            "dataset_name": "fastapi_rag_baseline_recursive_1024",
            "index_name": "fastapi_docs_baseline_recursive_1024",
            "config_path": lab2_dir / "source" / "config.yaml",
            "description": "Baseline (Recursive, 1024 tokens, 371 chunks)",
        },
        {
            "dataset_name": "fastapi_rag_markdown_512",
            "index_name": "fastapi_docs_markdown_512",
            "config_path": lab2_dir / "source" / "config.yaml",
            "description": "Markdown splitter (512 tokens, 991 chunks)",
        },
    ]
    
    # Запускаем эксперименты
    all_results = {}
    
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Конфигурация: {exp['description']}")
        print("="*60)
        
        try:
            results = run_experiment(
                dataset_name=exp["dataset_name"],
                index_name=exp["index_name"],
                config_path=exp["config_path"],
                k=5,
            )
            
            all_results[exp["dataset_name"]] = results
            
            print(f"\n{'='*60}")
            print("РЕЗУЛЬТАТЫ:")
            print("="*60)
            for metric_name, stats in results.items():
                print(f"{metric_name:15s}: mean={stats['mean']:.3f}, "
                      f"std={stats['std']:.3f}, "
                      f"min={stats['min']:.3f}, "
                      f"max={stats['max']:.3f}")
            
        except Exception as e:
            print(f"\nОшибка: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Сравнение результатов
    print("\n" + "="*60)
    print("СРАВНЕНИЕ КОНФИГУРАЦИЙ")
    print("="*60)
    
    if len(all_results) >= 2:
        baseline_name = experiments[0]["dataset_name"]
        markdown_name = experiments[1]["dataset_name"]
        
        baseline = all_results.get(baseline_name, {})
        markdown = all_results.get(markdown_name, {})
        
        print(f"\n{'Метрика':<20} {'Baseline':>12} {'Markdown':>12} {'Разница':>12}")
        print("-" * 60)
        
        for metric_name in ["recall@5", "precision@5", "mrr"]:
            if metric_name in baseline and metric_name in markdown:
                b_val = baseline[metric_name]["mean"]
                m_val = markdown[metric_name]["mean"]
                diff = m_val - b_val
                diff_pct = (diff / b_val * 100) if b_val > 0 else 0
                
                print(f"{metric_name:<20} {b_val:>12.3f} {m_val:>12.3f} "
                      f"{diff:>+12.3f} ({diff_pct:+.1f}%)")
    
    print("\n" + "="*60)
    print("ГОТОВО!")
    print("="*60)
    print("\nПросмотреть результаты:")
    print("1. Откройте Langfuse UI: http://localhost:3000")
    print("2. Перейдите в раздел 'Datasets'")
    print("3. Выберите датасет:")
    for exp in experiments:
        print(f"   - {exp['dataset_name']}")
    print("4. Посмотрите traces, spans и метрики")
    print("5. Сравните результаты между конфигурациями")
    
    print("\nЭкспорт результатов:")
    output_file = lab3_dir / "data" / "experiments_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"Результаты сохранены: {output_file}")


if __name__ == "__main__":
    main()
