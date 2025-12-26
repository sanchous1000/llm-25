import sys
import os
import time
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
from langfuse import Langfuse, Evaluation


sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

from config import Config
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from evaluation import evaluate_retrieval

load_dotenv()


def run_experiment_on_dataset(
    rag: RAGPipeline,
    dataset_name: str = "rag_python_qa",
    experiment_name: str = None,
    session_id: str = None,
) -> Dict[str, Any]:
    langfuse = rag.langfuse
    
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"rag_experiment_{rag.config.rag.llm_model}_{rag.config.embeddings.type}_{timestamp}"
    
    if session_id is None:
        session_id = f"experiment_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n{'=' * 60}")
    print(f"Запуск эксперимента: {experiment_name}")
    print(f"Датасет: {dataset_name}")
    print(f"Session ID: {session_id}")
    print(f"Модель: {rag.config.rag.llm_model}")
    print(f"Провайдер: {rag.config.rag.llm_provider}")
    print(f"Тип эмбеддингов: {rag.config.embeddings.type}")
    print(f"Top-K: {rag.config.rag.top_k}")
    print(f"{'=' * 60}\n")
    
    try:
        dataset = langfuse.get_dataset(name=dataset_name)
        print(f"Датасет '{dataset_name}' загружен успешно")
    except Exception as e:
        print(f"Ошибка при загрузке датасета '{dataset_name}': {e}")
        print("Убедитесь, что датасет создан. Запустите:")
        print(f"  python scripts/create_dataset.py --dataset-name {dataset_name}")
        return {"error": str(e), "success": False}
    
    # Получаем элементы датасета
    try:
        # В Langfuse элементы датасета получаются через итерацию по объекту датасета
        # Пробуем разные способы получения элементов в зависимости от версии SDK
        dataset_items = []
        
        # Сначала пробуем итерировать по датасету напрямую (наиболее распространенный способ)
        try:
            dataset_items = list(dataset)
        except (TypeError, AttributeError):
            # Если итерация не работает, пробуем через атрибут items
            if hasattr(dataset, 'items'):
                if callable(dataset.items):
                    dataset_items = list(dataset.items())
                else:
                    dataset_items = list(dataset.items)
            elif hasattr(dataset, 'get_items'):
                dataset_items = list(dataset.get_items())
            else:
                raise AttributeError("Не удалось получить элементы датасета. Объект датасета не поддерживает итерацию")
        
        print(f"Найдено элементов в датасете: {len(dataset_items)}")
    except Exception as e:
        print(f"Ошибка при получении элементов датасета: {e}")
        print(f"Убедитесь, что датасет содержит элементы и используется правильная версия Langfuse SDK")
        return {"error": str(e), "success": False}
    
    if not dataset_items:
        print("Датасет пуст. Добавьте элементы в датасет.")
        return {"error": "Dataset is empty", "success": False}
    
    # Определяем функцию задачи для run_experiment
    def task(item):
        """Функция задачи, которая обрабатывает элемент датасета и возвращает результат"""
        query = item.input.get("query", "")
        
        try:
            result = rag.answer(
                query=query,
                session_id=session_id,
                user_id=f"experiment_{session_id}",
                return_trace_id=True,
            )
            
            answer = result.get("answer", "")
            citations = result.get("citations", [])
            context_chunks = result.get("context_chunks", 0)
            trace_id = result.get("trace_id")
            
            # Формируем output для Dataset Run
            output = {
                "answer": answer,
                "citations": citations,
                "context_chunks": context_chunks,
                "trace_id": trace_id,
                "vector_db_metrics": result.get("vector_db_metrics", {}),
            }
            
            return output
        except Exception as e:
            error_msg = f"Ошибка при обработке вопроса: {e}"
            return {
                "answer": error_msg,
                "citations": [],
                "context_chunks": 0,
                "trace_id": None,
                "error": str(e),
            }
    
    # Определяем функции-оценщики для вычисления метрик
    def evaluate_retrieval_metrics(*, input, output, expected_output=None, **kwargs):
        """Оценщик для вычисления метрик retrieval"""
        evaluations = []
        
        if not expected_output:
            return evaluations
        
        relevant_chunk_ids = expected_output.get("relevant_chunk_ids", [])
        
        if not relevant_chunk_ids:
            return evaluations
        
        citations = output.get("citations", [])
        retrieved_chunk_ids = []
        
        for citation in citations:
            # Пытаемся извлечь chunk_id из citation
            # Если chunk_id не указан напрямую, пытаемся найти его в metadata
            if "chunk_id" in citation:
                retrieved_chunk_ids.append(citation["chunk_id"])
            elif "metadata" in citation and "chunk_id" in citation["metadata"]:
                retrieved_chunk_ids.append(citation["metadata"]["chunk_id"])
        
        if not retrieved_chunk_ids:
            return evaluations
        
        try:
            retrieved_results = [{"id": cid} for cid in retrieved_chunk_ids]
            ground_truth = set(relevant_chunk_ids)
            
            retrieval_metrics = evaluate_retrieval(
                retrieved_results,
                ground_truth,
                k_values=[5, 10],
            )
            
            for metric_name, metric_value in retrieval_metrics.items():
                evaluations.append(
                    Evaluation(
                        name=metric_name,
                        value=float(metric_value),
                        comment=f"Метрика retrieval: {metric_name}",
                    )
                )
        except Exception as e:
            print(f"  Предупреждение: не удалось вычислить метрики retrieval: {e}")
        
        return evaluations
    
    def evaluate_answer_quality(*, input, output, expected_output=None, **kwargs):
        """Оценщик для оценки качества ответа"""
        evaluations = []
        
        answer = output.get("answer", "")
        answer_length = len(answer) if isinstance(answer, str) else 0
        word_count = len(answer.split()) if isinstance(answer, str) else 0
        citations = output.get("citations", [])
        
        is_error = isinstance(answer, str) and (
            answer.startswith("Ошибка") or
            answer.startswith("Информация не найдена") or
            "error" in answer.lower()
        )
        
        evaluations.append(
            Evaluation(
                name="answer_length",
                value=float(answer_length),
                comment="Длина ответа в символах",
            )
        )
        
        evaluations.append(
            Evaluation(
                name="word_count",
                value=float(word_count),
                comment="Количество слов в ответе",
            )
        )
        
        evaluations.append(
            Evaluation(
                name="num_citations",
                value=float(len(citations)),
                comment="Количество источников",
            )
        )
        
        evaluations.append(
            Evaluation(
                name="success",
                value=1.0 if not is_error else 0.0,
                comment="Успешность выполнения",
            )
        )
        
        return evaluations
    
    print(f"Начинаем эксперимент через Langfuse run_experiment")
    print(f"Метаданные эксперимента:")
    print(f"  - LLM Provider: {rag.config.rag.llm_provider}")
    print(f"  - LLM Model: {rag.config.rag.llm_model}")
    print(f"  - Embedding Type: {rag.config.embeddings.type}")
    print(f"  - Top-K: {rag.config.rag.top_k}")
    print(f"  - Temperature: {rag.config.rag.temperature}")
    print(f"  - Max Tokens: {rag.config.rag.max_tokens}")
    
    # Запускаем эксперимент через Langfuse run_experiment
    try:
        experiment_metadata = {
            "llm_provider": rag.config.rag.llm_provider,
            "llm_model": rag.config.rag.llm_model,
            "embedding_type": rag.config.embeddings.type,
            "top_k": str(rag.config.rag.top_k),
            "temperature": str(rag.config.rag.temperature),
            "max_tokens": str(rag.config.rag.max_tokens),
            "session_id": session_id,
        }
        
        experiment_result = langfuse.run_experiment(
            name=experiment_name,
            data=dataset_items,
            task=task,
            evaluators=[evaluate_retrieval_metrics, evaluate_answer_quality],
            metadata=experiment_metadata,
        )
        
        print(f"\n{'=' * 60}")
        print("ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
        print(f"{'=' * 60}")
        print(f"Имя эксперимента: {experiment_name}")
        print(f"Всего элементов: {len(dataset_items)}")
        print(f"Успешных: {experiment_result.successful_count if hasattr(experiment_result, 'successful_count') else 'N/A'}")
        print(f"Неудачных: {experiment_result.failed_count if hasattr(experiment_result, 'failed_count') else 'N/A'}")
        
        langfuse_url = (
            os.getenv("LANGFUSE_BASE_URL")
            or os.getenv("LANGFUSE_HOST")
            or "http://localhost:3000"
        )
        print(f"\nРезультаты можно просмотреть в интерфейсе Langfuse:")
        print(f"{langfuse_url}/datasets/{dataset_name}")
        print(f"{'=' * 60}\n")
        
        # Формируем результат для обратной совместимости
        results = []
        successful = 0
        failed = 0
        
        # Извлекаем результаты из experiment_result
        if hasattr(experiment_result, 'items'):
            for run_item in experiment_result.items:
                item_result = {
                    "dataset_item_id": run_item.dataset_item_id if hasattr(run_item, 'dataset_item_id') else None,
                    "query": run_item.input.get("query", "") if hasattr(run_item, 'input') else "",
                    "answer": run_item.output.get("answer", "") if hasattr(run_item, 'output') else "",
                    "success": run_item.status == "success" if hasattr(run_item, 'status') else False,
                }
                results.append(item_result)
                if item_result["success"]:
                    successful += 1
                else:
                    failed += 1
        
        return {
            "summary": {
                "experiment_name": experiment_name,
                "dataset_name": dataset_name,
                "session_id": session_id,
                "total_items": len(dataset_items),
                "successful": successful,
                "failed": failed,
                "success_rate": successful / len(dataset_items) if dataset_items else 0,
                "timestamp": datetime.now().isoformat(),
            },
            "results": results,
            "success": True,
            "experiment_result": experiment_result,
        }
        
    except Exception as e:
        print(f"Ошибка при запуске эксперимента через run_experiment: {e}")
        print("Пробуем альтернативный способ...")
        # Fallback на старый способ, если run_experiment не работает
        return run_experiment_fallback(rag, dataset_items, dataset_name, experiment_name, session_id)


def run_experiment_fallback(
    rag: RAGPipeline,
    dataset_items: List,
    dataset_name: str,
    experiment_name: str,
    session_id: str,
) -> Dict[str, Any]:
    """Альтернативный способ запуска эксперимента (fallback)"""
    langfuse = rag.langfuse
    results = []
    successful = 0
    failed = 0
    
    for i, item in enumerate(dataset_items, 1):
        query = item.input.get("query", "")
        expected_output = item.expected_output or {}
        relevant_chunk_ids = expected_output.get("relevant_chunk_ids", [])
        
        print(f"\n[{i}/{len(dataset_items)}] Обработка вопроса: {query[:60]}...")
        
        start_time = time.time()
        
        try:
            result = rag.answer(
                query=query,
                session_id=session_id,
                user_id=f"experiment_{session_id}",
                return_trace_id=True,
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            answer = result.get("answer", "")
            citations = result.get("citations", [])
            context_chunks = result.get("context_chunks", 0)
            trace_id = result.get("trace_id")
            
            # Метаданные эксперимента будут добавлены в комментарии к scores
            # и доступны через session_id для фильтрации traces
            
            retrieved_chunk_ids = []
            if citations:
                for citation in citations:
                    if "chunk_id" in citation:
                        retrieved_chunk_ids.append(citation["chunk_id"])
            
            retrieval_metrics = {}
            if relevant_chunk_ids and retrieved_chunk_ids:
                try:
                    retrieved_results = [
                        {"id": cid} for cid in retrieved_chunk_ids
                    ]
                    ground_truth = set(relevant_chunk_ids)
                    
                    retrieval_metrics = evaluate_retrieval(
                        retrieved_results,
                        ground_truth,
                        k_values=[5, 10],
                    )
                except Exception as e:
                    print(f"  Предупреждение: не удалось вычислить метрики retrieval: {e}")
            
            answer_length = len(answer) if isinstance(answer, str) else 0
            word_count = len(answer.split()) if isinstance(answer, str) else 0
            
            is_error = isinstance(answer, str) and (
                answer.startswith("Ошибка") or
                answer.startswith("Информация не найдена") or
                "error" in answer.lower()
            )
            
            item_result = {
                "dataset_item_id": item.id if hasattr(item, "id") else None,
                "query": query,
                "answer": answer,
                "execution_time": execution_time,
                "answer_length": answer_length,
                "word_count": word_count,
                "num_citations": len(citations),
                "num_context_chunks": context_chunks,
                "is_error": is_error,
                "success": not is_error,
                "trace_id": trace_id,
                "retrieval_metrics": retrieval_metrics,
                "vector_db_metrics": result.get("vector_db_metrics", {}),
            }
            
            # В Langfuse Python SDK нет метода create_experiment_run_item
            # Результаты логируются через traces и scores (см. код ниже)
            
            if trace_id:
                try:
                    langfuse.create_score(
                        name="execution_time",
                        value=execution_time,
                        trace_id=trace_id,
                        comment=f"Эксперимент: {experiment_name}, Датасет: {dataset_name}, Вопрос: {query[:50]}",
                    )
                    
                    langfuse.create_score(
                        name="answer_length",
                        value=float(answer_length),
                        trace_id=trace_id,
                        comment=f"Длина ответа в символах",
                    )
                    
                    langfuse.create_score(
                        name="word_count",
                        value=float(word_count),
                        trace_id=trace_id,
                        comment=f"Количество слов в ответе",
                    )
                    
                    langfuse.create_score(
                        name="num_citations",
                        value=float(len(citations)),
                        trace_id=trace_id,
                        comment=f"Количество источников",
                    )
                    
                    langfuse.create_score(
                        name="success",
                        value=1.0 if not is_error else 0.0,
                        trace_id=trace_id,
                        comment=f"Успешность выполнения",
                    )
                    
                    for metric_name, metric_value in retrieval_metrics.items():
                        langfuse.create_score(
                            name=metric_name,
                            value=float(metric_value),
                            trace_id=trace_id,
                            comment=f"Метрика retrieval: {metric_name}",
                        )
                    
                    langfuse.flush()
                except Exception as e:
                    print(f"  Предупреждение: не удалось добавить метрики в Langfuse: {e}")
            
            results.append(item_result)
            
            if not is_error:
                successful += 1
            else:
                failed += 1
            
            print(f"  ✓ Время: {execution_time:.2f}с, Длина: {answer_length} символов, "
                  f"Источников: {len(citations)}, {'Успех' if not is_error else 'Ошибка'}")
            
            if retrieval_metrics:
                print(f"  Метрики retrieval: {retrieval_metrics}")
        
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            error_msg = f"Ошибка при обработке вопроса: {e}"
            print(f"  ✗ {error_msg}")
            
            item_result = {
                "dataset_item_id": item.id if hasattr(item, "id") else None,
                "query": query,
                "answer": error_msg,
                "execution_time": execution_time,
                "answer_length": 0,
                "word_count": 0,
                "num_citations": 0,
                "num_context_chunks": 0,
                "is_error": True,
                "success": False,
                "error": str(e),
                "trace_id": None,
            }
            
            results.append(item_result)
            failed += 1
            
            # В Langfuse Python SDK нет метода create_experiment_run_item
            # Ошибки логируются через traces (если trace_id доступен)
        
        time.sleep(0.5)
    
    if successful > 0:
        avg_time = sum(r["execution_time"] for r in results if r["success"]) / successful
        avg_length = sum(r["answer_length"] for r in results if r["success"]) / successful
        avg_words = sum(r["word_count"] for r in results if r["success"]) / successful
        avg_citations = sum(r["num_citations"] for r in results if r["success"]) / successful
    else:
        avg_time = 0
        avg_length = 0
        avg_words = 0
        avg_citations = 0
    
    avg_retrieval_metrics = {}
    if results:
        retrieval_metrics_list = [r.get("retrieval_metrics", {}) for r in results if r.get("retrieval_metrics")]
        if retrieval_metrics_list:
            for metric_name in retrieval_metrics_list[0].keys():
                values = [m.get(metric_name, 0) for m in retrieval_metrics_list if metric_name in m]
                if values:
                    avg_retrieval_metrics[metric_name] = sum(values) / len(values)
    
    summary = {
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "session_id": session_id,
        "total_items": len(dataset_items),
        "successful": successful,
        "failed": failed,
        "success_rate": successful / len(dataset_items) if dataset_items else 0,
        "avg_execution_time": avg_time,
        "avg_answer_length": avg_length,
        "avg_word_count": avg_words,
        "avg_citations": avg_citations,
        "avg_retrieval_metrics": avg_retrieval_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    
    print(f"\n{'=' * 60}")
    print("СВОДКА ЭКСПЕРИМЕНТА")
    print(f"{'=' * 60}")
    print(f"Всего элементов: {summary['total_items']}")
    print(f"Успешных: {summary['successful']}")
    print(f"Неудачных: {summary['failed']}")
    print(f"Процент успеха: {summary['success_rate'] * 100:.1f}%")
    print(f"\nСредние метрики (только успешные):")
    print(f"  Время выполнения: {avg_time:.2f}с")
    print(f"  Длина ответа: {avg_length:.0f} символов")
    print(f"  Количество слов: {avg_words:.0f}")
    print(f"  Количество источников: {avg_citations:.1f}")
    
    if avg_retrieval_metrics:
        print(f"\nСредние метрики retrieval:")
        for metric_name, metric_value in avg_retrieval_metrics.items():
            print(f"  {metric_name}: {metric_value:.3f}")
    
    print(f"\n{'=' * 60}")
    print(f"Эксперимент завершен: {experiment_name}")
    print(f"Session ID: {session_id}")
    print("Результаты можно просмотреть в интерфейсе Langfuse:")
    langfuse_url = (
        os.getenv("LANGFUSE_BASE_URL")
        or os.getenv("LANGFUSE_HOST")
        or "http://localhost:3000"
    )
    print(f"{langfuse_url}/traces?sessionId={session_id}")
    print(f"Все traces связаны с session_id: {session_id}")
    print(f"{'=' * 60}\n")
    
    return {
        "summary": summary,
        "results": results,
        "success": True,
    }


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="rag_python_qa",
        help="Имя датасета в Langfuse (по умолчанию: rag_python_qa)",
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Имя эксперимента (по умолчанию генерируется автоматически)",
    )
    
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="ID сессии (по умолчанию генерируется автоматически)",
    )
    
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Путь к файлу для сохранения результатов (JSON)",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Путь к файлу конфигурации (по умолчанию: source/config.yaml)",
    )
    
    args = parser.parse_args()
    
    try:
        print("Инициализация RAG системы...")
        config = Config(config_path=args.config)
        embedding_generator = EmbeddingGenerator(config)
        vector_store = VectorStore(config, embedding_generator)
        rag = RAGPipeline(config, embedding_generator, vector_store)
        print("RAG система инициализирована успешно")
    except Exception as e:
        print(f"Ошибка инициализации RAG системы: {e}")
        return 1
    
    result = run_experiment_on_dataset(
        rag=rag,
        dataset_name=args.dataset_name,
        experiment_name=args.experiment_name,
        session_id=args.session_id,
    )
    
    if args.save_results:
        output_path = Path(args.save_results)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Результаты сохранены в {output_path}")
    
    return 0 if result.get("success", False) else 1


if __name__ == "__main__":
    sys.exit(main())

