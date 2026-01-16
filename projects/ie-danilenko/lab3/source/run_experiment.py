"""
Скрипт для запуска оценки RAG-пайплайна через Langfuse Experiment Run.
Выполняет задание 5: оценка и логирование метрик по датасету.
"""

from typing import List, Dict, Any, Optional
import argparse
from dotenv import load_dotenv

from langfuse import Evaluation

from rag_pipeline import RAGPipeline, create_llm_client, create_langfuse_client
from utils import get_device, load_chunks
from utils.metrics import calculate_recall_at_k, calculate_precision_at_k, calculate_mrr

# Загружаем переменные окружения
load_dotenv()


def find_chunk_index_by_content(
    chunks: List[Dict[str, Any]],
    target_text: str,
    target_metadata: Dict[str, Any]
) -> Optional[int]:
    """
    Находит индекс чанка в корпусе по тексту и метаданным.
    
    Args:
        chunks: Список всех чанков
        target_text: Текст целевого чанка
        target_metadata: Метаданные целевого чанка
    
    Returns:
        Индекс чанка или None, если не найден
    """
    if not target_text:
        return None
    
    target_text_normalized = target_text.strip().lower()
    
    # Сначала ищем точное совпадение текста
    for idx, chunk in enumerate(chunks):
        chunk_text = chunk.get("text", "").strip().lower()
        if chunk_text == target_text_normalized:
            # Если есть метаданные, проверяем их для дополнительной уверенности
            if target_metadata:
                chunk_metadata = chunk.get("metadata", {})
                
                # Проверяем ключевые метаданные
                metadata_matches = []
                
                if target_metadata.get("source_file"):
                    metadata_matches.append(
                        target_metadata.get("source_file") == chunk_metadata.get("source_file")
                    )
                
                if target_metadata.get("repository"):
                    metadata_matches.append(
                        target_metadata.get("repository") == chunk_metadata.get("repository")
                    )
                
                # Если есть метаданные для проверки, все должны совпадать
                if metadata_matches and all(metadata_matches):
                    return idx
                elif not metadata_matches:
                    # Если нет метаданных для проверки, возвращаем по тексту
                    return idx
            else:
                return idx
    
    # Если точное совпадение не найдено, ищем по метаданным и частичному совпадению текста
    if target_metadata:
        best_match_idx = None
        best_match_score = 0
        
        for idx, chunk in enumerate(chunks):
            chunk_metadata = chunk.get("metadata", {})
            chunk_text = chunk.get("text", "").strip().lower()
            
            # Вычисляем score совпадения
            score = 0
            
            # Проверяем метаданные
            if target_metadata.get("source_file") == chunk_metadata.get("source_file"):
                score += 3
            if target_metadata.get("repository") == chunk_metadata.get("repository"):
                score += 2
            if target_metadata.get("Header 1") == chunk_metadata.get("Header 1"):
                score += 1
            
            # Проверяем частичное совпадение текста
            if target_text_normalized in chunk_text or chunk_text in target_text_normalized:
                # Вычисляем процент совпадения
                shorter_len = min(len(target_text_normalized), len(chunk_text))
                longer_len = max(len(target_text_normalized), len(chunk_text))
                if shorter_len > 0:
                    overlap_ratio = shorter_len / longer_len
                    score += overlap_ratio * 2
            
            if score > best_match_score and score >= 3:  # Минимальный порог
                best_match_score = score
                best_match_idx = idx
        
        return best_match_idx
    
    return None


def get_relevant_indices_from_expected_output(
    expected_output: Dict[str, Any],
    all_chunks: List[Dict[str, Any]]
) -> List[int]:
    """
    Извлекает индексы релевантных чанков из expected_output.
    
    Args:
        expected_output: Ожидаемый вывод из датасета
        all_chunks: Все чанки из корпуса
    
    Returns:
        Список индексов релевантных чанков
    """
    relevant_chunks = expected_output.get("relevant_chunks", [])
    relevant_indices = []
    
    for relevant_chunk in relevant_chunks:
        # Сначала проверяем, есть ли индекс напрямую
        if "index" in relevant_chunk:
            idx = relevant_chunk["index"]
            if isinstance(idx, int) and 0 <= idx < len(all_chunks):
                relevant_indices.append(idx)
                continue
        
        # Если индекса нет, ищем по тексту и метаданным
        chunk_text = relevant_chunk.get("text", "")
        chunk_metadata = relevant_chunk.get("metadata", {})
        
        idx = find_chunk_index_by_content(all_chunks, chunk_text, chunk_metadata)
        if idx is not None:
            relevant_indices.append(idx)
    
    return relevant_indices


def rag_task(*, item, rag_pipeline: RAGPipeline, **kwargs) -> Dict[str, Any]:
    """
    Задача для выполнения RAG-пайплайна на элементе датасета.
    
    Args:
        item: Элемент датасета (DatasetItemClient) с input и expected_output
        rag_pipeline: RAG-пайплайн
        **kwargs: Дополнительные параметры
    
    Returns:
        Словарь с результатами: answer, retrieved_chunks, retrieved_indices
    """
    question = item.input["question"]
    
    # Выполняем RAG-пайплайн
    result = rag_pipeline.generate_answer(question, session_id=None)
    
    # Извлекаем индексы извлеченных чанков
    retrieved_chunks = result.get("context", [])
    retrieved_indices = [chunk_result.get("index") for chunk_result in retrieved_chunks if chunk_result.get("index") is not None]
    
    return {
        "answer": result.get("answer", ""),
        "raw_answer": result.get("raw_answer", ""),
        "retrieved_chunks": retrieved_chunks,
        "retrieved_indices": retrieved_indices,
        "sources": result.get("sources", [])
    }


def run_evaluator(
    *,
    input: Dict[str, Any],
    output: Dict[str, Any],
    expected_output: Optional[Dict[str, Any]] = None,
    all_chunks: List[Dict[str, Any]] = None,
    k_values: List[int] = [5, 10],
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Оценщик для вычисления retrieval-метрик.
    
    Args:
        input: Входные данные (вопрос)
        output: Выходные данные (результат RAG-пайплайна)
        expected_output: Ожидаемый вывод (релевантные чанки)
        all_chunks: Все чанки из корпуса
        k_values: Значения k для метрик
        **kwargs: Дополнительные параметры
    
    Returns:
        Список Evaluation объектов с метриками
    """
    # Получаем индексы извлеченных чанков
    retrieved_indices = output.get("retrieved_indices", [])
    
    # Получаем индексы релевантных чанков из expected_output
    relevant_indices = get_relevant_indices_from_expected_output(expected_output, all_chunks)
    
    # Вычисляем метрики
    evaluations = []
    
    for k in k_values:
        recall = calculate_recall_at_k(retrieved_indices, relevant_indices, k)
        precision = calculate_precision_at_k(retrieved_indices, relevant_indices, k)
        
        evaluations.append(Evaluation(
            name=f"Recall@{k}",
            value=recall,
            comment=f"Recall@{k} = {recall:.4f}"
        ))
        
        evaluations.append(Evaluation(
            name=f"Precision@{k}",
            value=precision,
            comment=f"Precision@{k} = {precision:.4f}"
        ))
    
    mrr = calculate_mrr(retrieved_indices, relevant_indices)
    evaluations.append(Evaluation(
        name="MRR",
        value=mrr,
        comment=f"MRR = {mrr:.4f}"
    ))
    
    return evaluations


def main():
    parser = argparse.ArgumentParser(description='Запуск оценки RAG-пайплайна через Langfuse Experiment Run')
    parser.add_argument('--dataset-name', type=str, default="answers",
                       help='Название датасета в Langfuse')
    parser.add_argument('--faiss-index-dir', type=str, default='faiss_index',
                       help='Директория с индексом Faiss')
    parser.add_argument('--chunks-dir', type=str, default='chunks',
                       help='Директория с чанками')
    parser.add_argument('--dense-model', type=str, default='intfloat/multilingual-e5-large',
                       help='Модель для dense эмбеддингов')
    parser.add_argument('--llm-model', type=str, default='qwen3:latest',
                       help='Название модели LLM из Ollama')
    parser.add_argument('--search-type', type=str, choices=['dense', 'sparse', 'hybrid'],
                       default='hybrid', help='Тип поиска')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Количество релевантных чанков для контекста')
    parser.add_argument('--device', type=str, default=None,
                       help='Устройство для вычислений (mps/cuda/cpu)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Температура для генерации')
    parser.add_argument('--max-tokens', type=int, default=1000,
                       help='Максимальное количество токенов в ответе')
    parser.add_argument('--k-values', type=int, nargs='+', default=[5, 10],
                       help='Значения k для метрик Recall@k и Precision@k')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Название эксперимента (по умолчанию: dataset_name + конфигурация)')
    parser.add_argument('--description', type=str, default=None,
                       help='Описание эксперимента')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Запуск оценки RAG-пайплайна через Langfuse Experiment Run")
    print("=" * 60)
    
    # Создаем клиент Langfuse
    print(f"\n🔧 Инициализация Langfuse клиента...")
    langfuse_client = create_langfuse_client()
    print("✅ Langfuse клиент создан")
    
    # Загружаем датасет из Langfuse
    print(f"\n📦 Загрузка датасета '{args.dataset_name}' из Langfuse...")
    dataset = langfuse_client.get_dataset(args.dataset_name)
    dataset_items = list(dataset.items)
    print(f"✅ Датасет загружен: {len(dataset_items)} элементов")
    
    # Загружаем все чанки для сопоставления
    print(f"\n📚 Загрузка чанков из {args.chunks_dir}...")
    all_chunks = load_chunks(args.chunks_dir)
    print(f"✅ Загружено чанков: {len(all_chunks)}")
    
    # Создаем клиент LLM
    print(f"\n🔧 Инициализация LLM клиента...")
    llm_client = create_llm_client()
    print("✅ LLM клиент создан")
    
    # Инициализируем RAG-пайплайн
    print(f"\n🔧 Инициализация RAG-пайплайна...")
    device = get_device(args.device) if args.device else None
    rag_pipeline = RAGPipeline(
        faiss_index_dir=args.faiss_index_dir,
        chunks_dir=args.chunks_dir,
        dense_model_name=args.dense_model,
        llm_client=llm_client,
        llm_model=args.llm_model,
        search_type=args.search_type,
        top_k=args.top_k,
        device=device,
        langfuse_client=langfuse_client  # Передаем клиент для логирования
    )
    print("✅ RAG-пайплайн инициализирован")
    
    # Формируем название эксперимента
    if not args.experiment_name:
        experiment_name = f"{args.dataset_name}_eval_{args.search_type}_top{args.top_k}"
    else:
        experiment_name = args.experiment_name
    
    # Формируем описание
    if not args.description:
        description = f"Оценка RAG-пайплайна на датасете '{args.dataset_name}'. Конфигурация: search_type={args.search_type}, top_k={args.top_k}, llm_model={args.llm_model}"
    else:
        description = args.description
    
    # Создаем обёртку для task функции
    def task_wrapper(*, item, **kwargs):
        return rag_task(item=item, rag_pipeline=rag_pipeline, **kwargs)
    
    # Создаем обёртку для evaluator функции
    def evaluator_wrapper(*, input, output, expected_output, **kwargs):
        return run_evaluator(
            input=input,
            output=output,
            expected_output=expected_output,
            all_chunks=all_chunks,
            k_values=args.k_values,
            **kwargs
        )
    
    # Запускаем эксперимент на датасете
    print(f"\n🚀 Запуск эксперимента '{experiment_name}'...")
    print(f"   Описание: {description}")
    print(f"   Количество элементов: {len(dataset_items)}")
    print(f"   Конфигурация: search_type={args.search_type}, top_k={args.top_k}, k_values={args.k_values}")
    
    result = dataset.run_experiment(
        name=experiment_name,
        description=description,
        task=task_wrapper,
        evaluators=[evaluator_wrapper],
        metadata={
            "search_type": args.search_type,
            "top_k": args.top_k,
            "llm_model": args.llm_model,
            "dense_model": args.dense_model,
            "k_values": args.k_values,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens
        }
    )
    
    print(f"\n✅ Эксперимент завершен!")
    print(f"\n📊 Результаты:")
    print(result.format())
    
    # Отправляем все данные на сервер Langfuse
    print(f"\n💾 Сохранение данных в Langfuse...")
    langfuse_client.flush()
    print(f"✅ Данные сохранены")
    
    print(f"\n{'='*60}")
    print("✅ Оценка завершена!")
    print(f"📊 Результаты доступны в интерфейсе Langfuse")
    print("=" * 60)


if __name__ == "__main__":
    main()
