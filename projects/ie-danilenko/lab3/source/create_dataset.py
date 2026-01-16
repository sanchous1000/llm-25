"""
Скрипт для создания датасета в Langfuse из вопросов и релевантных документов.
Выполняет задание 4: загрузка датасета в Langfuse.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from dotenv import load_dotenv

from langfuse import Langfuse
from evaluate_retrieval import RetrievalEvaluator
from utils import load_questions, get_device

# Загружаем переменные окружения
load_dotenv()


def create_langfuse_client() -> Optional[Langfuse]:
    """
    Создает клиент Langfuse из переменных окружения.
    
    Returns:
        Клиент Langfuse или None, если не настроен
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    
    if not public_key or not secret_key:
        print("⚠️  Предупреждение: LANGFUSE_PUBLIC_KEY и LANGFUSE_SECRET_KEY не установлены")
        return None
    
    try:
        return Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
    except Exception as e:
        print(f"⚠️  Предупреждение: не удалось создать клиент Langfuse: {e}")
        return None


def find_relevant_chunks(
    evaluator: RetrievalEvaluator,
    question: str,
    top_k: int = 10,
    search_type: str = "hybrid"
) -> List[Dict[str, Any]]:
    """
    Находит релевантные чанки для вопроса.
    
    Args:
        evaluator: Оценщик для поиска
        question: Вопрос
        top_k: Количество релевантных чанков
        search_type: Тип поиска (dense, sparse, hybrid)
    
    Returns:
        Список релевантных чанков с метаданными
    """
    try:
        if search_type == "dense":
            results = evaluator.search_dense(question, k=top_k)
        elif search_type == "sparse":
            results = evaluator.search_sparse(question, k=top_k)
        elif search_type == "hybrid":
            results = evaluator.search_hybrid(question, k=top_k)
        else:
            raise ValueError(f"Неизвестный тип поиска: {search_type}")
        
        # Формируем список релевантных чанков
        relevant_chunks = []
        for result in results:
            chunk = result.get("chunk", {})
            if chunk:
                relevant_chunks.append({
                    "text": chunk.get("text", ""),
                    "metadata": chunk.get("metadata", {}),
                    "index": result.get("index"),
                    "score": result.get("distance", 0.0)
                })
        
        return relevant_chunks
    except Exception as e:
        print(f"⚠️  Ошибка при поиске релевантных чанков для вопроса '{question[:50]}...': {e}")
        return []


def create_dataset_in_langfuse(
    langfuse_client: Langfuse,
    dataset_name: str,
    questions: List[Dict[str, Any]],
    relevant_chunks_map: Dict[int, List[Dict[str, Any]]],
    description: Optional[str] = None
) -> None:
    """
    Создает датасет в Langfuse.
    
    Args:
        langfuse_client: Клиент Langfuse
        dataset_name: Название датасета
        questions: Список вопросов
        relevant_chunks_map: Словарь с релевантными чанками для каждого вопроса (по ID)
        description: Описание датасета
    """
    try:
        # Создаем датасет
        print(f"\n📦 Создание датасета '{dataset_name}' в Langfuse...")
        dataset = langfuse_client.create_dataset(
            name=dataset_name,
            description=description or f"Датасет из {len(questions)} вопросов для оценки RAG-системы"
        )
        print(f"✅ Датасет '{dataset_name}' создан")
        
        # Добавляем элементы датасета
        print(f"\n📝 Добавление элементов в датасет...")
        for question_data in questions:
            question_id = question_data["id"]
            question_text = question_data["question"]
            relevant_chunks = relevant_chunks_map.get(question_id, [])
            
            # Формируем expected_output с релевантными чанками
            expected_output = {
                "relevant_chunks": [
                    {
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                        "score": chunk.get("score", 0.0)
                    }
                    for chunk in relevant_chunks
                ],
                "num_chunks": len(relevant_chunks)
            }
            
            # Создаем элемент датасета
            try:
                langfuse_client.create_dataset_item(
                    dataset_name=dataset_name,
                    input={"question": question_text},
                    expected_output=expected_output,
                    metadata={
                        "question_id": question_id,
                        "num_relevant_chunks": len(relevant_chunks)
                    }
                )
                print(f"  ✅ Добавлен вопрос {question_id}: {question_text[:60]}...")
            except Exception as e:
                print(f"  ❌ Ошибка при добавлении вопроса {question_id}: {e}")
        
        print(f"\n✅ Датасет '{dataset_name}' успешно создан с {len(questions)} элементами")
        
    except Exception as e:
        print(f"❌ Ошибка при создании датасета: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Создание датасета в Langfuse из вопросов')
    parser.add_argument('--questions-file', type=str, default='questions.md',
                       help='Путь к файлу с вопросами (по умолчанию: questions.md)')
    parser.add_argument('--faiss-index-dir', type=str, default='faiss_index',
                       help='Директория с индексом Faiss')
    parser.add_argument('--chunks-dir', type=str, default='chunks',
                       help='Директория с чанками')
    parser.add_argument('--dense-model', type=str, default='intfloat/multilingual-e5-large',
                       help='Модель для dense эмбеддингов')
    parser.add_argument('--dataset-name', type=str, default='answers',
                       help='Название датасета в Langfuse')
    parser.add_argument('--search-type', type=str, choices=['dense', 'sparse', 'hybrid'],
                       default='hybrid', help='Тип поиска для нахождения релевантных чанков')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Количество релевантных чанков для каждого вопроса')
    parser.add_argument('--device', type=str, default=None,
                       help='Устройство для вычислений (mps/cuda/cpu)')
    parser.add_argument('--description', type=str, default=None,
                       help='Описание датасета')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Создание датасета в Langfuse")
    print("=" * 60)
    
    # Проверяем наличие клиента Langfuse
    langfuse_client = create_langfuse_client()
    if not langfuse_client:
        print("❌ Ошибка: не удалось создать клиент Langfuse")
    
    # Загружаем вопросы
    print(f"\n📖 Загрузка вопросов из {args.questions_file}...")
    try:
        questions = load_questions(args.questions_file)
        print(f"✅ Загружено вопросов: {len(questions)}")
    except FileNotFoundError as e:
        print(f"❌ Ошибка: {e}")
        return
    
    # Инициализируем оценщик для поиска релевантных чанков
    print(f"\n🔧 Инициализация оценщика для поиска релевантных чанков...")
    device = get_device(args.device) if args.device else None
    evaluator = RetrievalEvaluator(
        faiss_index_dir=args.faiss_index_dir,
        chunks_dir=args.chunks_dir,
        dense_model_name=args.dense_model,
        device=device
    )
    print("✅ Оценщик инициализирован")
    
    # Находим релевантные чанки для каждого вопроса
    print(f"\n🔍 Поиск релевантных чанков для каждого вопроса...")
    print(f"   Тип поиска: {args.search_type}, top_k: {args.top_k}")
    relevant_chunks_map = {}
    
    for i, question_data in enumerate(questions, 1):
        question_text = question_data["question"]
        print(f"\n  [{i}/{len(questions)}] Вопрос: {question_text[:60]}...")
        
        relevant_chunks = find_relevant_chunks(
            evaluator,
            question_text,
            top_k=args.top_k,
            search_type=args.search_type
        )
        
        relevant_chunks_map[question_data["id"]] = relevant_chunks
        print(f"    ✅ Найдено релевантных чанков: {len(relevant_chunks)}")
    
    # Создаем датасет в Langfuse
    create_dataset_in_langfuse(
        langfuse_client,
        args.dataset_name,
        questions,
        relevant_chunks_map,
        description=args.description
    )
    
    print("\n" + "=" * 60)
    print("✅ Создание датасета завершено!")
    print(f"📊 Датасет '{args.dataset_name}' доступен в Langfuse")
    print("=" * 60)


if __name__ == "__main__":
    main()
