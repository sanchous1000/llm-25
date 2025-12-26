#!/usr/bin/env python3
"""
Скрипт для создания Dataset в Langfuse на основе test_questions.json
"""
import sys
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langfuse import Langfuse

# Добавляем путь к source для импорта конфигурации
sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

load_dotenv()


def create_rag_dataset(
    dataset_name: str = "rag_python_qa",
    test_questions_file: str = None,
):
    """
    Создает Dataset в Langfuse для проведения экспериментов с RAG системой
    
    Args:
        dataset_name: Имя датасета в Langfuse
        test_questions_file: Путь к файлу с тестовыми вопросами
    """
    project_root = Path(__file__).parent.parent
    
    if test_questions_file is None:
        test_questions_file = project_root / "data" / "test_questions.json"
    else:
        test_questions_file = Path(test_questions_file)
    
    if not test_questions_file.exists():
        print(f"Файл с тестовыми вопросами не найден: {test_questions_file}")
        print("Пожалуйста, создайте test_questions.json в формате:")
        print("""
[
  {
    "query": "текст вопроса",
    "relevant_chunk_ids": [0, 1, 2]
  }
]
        """)
        return
    
    # Загружаем тестовые вопросы
    with open(test_questions_file, "r", encoding="utf-8") as f:
        test_queries = json.load(f)
    
    print(f"Загружено {len(test_queries)} тестовых вопросов")
    
    # Инициализируем Langfuse
    langfuse_url = (
        os.getenv("LANGFUSE_BASE_URL")
        or os.getenv("LANGFUSE_HOST")
        or "http://localhost:3000"
    )
    
    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        host=langfuse_url,
    )
    
    # Создаем или получаем датасет
    print(f"\nСоздание датасета '{dataset_name}'...")
    
    try:
        langfuse.create_dataset(name=dataset_name)
        print(f"Датасет '{dataset_name}' создан/получен успешно.")
    except Exception as e:
        if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
            print(f"Датасет '{dataset_name}' уже существует, продолжаем...")
        else:
            print(f"Ошибка при создании датасета: {e}")
            raise
    
    # Добавляем элементы в датасет
    print(f"\nДобавление {len(test_queries)} элементов в датасет...")
    
    added_items = 0
    for i, query_data in enumerate(test_queries, 1):
        query = query_data["query"]
        relevant_chunk_ids = query_data.get("relevant_chunk_ids", [])
        
        # Формируем input для датасета
        input_data = {
            "query": query,
        }
        
        # Формируем expected_output (можно оставить пустым или добавить описание)
        expected_output = {
            "relevant_chunk_ids": relevant_chunk_ids,
            "description": f"Ожидается, что ответ будет основан на чанках с ID: {relevant_chunk_ids}",
        }
        
        # Метаданные для дополнительной информации
        metadata = {
            "num_relevant_chunks": len(relevant_chunk_ids),
            "relevant_chunk_ids": relevant_chunk_ids,
            "source": "test_questions.json",
        }
        
        try:
            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                input=input_data,
                expected_output=expected_output,
                metadata=metadata,
            )
            print(f"  [{i}/{len(test_queries)}] Добавлен вопрос: {query[:60]}...")
            added_items += 1
        except Exception as e:
            print(f"  [{i}/{len(test_queries)}] Ошибка при добавлении вопроса '{query[:60]}...': {e}")
    
    langfuse.flush()
    
    print(f"\n{'=' * 60}")
    print(f"Датасет '{dataset_name}' успешно создан!")
    print(f"Добавлено элементов: {added_items}/{len(test_queries)}")
    print(f"{'=' * 60}")
    print(f"\nДатасет можно просмотреть в интерфейсе Langfuse:")
    print(f"{langfuse_url}/datasets/{dataset_name}")
    print(f"\nИспользуйте этот датасет для оценки ответов RAG системы через Experiment Run.")
    print(f"\nПример использования:")
    print(f"  from langfuse import Langfuse")
    print(f"  langfuse = Langfuse(...)")
    print(f"  dataset = langfuse.get_dataset('{dataset_name}')")
    print(f"  # Запуск эксперимента с датасетом")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Создание Dataset в Langfuse для проведения экспериментов с RAG системой"
    )
    
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="rag_python_qa",
        help="Имя датасета в Langfuse (по умолчанию: rag_python_qa)",
    )
    
    parser.add_argument(
        "--test-questions-file",
        type=str,
        default=None,
        help="Путь к файлу с тестовыми вопросами (по умолчанию: data/test_questions.json)",
    )
    
    args = parser.parse_args()
    
    create_rag_dataset(
        dataset_name=args.dataset_name,
        test_questions_file=args.test_questions_file,
    )

