"""
Создание датасета Langfuse из размеченных данных Lab2.

Использует файлы expected_chunks_*.json из lab2/data/evaluation/
для создания датасета в Langfuse с правильными chunk_id.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Langfuse  # type: ignore


def load_lab2_expected_chunks(version: str) -> dict:
    """Загрузить размеченные данные из lab2.
    
    Args:
        version: Версия индекса (например, '4530f7569b81' или 'e6233de3342d')
    
    Returns:
        dict: Словарь {вопрос: [chunk_ids]}
    """
    lab3_dir = Path(__file__).parent.parent
    lab2_dir = lab3_dir.parent / "lab2"
    
    expected_chunks_file = lab2_dir / "data" / "evaluation" / f"expected_chunks_{version}.json"
    
    if not expected_chunks_file.exists():
        raise FileNotFoundError(f"Файл не найден: {expected_chunks_file}")
    
    with open(expected_chunks_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data


def create_langfuse_dataset(
    dataset_name: str,
    questions_and_chunks: dict,
    metadata: dict = None,
) -> None:
    """Создать датасет в Langfuse.
    
    Args:
        dataset_name: Имя датасета в Langfuse
        questions_and_chunks: Словарь {вопрос: [chunk_ids]}
        metadata: Дополнительные метаданные датасета
    """
    # Загрузка .env
    lab3_dir = Path(__file__).parent.parent
    env_path = lab3_dir / "source" / ".env"
    
    if not env_path.exists():
        print(f"❌ Файл .env не найден: {env_path}")
        print("Создайте .env файл с переменными LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST")
        sys.exit(1)
    
    load_dotenv(env_path)
    
    # Инициализация Langfuse
    try:
        langfuse = Langfuse()
        print("✓ Подключено к Langfuse")
    except Exception as e:
        print(f"❌ Ошибка подключения к Langfuse: {e}")
        sys.exit(1)
    
    # Создание датасета
    print(f"\n{'='*60}")
    print(f"СОЗДАНИЕ ДАТАСЕТА: {dataset_name}")
    print("="*60)
    
    try:
        # Создаем датасет
        dataset = langfuse.create_dataset(
            name=dataset_name,
            description=f"RAG evaluation dataset from Lab2 with {len(questions_and_chunks)} questions",
            metadata=metadata or {},
        )
        print(f"✓ Датасет создан: {dataset_name}")
        
    except Exception as e:
        if "already exists" in str(e).lower() or "unique constraint" in str(e).lower():
            print(f"⚠️  Датасет '{dataset_name}' уже существует")
            print("Используется существующий датасет")
            dataset = langfuse.get_dataset(dataset_name)
        else:
            raise e
    
    # Добавление элементов датасета
    print(f"\nДобавление {len(questions_and_chunks)} вопросов...")
    
    for i, (question, chunk_ids) in enumerate(questions_and_chunks.items(), 1):
        try:
            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                input={"question": question},
                expected_output={
                    "expected_chunk_ids": chunk_ids,
                },
                metadata={
                    "question_index": i,
                    "num_expected_chunks": len(chunk_ids),
                },
            )
            print(f"  ✓ [{i}/{len(questions_and_chunks)}] {question[:60]}...")
            
        except Exception as e:
            print(f"  ✗ [{i}/{len(questions_and_chunks)}] Ошибка: {e}")
    
    print(f"\n✓ Датасет '{dataset_name}' готов!")
    print(f"  Вопросов: {len(questions_and_chunks)}")
    print(f"\nПросмотреть в UI: http://localhost:3000/datasets/{dataset_name}")


def main():
    """Главная функция."""
    print("="*60)
    print("СОЗДАНИЕ ДАТАСЕТОВ LANGFUSE ИЗ LAB2")
    print("="*60)
    
    # Конфигурации из lab2
    configs = [
        {
            "version": "4530f7569b81",
            "dataset_name": "fastapi_rag_baseline_recursive_1024",
            "description": "Baseline (Recursive splitter, 1024 tokens, 371 chunks)",
            "metadata": {
                "source": "lab2",
                "splitter_type": "recursive",
                "chunk_size": 1024,
                "chunk_overlap": 128,
                "num_chunks": 371,
            },
        },
        {
            "version": "e6233de3342d",
            "dataset_name": "fastapi_rag_markdown_512",
            "description": "Markdown splitter (512 tokens, 991 chunks)",
            "metadata": {
                "source": "lab2",
                "splitter_type": "markdown",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "num_chunks": 991,
            },
        },
    ]
    
    # Создание датасетов
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Конфигурация: {config['description']}")
        print("="*60)
        
        try:
            # Загрузить размеченные данные
            print(f"\n1. Загрузка размеченных данных (версия {config['version']})...")
            questions_and_chunks = load_lab2_expected_chunks(config["version"])
            print(f"   ✓ Загружено {len(questions_and_chunks)} вопросов")
            
            # Создать датасет в Langfuse
            print(f"\n2. Создание датасета '{config['dataset_name']}'...")
            create_langfuse_dataset(
                dataset_name=config["dataset_name"],
                questions_and_chunks=questions_and_chunks,
                metadata=config["metadata"],
            )
            
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            continue
    
    print("\n" + "="*60)
    print("ГОТОВО!")
    print("="*60)
    print("\nСозданы датасеты:")
    for config in configs:
        print(f"  - {config['dataset_name']}")
    
    print("\nСледующие шаги:")
    print("1. Откройте Langfuse UI: http://localhost:3000")
    print("2. Перейдите в раздел 'Datasets'")
    print("3. Выберите датасет и просмотрите элементы")
    print("4. Запустите эксперименты:")
    print("   python run_experiments_langfuse.py")


if __name__ == "__main__":
    main()
