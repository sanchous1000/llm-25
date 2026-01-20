"""
Скрипт для загрузки датасета вопросов-ответов в Langfuse
"""
import json
import os
import sys
from dotenv import load_dotenv
from langfuse import Langfuse

# Загружаем переменные окружения
load_dotenv()

# Добавляем путь к LR2 для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "LR2", "scripts"))

# Настройки из переменных окружения
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
)

DATASET_NAME = "python_docs_qa"
QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "LR2", "data", "questions.json")


def load_dataset():
    """Загружает датасет в Langfuse"""
    # Читаем вопросы из LR2
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    print(f"Загрузка {len(questions)} вопросов в Langfuse...")
    
    # Создаем или получаем датасет
    try:
        dataset = langfuse.create_dataset(name=DATASET_NAME)
        print(f"Создан новый датасет: {DATASET_NAME}")
    except Exception as e:
        # Если датасет уже существует, получаем его
        dataset = langfuse.get_dataset(name=DATASET_NAME)
        print(f"Используется существующий датасет: {DATASET_NAME}")
    
    # Добавляем элементы датасета
    for idx, item in enumerate(questions, 1):
        question = item["question"]
        relevant_chunks = item.get("relevant_chunks", [])
        
        # Формируем expected_output как список релевантных путей
        expected_output = {
            "relevant_chunks": relevant_chunks,
            "description": f"Ожидаемые релевантные чанки для вопроса: {question[:100]}..."
        }
        
        try:
            langfuse.create_dataset_item(
                dataset_name=DATASET_NAME,
                input={"question": question},
                expected_output=expected_output,
                metadata={
                    "question_id": idx,
                    "num_relevant_chunks": len(relevant_chunks)
                }
            )
            print(f"✓ Добавлен вопрос {idx}/{len(questions)}: {question[:60]}...")
        except Exception as e:
            print(f"✗ Ошибка при добавлении вопроса {idx}: {e}")
    
    print(f"\nДатасет '{DATASET_NAME}' успешно загружен в Langfuse!")
    print(f"Всего элементов: {len(questions)}")


if __name__ == "__main__":
    load_dataset()
