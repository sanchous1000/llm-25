import os
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()

# Инициализация Langfuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL")
    )


def create_dataset():
    # 1. Создаем сам датасет
    dataset = langfuse.create_dataset(name="dataset_2")
    # 2. Добавляем примеры
    test_cases = [
          {
              "question": "What are neural networks and how to build them from scratch?",
              "relevant_documents": ["micrograd__README.md", "nn-zero-to-hero__README.md", "optim__README.md"]
          },    
          {
              "question": "What are transformers and how do they work?",
              "relevant_documents": ["transformers__README.md", "nanoGPT__README.md", "minGPT__README.md"]
          },
          {
              "question": "How to train large language models efficiently?",
              "relevant_documents": ["nanoGPT__README.md", "minGPT__README.md"]
          },
          {
              "question": "What is backpropagation and how does it work?",
              "relevant_documents": ["micrograd__README.md", "nn-zero-to-hero__README.md"]
          },
          {
              "question": "Explain attention mechanism in transformers",
              "relevant_documents": ["transformers__README.md", "nanoGPT__README.md"]
          }
    ]

    for case in test_cases:
        item = langfuse.create_dataset_item(
            dataset_name=dataset.name,
            input={"question": case["question"]},
            expected_output={"relevant_documents": case["relevant_documents"]},
            metadata={"source": "manual"}
        )
        print(f"  ➕ Пример добавлен: {item.id} | Вопрос: '{case['question'][:50]}...'")

    print(f"\nДатасет '{dataset.name}' создан с {len(test_cases)} примерами")
    return dataset


if __name__ == "__main__":
    dataset = create_dataset()
    print(f"\nГотово! Dataset ID: {dataset.name}")