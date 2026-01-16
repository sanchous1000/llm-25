import json
import os
import argparse
from langfuse import Langfuse
from langfuse.model import CreateDatasetItemRequest
from dotenv import load_dotenv

load_dotenv()

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    parser = argparse.ArgumentParser(description="Загрузка датасета в Langfuse")
    parser.add_argument("--file", default="data/verl_questions.jsonl", help="Путь к файлу с вопросами")
    parser.add_argument("--dataset_name", default="verl_lab_dataset", help="Имя датасета в Langfuse")
    parser.add_argument("--input_key", default="question", help="Ключ JSON, где лежит вопрос")
    parser.add_argument("--output_key", default="answer", help="Ключ JSON, где лежит эталонный ответ")
    
    args = parser.parse_args()

    if not os.environ.get("LANGFUSE_PUBLIC_KEY") or not os.environ.get("LANGFUSE_SECRET_KEY"):
        print("❌ Ошибка: Не заданы LANGFUSE_PUBLIC_KEY или LANGFUSE_SECRET_KEY")
        return

    langfuse = Langfuse()

    if not os.path.exists(args.file):
        print(f"❌ Файл не найден: {args.file}")
        return
        
    items = load_jsonl(args.file)
    print(f"📄 Прочитано записей: {len(items)}")

    try:
        dataset = langfuse.get_dataset(args.dataset_name)
        print(f"ℹ️  Датасет '{args.dataset_name}' уже существует. Добавляем новые элементы.")
    except:
        print(f"✨ Создается новый датасет: '{args.dataset_name}'")
        langfuse.create_dataset(name=args.dataset_name)

    print("🚀 Загрузку в langfuse...")
    count = 0
    for item in items:
        input_data = item.get(args.input_key) or item.get('input')
        expected_output = item.get(args.output_key) or item.get('ground_truth')
        
        if not input_data:
            print(f"⚠️  Пропуск строки (нет ключа '{args.input_key}'): {item}")
            continue

        metadata = {k: v for k, v in item.items() if k not in [args.input_key, args.output_key]}

        langfuse.create_dataset_item(
            dataset_name=args.dataset_name,
            input=input_data,
            expected_output=expected_output,
            metadata=metadata
        )
        count += 1

    langfuse.flush()
    print(f"✅ Успешно загружено {count} элементов в датасет '{args.dataset_name}'")

if __name__ == "__main__":
    main()