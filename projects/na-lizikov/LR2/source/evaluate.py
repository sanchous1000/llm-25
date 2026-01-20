import json
import os
import sys
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import yaml

# Определяем путь к корню проекта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Добавляем путь к scripts для импорта
sys.path.insert(0, SCRIPT_DIR)
from retrieve import hybrid_search

# Имя индекса в Elasticsearch
INDEX_NAME = "rag_docs"

# Подключение к Elasticsearch
es = Elasticsearch("http://localhost:9200")

if not es.ping():
    raise RuntimeError("Elasticsearch недоступен. Проверь, что контейнер запущен.")

if not es.indices.exists(index=INDEX_NAME):
    raise RuntimeError(f"Индекс '{INDEX_NAME}' не существует. Сначала запусти create_index.py и load_to_elasticsearch.py")


def normalize_path(path):
    """Нормализует путь для сравнения (убирает различия в слешах)"""
    if not path:
        return ""
    # Заменяем все слеши на обратные (Windows формат)
    return path.replace("/", "\\")


def evaluate(k_values=[5, 10], verbose=False):
    """
    Считает Recall@k, Precision@k и MRR на основе questions.json
    """
    QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "data", "questions.json")
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = {k: {"recall": 0.0, "precision": 0.0, "mrr": 0.0} for k in k_values}

    print(f"Оценка на {len(questions)} вопросах...")
    for idx, item in enumerate(questions, 1):
        question = item["question"]
        # Нормализуем ожидаемые пути
        relevant_paths = set(normalize_path(p) for p in item["relevant_chunks"])
        
        if not relevant_paths:
            print(f"Предупреждение: вопрос {idx} не имеет relevant_chunks")
            continue

        # Используем hybrid_search для получения результатов
        hits = hybrid_search(question, k=max(k_values))
        
        if not hits:
            if verbose:
                print(f"Предупреждение: для вопроса {idx} не найдено результатов")
            continue

        # Извлекаем и нормализуем relative_path из результатов поиска
        retrieved_paths = [
            normalize_path(hit["_source"].get("relative_path", "")) for hit in hits
        ]

        if verbose:
            print(f"\nВопрос {idx}: {question[:60]}...")
            print(f"  Ожидаемые пути: {list(relevant_paths)}")
            print(f"  Найденные пути (top-{max(k_values)}): {retrieved_paths[:max(k_values)]}")

        for k in k_values:
            top_k_paths = retrieved_paths[:k]
            
            # Recall@k: доля релевантных документов, которые были найдены
            found_relevant = [path for path in top_k_paths if path in relevant_paths]
            recall = len(found_relevant) / len(relevant_paths) if relevant_paths else 0.0
            results[k]["recall"] += recall

            # Precision@k: доля найденных документов, которые релевантны
            precision = len(found_relevant) / k if k > 0 else 0.0
            results[k]["precision"] += precision

            # MRR: обратный ранг первого релевантного документа
            rr = 0.0
            for rank, path in enumerate(top_k_paths, start=1):
                if path in relevant_paths:
                    rr = 1.0 / rank
                    break
            results[k]["mrr"] += rr
            
            if verbose and found_relevant:
                print(f"  Найдено релевантных для k={k}: {found_relevant}")

    n = len(questions)
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ:")
    print("="*50)
    for k in k_values:
        recall = results[k]["recall"] / n
        precision = results[k]["precision"] / n
        mrr = results[k]["mrr"] / n
        print(f"\nМетрики для k={k}:")
        print(f"  Recall@{k}:    {recall:.3f}")
        print(f"  Precision@{k}: {precision:.3f}")
        print(f"  MRR:           {mrr:.3f}")
    print("="*50)


if __name__ == "__main__":
    import sys
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    evaluate(verbose=verbose)
