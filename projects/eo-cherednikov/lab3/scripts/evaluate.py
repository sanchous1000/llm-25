from dotenv import load_dotenv
from langfuse import Langfuse, Evaluation

load_dotenv()

import os

from utils.ollama import embed_query
from utils.qdrant import QdrantCollection

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    base_url=os.getenv("LANGFUSE_BASE_URL"),
)

qdrant = QdrantCollection(
    name="dnd_2024_docs",
    host=os.getenv("QDRANT_HOST", default="localhost"),
    port=os.getenv("QDRANT_PORT", default=6333)
)


def _normalize_item_match(retrieved_item: str, expected_item: str) -> bool:
    if retrieved_item == expected_item:
        return True

    if ':' not in expected_item:
        if ':' in retrieved_item:
            retrieved_doc_id = retrieved_item.split(':')[0]
        else:
            retrieved_doc_id = retrieved_item
        
        if retrieved_doc_id == expected_item:
            return True
    return False


def recall_evaluator(*args, **kwargs):
    if 'output' in kwargs:
        output = kwargs['output']
    elif len(args) > 0:
        output = args[0]
    else:
        output = []
    
    if 'expected_output' in kwargs:
        expected_output = kwargs['expected_output']
    elif len(args) > 1:
        expected_output = args[1]
    else:
        expected_output = []
    
    k_values = kwargs.get('k_values', [5, 10])

    retrieved_items = output if isinstance(output, list) else []
    expected_items = expected_output if expected_output else []
    
    evals = []
    for k in k_values:
        retrieved_k = retrieved_items[:k]
        
        # Подсчитываем совпадения с учетом страниц
        matched_count = 0
        for retrieved_item in retrieved_k:
            for expected_item in expected_items:
                if _normalize_item_match(retrieved_item, expected_item):
                    matched_count += 1
                    break  # Каждый retrieved_item считается только один раз
        
        recall = matched_count / len(expected_items) if expected_items else 0.0
        evals.append(Evaluation(
            name=f"recall@{k}",
            value=recall,
            comment=f"Retrieved {matched_count}/{len(expected_items)} relevant items in top-{k}"
        ))
    return evals


def precision_evaluator(*args, **kwargs):
    if 'output' in kwargs:
        output = kwargs['output']
    elif len(args) > 0:
        output = args[0]
    else:
        output = []
    
    if 'expected_output' in kwargs:
        expected_output = kwargs['expected_output']
    elif len(args) > 1:
        expected_output = args[1]
    else:
        expected_output = []
    
    k_values = kwargs.get('k_values', [5, 10])

    retrieved_items = output if isinstance(output, list) else []
    expected_items = expected_output if expected_output else []
    
    evals = []
    for k in k_values:
        retrieved_k = retrieved_items[:k]

        matched_count = 0
        for retrieved_item in retrieved_k:
            for expected_item in expected_items:
                if _normalize_item_match(retrieved_item, expected_item):
                    matched_count += 1
                    break
        
        precision = matched_count / k if k > 0 else 0.0
        evals.append(Evaluation(
            name=f"precision@{k}",
            value=precision,
            comment=f"{matched_count}/{k} retrieved items were relevant"
        ))
    return evals


def mrr_evaluator(*args, **kwargs):
    if 'output' in kwargs:
        output = kwargs['output']
    elif len(args) > 0:
        output = args[0]
    else:
        output = []
    
    if 'expected_output' in kwargs:
        expected_output = kwargs['expected_output']
    elif len(args) > 1:
        expected_output = args[1]
    else:
        expected_output = []

    retrieved_items = output if isinstance(output, list) else []
    expected_items = expected_output if expected_output else []

    for rank, retrieved_item in enumerate(retrieved_items, 1):
        for expected_item in expected_items:
            if _normalize_item_match(retrieved_item, expected_item):
                rr = 1.0 / rank
                return [Evaluation(
                    name="MRR",
                    value=rr,
                    comment=f"First relevant item at rank {rank}"
                )]

    return [Evaluation(
        name="MRR",
        value=0.0,
        comment="No relevant items found"
    )]


def task(item,
         embed_model: str = "nomic-embed-text",
         ollama_host: str = "http://localhost:11434",
         top_k: int = 10):
    question = item.input  # `run_experiment` passes a `DatasetItemClient` to the task function. The input of the dataset item is available as `item.input`.

    query_vec = embed_query(question, model=embed_model, host=ollama_host)

    results = qdrant.search(query_vector=query_vec, top_k=top_k)
    
    # Формируем идентификаторы с учетом страниц
    retrieved_items = []
    seen_items = set()
    for r in results:
        doc_id = r.payload.get("id", "")
        page_number = r.payload.get("page_number")
        
        if page_number is not None:
            # Для PDF с указанием страницы используем формат "doc_id:page_number"
            item_id = f"{doc_id}:{page_number}"
        else:
            # Для документов без страниц используем только doc_id
            item_id = doc_id
        
        # Убираем дубликаты
        if item_id not in seen_items:
            retrieved_items.append(item_id)
            seen_items.add(item_id)

    return retrieved_items


dataset = langfuse.get_dataset("dnd_2024_questions")

# Определяем k_values для метрик
k_values = [5, 10]

# Создаем обёртки для evaluator функций с k_values
def recall_eval(*args, **kwargs):
    kwargs['k_values'] = k_values
    return recall_evaluator(*args, **kwargs)

def precision_eval(*args, **kwargs):
    kwargs['k_values'] = k_values
    return precision_evaluator(*args, **kwargs)

result = langfuse.run_experiment(
    name="RAG test",
    task=task,
    data=dataset.items,
    evaluators=[recall_eval, precision_eval, mrr_evaluator],
    max_concurrency=1
)

print(result.format())