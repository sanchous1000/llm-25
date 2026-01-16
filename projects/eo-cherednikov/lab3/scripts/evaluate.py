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
    name="vllm_docs",
    host=os.getenv("QDRANT_HOST", default="localhost"),
    port=os.getenv("QDRANT_PORT", default=6333)
)


def recall_evaluator(*args, **kwargs):
    """
    Вычисляет Recall@k метрики.
    Langfuse передает параметры как keyword arguments или позиционные.
    """
    # Извлекаем параметры из kwargs или args
    # Используем 'in' вместо 'or', чтобы не потерять пустые списки
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
    
    # output - это список ID, который возвращает функция task
    retrieved_ids = output if isinstance(output, list) else []
    expected_ids = set(expected_output) if expected_output else set()
    
    evals = []
    for k in k_values:
        retrieved_k = retrieved_ids[:k]
        retrieved_set = set(retrieved_k)
        
        # Находим пересечение - релевантные документы среди найденных
        matched_ids = retrieved_set & expected_ids
        
        recall = len(matched_ids) / len(expected_ids) if expected_ids else 0.0
        evals.append(Evaluation(
            name=f"recall@{k}",
            value=recall,
            comment=f"Retrieved {len(matched_ids)}/{len(expected_ids)} relevant docs in top-{k}"
        ))
    return evals


def precision_evaluator(*args, **kwargs):
    """
    Вычисляет Precision@k метрики.
    Langfuse передает параметры как keyword arguments или позиционные.
    """
    # Извлекаем параметры из kwargs или args
    # Используем 'in' вместо 'or', чтобы не потерять пустые списки
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
    
    # output - это список ID, который возвращает функция task
    retrieved_ids = output if isinstance(output, list) else []
    expected_ids = set(expected_output) if expected_output else set()
    
    evals = []
    for k in k_values:
        retrieved_k = retrieved_ids[:k]
        retrieved_set = set(retrieved_k)
        
        # Находим пересечение - релевантные документы среди найденных
        matched_ids = retrieved_set & expected_ids
        
        precision = len(matched_ids) / k if k > 0 else 0.0
        evals.append(Evaluation(
            name=f"precision@{k}",
            value=precision,
            comment=f"{len(matched_ids)}/{k} retrieved docs were relevant"
        ))
    return evals


def mrr_evaluator(*args, **kwargs):
    """
    Вычисляет MRR (Mean Reciprocal Rank) метрику.
    Langfuse передает параметры как keyword arguments или позиционные.
    """
    # Извлекаем параметры из kwargs или args
    # Используем 'in' вместо 'or', чтобы не потерять пустые списки
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
    
    # output - это список ID, который возвращает функция task
    retrieved_ids = output if isinstance(output, list) else []
    expected_ids = set(expected_output) if expected_output else set()
    
    # Ищем первый релевантный документ в списке
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in expected_ids:
            rr = 1.0 / rank
            return [Evaluation(
                name="MRR",
                value=rr,
                comment=f"First relevant doc at rank {rank}"
            )]
    
    # Если релевантных документов не найдено
    return [Evaluation(
        name="MRR",
        value=0.0,
        comment="No relevant docs found"
    )]


def task(item,
         embed_model: str = "nomic-embed-text",
         ollama_host: str = "http://localhost:11434",
         top_k: int = 10):
    question = item.input  # `run_experiment` passes a `DatasetItemClient` to the task function. The input of the dataset item is available as `item.input`.

    query_vec = embed_query(question, model=embed_model, host=ollama_host)

    results = qdrant.search(query_vector=query_vec, top_k=top_k)
    retrieved_ids = [r.payload.get("id", "") for r in results]

    return retrieved_ids


dataset = langfuse.get_dataset("vllm_questions")

k_values = [5, 10]

# Обкртки для evaluator функций с k_values
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