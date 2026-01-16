import os
import requests
import json
import time
from typing import List, Dict
from dotenv import load_dotenv
from langfuse.decorators import observe, langfuse_context
from langfuse import Langfuse

load_dotenv()

LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")

langfuse = Langfuse(
    secret_key=LANGFUSE_SECRET_KEY,
    public_key=LANGFUSE_PUBLIC_KEY,
    host=LANGFUSE_HOST
)

KNOWLEDGE_BASE = [
    {"id": "doc1", "content": "Langfuse — это платформа для инжиниринга LLM с открытым исходным кодом."},
    {"id": "doc2", "content": "Yandex GPT — это генеративная нейросеть от Яндекса."},
    {"id": "doc3", "content": "RAG (Retrieval Augmented Generation) позволяет LLM использовать внешние данные."},
    {"id": "doc4", "content": "В Санкт-Петербурге много мостов и каналов."},
    {"id": "doc5", "content": "Python — отличный язык для Data Science."}
]

def call_yandex_gpt(messages, temperature=0.6):
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    
    prompt = {
        "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt",
        "completionOptions": {
            "stream": False,
            "temperature": temperature,
            "maxTokens": "2000"
        },
        "messages": messages
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "x-folder-id": YANDEX_FOLDER_ID
    }

    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=prompt)
    except Exception as e:
        raise Exception(f"Connection Error: {e}")

    end_time = time.time()
    
    if response.status_code != 200:
        raise Exception(f"Yandex GPT Error ({response.status_code}): {response.text}")

    result = response.json()
    
    try:
        text = result['result']['alternatives'][0]['message']['text']
        usage = result['result']['usage']
    except KeyError:
        raise Exception(f"Unexpected response structure: {result}")
    
    return {
        "text": text,
        "input_tokens": int(usage['inputTextTokens']),
        "output_tokens": int(usage['completionTokens']),
        "total_tokens": int(usage['totalTokens']),
        "duration": end_time - start_time
    }

@observe(as_type="generation")
def llm_generation_step(context_text: str, question: str):
    """Генерация ответа с использованием LLM"""
    messages = [
        {"role": "system", "text": "Ты помощник, отвечающий на вопросы на основе контекста."},
        {"role": "user", "text": f"Контекст: {context_text}\n\nВопрос: {question}"}
    ]
    
    res = call_yandex_gpt(messages)
    
    langfuse_context.update_current_observation(
        input=messages,
        output=res["text"],
        usage={
            "input": res["input_tokens"],
            "output": res["output_tokens"],
            "total": res["total_tokens"]
        },
        model="yandexgpt",
        metadata={"provider": "yandex_api_key"}
    )
    
    return res["text"]

@observe(as_type="span")
def retrieval_step(question: str, top_k=2) -> List[Dict]:
    """Поиск релевантных документов"""
    keywords = question.lower().split()
    results = []
    
    for doc in KNOWLEDGE_BASE:
        score = 0
        for word in keywords:
            if word in doc["content"].lower():
                score += 1
        if score > 0:
            results.append({**doc, "score": score})
            
    results.sort(key=lambda x: x["score"], reverse=True)
    final_results = results[:top_k]
    
    langfuse_context.update_current_observation(
        input=question, 
        output=final_results
    )
    
    return final_results

@observe()
def rag_pipeline(user_query: str, user_id: str):
    """Основной RAG пайплайн"""
    langfuse_context.update_current_trace(
        user_id=user_id, 
        tags=["production", "rag_demo"]
    )
    
    docs = retrieval_step(user_query)
    
    context_str = "\n".join([d["content"] for d in docs])
    answer = llm_generation_step(context_str, user_query)
    
    return {
        "answer": answer, 
        "retrieved_ids": [d["id"] for d in docs]
    }, langfuse_context.get_current_trace_id()

DATASET_NAME = "My_RAG_Eval_Dataset_v1"

def upload_dataset():
    """Создание и загрузка датасета для оценки"""
    print(f"Creating dataset '{DATASET_NAME}'...")
    
    try:
        langfuse.create_dataset(name=DATASET_NAME)
        print(f"Dataset '{DATASET_NAME}' created.")
    except Exception as e:
        print(f"Dataset probably exists: {e}")

    items = [
        {
            "input": "Что такое Langfuse?", 
            "expected_output": "Платформа для инжиниринга LLM.", 
            "metadata": {"expected_ids": ["doc1"]}
        },
        {
            "input": "Кто создал Yandex GPT?", 
            "expected_output": "Яндекс.", 
            "metadata": {"expected_ids": ["doc2"]}
        },
        {
            "input": "Что такое RAG?", 
            "expected_output": "Использование внешних данных.", 
            "metadata": {"expected_ids": ["doc3"]}
        }
    ]

    for item in items:
        try:
            langfuse.create_dataset_item(
                dataset_name=DATASET_NAME,
                input=item["input"],
                expected_output=item["expected_output"],
                metadata=item["metadata"]
            )
            print(f"Added item: {item['input'][:30]}...")
        except Exception as e:
            print(f"Item already exists or error: {e}")
    
    print("Dataset upload completed.")

def calculate_rag_metrics(retrieved_ids: List[str], expected_ids: List[str], k=2):
    """Расчет метрик качества retrieval"""
    retrieved_at_k = retrieved_ids[:k]
    intersection = set(retrieved_at_k) & set(expected_ids)
    
    precision = len(intersection) / len(retrieved_at_k) if retrieved_at_k else 0
    recall = len(intersection) / len(expected_ids) if expected_ids else 0
    
    mrr = 0
    for i, doc_id in enumerate(retrieved_at_k):
        if doc_id in expected_ids:
            mrr = 1 / (i + 1)
            break
            
    return {
        "precision@k": precision, 
        "recall@k": recall, 
        "mrr": mrr
    }

def run_experiment():
    """Запуск эксперимента на датасете"""
    print(f"\n>>> Running experiment on '{DATASET_NAME}'...")
    
    try:
        dataset = langfuse.get_dataset(DATASET_NAME)
    except Exception as e:
        print(f"Error getting dataset: {e}")
        return
    
    run_name = f"experiment_yandex_{int(time.time())}"
    
    for idx, item in enumerate(dataset.items):
        print(f"\nProcessing item {idx + 1}/{len(dataset.items)}: {item.input[:50]}...")
        
        try:
            output, trace_id = rag_pipeline(item.input, user_id="eval_bot")
            
            expected_ids = item.metadata.get("expected_ids", [])
            retrieved_ids = output["retrieved_ids"]
            metrics = calculate_rag_metrics(retrieved_ids, expected_ids, k=2)
            
            print(f"  Retrieved: {retrieved_ids}")
            print(f"  Expected: {expected_ids}")
            print(f"  Metrics: {metrics}")
            
            item.link(trace_id=trace_id, run_name=run_name, trace_or_observation=None)
            
            for name, value in metrics.items():
                langfuse.score(
                    trace_id=trace_id, 
                    name=name, 
                    value=value
                )
                
        except Exception as e:
            print(f"  Error processing item: {e}")
            continue
    
    langfuse.flush()
    print(f"\n✓ Experiment '{run_name}' completed!")


if __name__ == "__main__":
    print("=" * 60)
    print("RAG System with Yandex GPT + Langfuse")
    print("=" * 60)
    
    print("\n>>> Testing Single Interaction...")
    try:
        result, _ = rag_pipeline("Что известно про Langfuse?", user_id="test_user")
        print(f"\n✓ Answer: {result['answer']}")
        print(f"✓ Retrieved docs: {result['retrieved_ids']}")
    except Exception as e:
        print(f"✗ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    
    langfuse.flush()
    time.sleep(2)
    
    print("\n" + "=" * 60)
    upload_dataset()
    time.sleep(2)
    
    print("\n" + "=" * 60)
    run_experiment()
    
    print("\n" + "=" * 60)
    print("✓ All operations completed!")
    print("=" * 60)
