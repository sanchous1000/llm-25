from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import IndexConfig
import requests
import os

def call_llm(prompt: str):
    headers = {
        "Authorization": f"Api-Key {os.getenv('YANDEXGPT_APIKEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "modelUri": f"gpt://{os.getenv('YANDEX_FOLDERID')}/yandexgpt-lite",
        "completionOptions": {
            "stream": False,
            "temperature": float(os.getenv("MODEL_TEMP"))
        },
        "messages": [
            {
                "role": "user",
                "text": prompt
            }
        ]
    }
    answer = requests.post(
        "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        headers=headers,
        json=data
    )
    return answer

class RAGService:
    def __init__(self, config: IndexConfig):
        self.cfg = config
        self.client = QdrantClient(path=self.cfg.storage_path)
        self.embeddings = HuggingFaceEmbeddings(model_name=self.cfg.embedding_model)

    def retrieve(self, query: str, top_k=3):
        query_vec = self.embeddings.embed_query(f"query: {query}")
        results = self.client.search(
            collection_name=self.cfg.collection_name,
            query_vector=query_vec,
            limit=top_k
        )
        return results

    def answer_question(self, query: str):
        # 1. Поиск
        chunks = self.retrieve(query)
        
        # 2. Сборка промпта
        context_str = "\n\n".join([
            f"Source ({c.payload['source']}): {c.payload['text']}" 
            for c in chunks
        ])
        
        system_prompt = "Ты полезный помощник. Отвечай только на основе контекста ниже."
        user_prompt = f"""
        Контекст:
        {context_str}
        
        Вопрос: {query}
        
        Ответ должен содержать ссылки на источники.
        """
        
        print("--- LLM PROMPT ---")
        print(user_prompt)
        print("------------------")
        
        response = call_llm(f"{system_prompt}\n{user_prompt}")
        result = response.json()
        return result["result"]["alternatives"][0]["message"]["text"]

if __name__ == "__main__":
    rag = RAGService(IndexConfig())
    ans = rag.answer_question("Как работает испытательный срок?")
    print("Answer:", ans)
