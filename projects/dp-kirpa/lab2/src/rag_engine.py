from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import IndexConfig

# Заглушка для LLM из Лабораторной 1
def call_llm_lab1(prompt):
    # import openai
    # return openai.ChatCompletion.create(...)
    return f"Simulated LLM Answer based on prompt length: {len(prompt)}"

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
        
        # 3. Генерация (имитация вызова)
        print("--- LLM PROMPT ---")
        print(user_prompt)
        print("------------------")
        
        response = call_llm_lab1(f"{system_prompt}\n{user_prompt}")
        return response

if __name__ == "__main__":
    rag = RAGService(IndexConfig())
    ans = rag.answer_question("Как работает испытательный срок?")
    print("Answer:", ans)
