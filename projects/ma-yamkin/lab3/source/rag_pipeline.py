from query_rag import query_rag
from query_llm import query_model_with_metrics
from langfuse import Langfuse
from dotenv import load_dotenv
import os
import uuid

load_dotenv()


def run_pipeline(question, langfuse, name, trace_id=None):
    context_block, chunk_arxiv_ids, indices = query_rag(question, langfuse, name, trace_id)

    prompt = f"""
        Ответь на вопрос, используя ТОЛЬКО приведённую ниже информацию.
        Не выдумывай. Если в тексте нет ответа — напиши "Информация отсутствует".
        
        Вопрос: {question}
        
        Контекст:
        {context_block}
        
        Ответ:
    """

    answer = query_model_with_metrics("qwen2.5:3b", prompt, langfuse, trace_id)

    langfuse.flush()

    return answer, chunk_arxiv_ids, indices


if __name__ == "__main__":
    question = input("Введите вопрос: ").strip()

    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST")
    )

    trace_id = langfuse.create_trace_id(seed=str(uuid.uuid4()))

    run_pipeline(question, langfuse, 'index_5790b8cf', trace_id)
