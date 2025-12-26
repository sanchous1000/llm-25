from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

import os
import sys

sys.path.append("../lab2/scripts")

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

def task(item,
        embed_model: str = "nomic-embed-text",
        ollama_host: str = "http://localhost:11434",
        top_k: int = 10):
    question = item.input # `run_experiment` passes a `DatasetItemClient` to the task function. The input of the dataset item is available as `item.input`.

    query_vec = embed_query(question, model=embed_model, host=ollama_host)
    
    results = qdrant.search(query_vector=query_vec, top_k=top_k)
    retrieved_ids = [r.payload.get("id", "") for r in results]
 
    return retrieved_ids
 
dataset = langfuse.get_dataset("vllm_questions")
result = dataset.run_experiment(
    name="RAG test",
    task=task
)
 
print(result.format())