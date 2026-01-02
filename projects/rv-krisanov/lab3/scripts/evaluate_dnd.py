from langfuse import Langfuse, Evaluation
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import yaml

load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

with open("scripts/configs/config_baseline.yaml") as f:
    config = yaml.safe_load(f)["agent"]

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
embedding_model = SentenceTransformer(config["embedding_model"])
collection_name = config["collection_name"]

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "gpt-5-nano"),
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

DATASET_NAME = "dnd5e_evaluation"
K_VALUES = [5, 10]


def task(*, item, **kwargs):
    query = item.input["query"]
    
    query_embedding = embedding_model.encode(query)
    
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        with_payload=True,
        limit=max(K_VALUES)
    ).points
    
    retrieved_chunks = []
    for point in results:
        retrieved_chunks.append({
            "document": point.payload["metadata"]["document"]["source"],
            "md_header": point.payload["metadata"]["md_header"],
            "text": point.payload["text"]
        })
    
    context = "\n\n".join([f"[{c['document']}]\n{c['text']}" for c in retrieved_chunks])
    prompt = f"{config['system_prompt']}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    
    response = llm.invoke(prompt)
    
    return {
        "answer": response.content,
        "retrieved_chunks": retrieved_chunks
    }


def match_chunk(retrieved_chunk, expected_doc):
    retrieved_doc = retrieved_chunk["document"]
    expected_doc_name = expected_doc["document"]
    
    if not retrieved_doc.endswith(expected_doc_name):
        return False
    
    retrieved_headers = retrieved_chunk["md_header"]
    
    for expected_header in expected_doc["md_headers"]:
        match = all(
            retrieved_headers.get(k) == v 
            for k, v in expected_header.items()
        )
        if match:
            return True
    return False


def recall_evaluator(*, input, output, expected_output, **kwargs):
    retrieved = output["retrieved_chunks"]
    expected_docs = expected_output["relevant_docs"]
    
    evals = []
    for k in K_VALUES:
        retrieved_k = retrieved[:k]
        matched_docs = set()
        for r in retrieved_k:
            for i, e in enumerate(expected_docs):
                if match_chunk(r, e):
                    matched_docs.add(i)
        
        recall = len(matched_docs) / len(expected_docs) if expected_docs else 0.0
        evals.append(Evaluation(
            name=f"recall@{k}",
            value=recall,
            comment=f"Retrieved {len(matched_docs)}/{len(expected_docs)} relevant docs in top-{k}"
        ))
    return evals


def precision_evaluator(*, input, output, expected_output, **kwargs):
    retrieved = output["retrieved_chunks"]
    expected_docs = expected_output["relevant_docs"]
    
    evals = []
    for k in K_VALUES:
        retrieved_k = retrieved[:k]
        relevant_count = sum(
            1 for r in retrieved_k 
            if any(match_chunk(r, e) for e in expected_docs)
        )
        precision = relevant_count / k if k > 0 else 0.0
        evals.append(Evaluation(
            name=f"precision@{k}",
            value=precision,
            comment=f"{relevant_count}/{k} retrieved docs were relevant"
        ))
    return evals


def mrr_evaluator(*, input, output, expected_output, **kwargs):
    retrieved = output["retrieved_chunks"]
    expected_docs = expected_output["relevant_docs"]
    
    for rank, chunk in enumerate(retrieved, 1):
        if any(match_chunk(chunk, e) for e in expected_docs):
            rr = 1.0 / rank
            return [Evaluation(
                name="MRR",
                value=rr,
                comment=f"First relevant doc at rank {rank}"
            )]
    
    return [Evaluation(
        name="MRR",
        value=0.0,
        comment="No relevant docs found"
    )]


dataset = langfuse.get_dataset(DATASET_NAME)

result = langfuse.run_experiment(
    name="dnd5e_rag_baseline",
    description="RAG evaluation on D&D 5e queries with retrieval metrics",
    data=dataset.items,
    task=task,
    evaluators=[recall_evaluator, precision_evaluator, mrr_evaluator],
    max_concurrency=1
)

print(f"\n[OK] Experiment completed")
print(f"Processed: {len(result.item_results)} items")
print(f"View: {result.dataset_run_url}")
