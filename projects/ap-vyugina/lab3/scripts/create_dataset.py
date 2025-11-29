import argparse
import json

from dotenv import load_dotenv

load_dotenv()

import os

from langfuse import Langfuse
from qdrant_client import QdrantClient


def get_doc_id_to_path_mapping(qdrant_host: str, qdrant_port: int, collection: str):
    """Получение соответствия между ID документа и путем к файлу"""
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    
    # Получаем все документы из коллекции
    all_docs = client.scroll(collection_name=collection, limit=10000)[0]
    
    mapping = {}
    for doc in all_docs:
        doc_id = doc.payload.get("id", "")
        file_path = doc.payload.get("file_path", "")
        if doc_id and file_path:
            mapping[doc_id] = file_path
    
    return mapping


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Оценка качества retrieval системы")
    ap.add_argument("--collection", default="vllm_docs")
    args = ap.parse_args()

    langfuseClient = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"), 
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        base_url=os.getenv("LANGFUSE_BASE_URL"),
    )

    id_to_path_mapping = get_doc_id_to_path_mapping(
        qdrant_host=os.getenv("QDRANT_HOST", default="localhost"), 
        qdrant_port=os.getenv("QDRANT_PORT", default=6333), 
        collection=args.collection
    )
    
    path_to_id_mapping = {path: doc_id for doc_id, path in id_to_path_mapping.items()}

    ground_truth_file = '../artifacts/ground_truth_example.json'
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        data = json.load(f)


    dataset_name = "vllm_questions"
    langfuseClient.create_dataset(name=dataset_name)
    for item in data:
        file_path_ids = []
        for file_path in item["relevant_file_paths"]:
            if file_path in path_to_id_mapping:
                file_path_ids.append(path_to_id_mapping[file_path])
            else:
                print(f"Warning {file_path} not found in the collection")
        
        langfuseClient.create_dataset_item(
            dataset_name=dataset_name,
            input=item["question"],
            expected_output=file_path_ids
        )