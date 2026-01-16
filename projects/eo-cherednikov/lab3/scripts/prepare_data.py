import argparse
import json
import warnings

from dotenv import load_dotenv

load_dotenv()

import os

from langfuse import Langfuse
from qdrant_client import QdrantClient

warnings.filterwarnings("ignore")

def get_doc_id_to_path_mapping(qdrant_host: str, qdrant_port: int, collection: str):
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    all_docs = client.scroll(collection_name=collection, limit=10000)[0]

    mapping = {}
    for doc in all_docs:
        doc_id = doc.payload.get("id", "")
        file_path = doc.payload.get("file_path", "")
        page_number = doc.payload.get("page_number")
        if doc_id and file_path:
            mapping[doc_id] = {
                "file_path": file_path,
                "page_number": page_number
            }

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

    path_to_docs = {}
    for doc_id, info in id_to_path_mapping.items():
        file_path = info["file_path"].replace('\\', '/')  # Нормализуем путь
        page_number = info.get("page_number")
        if file_path not in path_to_docs:
            path_to_docs[file_path] = []
        path_to_docs[file_path].append((doc_id, page_number))

    ground_truth_file = 'data/ground_truth_2.json'
    
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_name = "dnd_2024_questions"
    langfuseClient.create_dataset(name=dataset_name)
    
    for item in data:
        expected_items = []
        relevant_pages = item.get("relevant_pages", {})
        
        for file_path in item["relevant_file_paths"]:
            normalized_path = file_path.replace('\\', '/')
            
            if normalized_path in path_to_docs:
                if normalized_path in relevant_pages:
                    expected_pages = relevant_pages[normalized_path]
                    if not isinstance(expected_pages, list):
                        expected_pages = [expected_pages]

                    for doc_id, page_number in path_to_docs[normalized_path]:
                        if page_number is not None and page_number in expected_pages:
                            expected_items.append(f"{doc_id}:{page_number}")
                        elif page_number is None:
                            expected_items.append(doc_id)
                else:
                    for doc_id, page_number in path_to_docs[normalized_path]:
                        expected_items.append(doc_id)
            else:
                print(f"Warning {file_path} not found in the collection")

        seen = set()
        unique_items = []
        for item_id in expected_items:
            if item_id not in seen:
                seen.add(item_id)
                unique_items.append(item_id)

        langfuseClient.create_dataset_item(
            dataset_name=dataset_name,
            input=item["question"],
            expected_output=unique_items
        )