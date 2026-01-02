from langfuse import Langfuse
import yaml
import os
from dotenv import load_dotenv

load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

DATASET_NAME = "dnd5e_evaluation"
QUERIES_PATH = "evaluation_queries.yaml"

with open(QUERIES_PATH) as f:
    data = yaml.safe_load(f)

langfuse.create_dataset(name=DATASET_NAME)
print(f"Created dataset: {DATASET_NAME}")

for query_item in data["queries"]:
    langfuse.create_dataset_item(
        dataset_name=DATASET_NAME,
        input={"query": query_item["query"]},
        expected_output={"relevant_docs": query_item["relevant_docs"]},
        metadata={
            "query_id": query_item["id"],
            "query": query_item["query"],
            "relevant_docs": query_item["relevant_docs"]
        }
    )
    print(f"  Added item {query_item['id']}: {query_item['query']}")

langfuse.flush()
print(f"\n[OK] Uploaded {len(data['queries'])} items to '{DATASET_NAME}'")
