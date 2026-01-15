import json
from pathlib import Path
from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()
langfuse = get_client()
print("HERE")
DATASET_NAME = "fastapi_rag_eval"

def main():
    if not langfuse.auth_check():
        raise RuntimeError("Langfuse auth_check() failed")
    print(Path("data/rag_eval.json"))
    items = json.loads(Path("llm/data/eval/questions.json").read_text(encoding="utf-8"))
    print(items)
    # create_dataset может быть неидемпотентным (если уже существует) — ловим ошибку грубо
    try:
        langfuse.create_dataset(name=DATASET_NAME)
    except Exception:
        pass

    for it in items:
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            input=it["question"],
            expected_output=it.get("gold_paths"),
        )

    langfuse.flush()
    print(f"Uploaded {len(items)} items to dataset={DATASET_NAME}")

if __name__ == "__main__":
    main()
