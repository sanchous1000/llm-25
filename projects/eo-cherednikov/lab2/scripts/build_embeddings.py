import argparse
import json
from typing import Any, Dict, List

from utils.ollama import embed_texts
from utils.qdrant import QdrantCollection


def load_jsonl(p: str) -> List[Dict[str, Any]]:
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--ollama_host", default="http://localhost:11434")
    ap.add_argument("--embed_model", default="nomic-embed-text")
    ap.add_argument("--qdrant_host", default="localhost")
    ap.add_argument("--qdrant_port", type=int, default=6333)
    ap.add_argument("--collection", default="vllm_docs")
    ap.add_argument("--distance", default="cosine", choices=["cosine","dot","euclidean"])
    ap.add_argument("--recreate", action="store_true")
    args = ap.parse_args()

    records = load_jsonl(args.input_jsonl)
    texts = [ f"{r['file_path']} | {r['heading']}\n{r['text']}" for r in records]

    vectors = embed_texts(texts, model=args.embed_model, host=args.ollama_host)
    vec_size = len(vectors[0])

    qdrant = QdrantCollection(name=args.collection, host=args.qdrant_host, port=args.qdrant_port)
    qdrant.ensure_exists(vec_size=vec_size, distance=args.distance, recreate=args.recreate)
    qdrant.upload(records, vectors)

    print(f"Uploaded {len(records)} points to '{args.collection}' (dim={vec_size})")

if __name__ == "__main__":
    main()