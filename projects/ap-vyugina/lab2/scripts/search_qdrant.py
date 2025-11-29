#!/usr/bin/env python3
import argparse

from utils.ollama import embed_query
from utils.qdrant import QdrantCollection

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--ollama_host", default="http://localhost:11434")
    ap.add_argument("--embed_model", default="nomic-embed-text")
    ap.add_argument("--qdrant_host", default="localhost")
    ap.add_argument("--qdrant_port", type=int, default=6333)
    ap.add_argument("--collection", default="vllm_docs")
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    qv = embed_query(args.query, args.embed_model, args.ollama_host)

    qdrant = QdrantCollection(name=args.collection, host=args.qdrant_host, port=args.qdrant_port)
    res = qdrant.search(query_vector=qv, top_k=args.top_k)

    for i, r in enumerate(res, 1):
        file_path = r.payload.get('file_path', '')
        print(f"{i}. score={r.score:.4f} heading={r.payload.get('heading')} file={file_path}")
        print(f"   text={r.payload.get('text').strip()}")

if __name__ == "__main__":
    main()