#!/usr/bin/env python3
import argparse
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

import os
import sys

import langfuse
from langfuse import Langfuse

sys.path.append("../lab2/scripts")

import uuid

from utils.ollama import embed_query, generate_answer
from utils.qdrant import QdrantCollection


def build_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, c in enumerate(contexts, 1):
        heading = c.payload.get("heading", "")
        level = c.payload.get("level", "")
        text = c.payload.get("text", "")
        file_path = c.payload.get("file_path", "")
        blocks.append(f"[{i}] heading={heading} (h{level}) file={file_path}\n{text}")
    context_str = "\n\n".join(blocks)
    instr = (
        "You are an assistant. Answer strictly based on the contexts below.\n"
        "If there isn't enough information, say so. At the end, cite sources like [1], [2]."
    )
    return f"{instr}\nQuestion: {question}\n\nContexts:\n{context_str}\n\nAnswer:"

def main():
    ap = argparse.ArgumentParser(description="RAG запрос к LLM (Ollama) с контекстом из Qdrant")
    ap.add_argument("--question", required=True)
    ap.add_argument("--embed_model", default="nomic-embed-text", help="Модель эмбеддингов в Ollama")
    ap.add_argument("--llm_model", default="llama3.2", help="LLM модель в Ollama")
    ap.add_argument("--collection", default="vllm_docs")
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    langfuseClient = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"), 
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        base_url=os.getenv("LANGFUSE_BASE_URL"),
    )
    
    with langfuse.propagate_attributes(
        user_id="avyuga",
        session_id=str(uuid.uuid4()),
        metadata={"environment": "test"}
    ):
        trace_id = langfuseClient.create_trace_id()
        
        with langfuseClient.start_as_current_observation(
            as_type="generation", 
            name="embed_query",
            trace_context={"trace_id": trace_id}
        ) as gen:
            # 1) Эмбеддинг запроса
            q_vec = embed_query(args.question, model=args.embed_model, host="http://localhost:11434")
            gen.update(
                model=args.embed_model,
                input={"query": args.question},
                output={"embedding": q_vec},
            )

            langfuseClient.flush()

        with langfuseClient.start_as_current_observation(
            as_type="span", 
            name="search_results",
            trace_context={"trace_id": trace_id}
        ) as span:
            # 2) Поиск в Qdrant
            qdrant = QdrantCollection(
                name=args.collection, 
                host=os.getenv("QDRANT_HOST", default="localhost"), 
                port=os.getenv("QDRANT_PORT", default=6333)
            )
            results = qdrant.search(q_vec, top_k=args.top_k)
            span.update(
                input={"embedding": q_vec},
                output={"rag_results":  results}
            )
            langfuseClient.flush()

        with langfuseClient.start_as_current_observation(
            as_type="span", 
            name="create_prompt",
            trace_context={"trace_id": trace_id}
        ) as span:
            # 3) Сборка промпта
            prompt = build_prompt(args.question, results)
            span.update(
                input={"question": args.question, "rag_results": results},
                output={"updated_prompt":  prompt}
            )
            langfuseClient.flush()

        with langfuseClient.start_as_current_observation(
            as_type="generation", 
            name="generate_LLM",
            trace_context={"trace_id": trace_id}
        ) as gen:
             # 4) Вызов LLM в Ollama
            answer = generate_answer(prompt, model=args.llm_model, host="http://localhost:11434")
            gen.update(
                input={"prompt": prompt, "model": args.llm_model},
                output={"answer":  answer}
            )
            langfuseClient.flush()

   

    # 5) Печать результата
    print("=== Answer ===")
    print(answer)
    print("\n=== Quotes ===")
    for i, r in enumerate(results, 1):
        heading = r.payload.get("heading", "")
        file_path = r.payload.get("file_path", "")
        print(f"[{i}] heading={heading} file={file_path}")

if __name__ == "__main__":
    main()