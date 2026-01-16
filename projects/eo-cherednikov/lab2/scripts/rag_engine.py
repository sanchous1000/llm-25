import argparse
from typing import Any, Dict, List

from utils.ollama import embed_query, generate_answer
from utils.qdrant import QdrantCollection


def build_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, c in enumerate(contexts, 1):
        heading = c.payload.get("heading", "")
        level = c.payload.get("level", "")
        text = c.payload.get("text", "")
        file_path = c.payload.get("file_path", "")
        page_number = c.payload.get("page_number")
        
        # Формируем строку источника с учетом типа документа
        source_info = f"file={file_path}"
        if page_number is not None:
            source_info += f" page={page_number}"
        if level:
            source_info += f" (h{level})"
        
        blocks.append(f"[{i}] heading={heading} {source_info}\n{text}")
    context_str = "\n\n".join(blocks)
    instr = (
        "You are an assistant. Answer strictly based on the contexts below.\n"
        "If there isn't enough information, say so. At the end, cite sources like [1], [2]."
    )
    return f"{instr}\nQuestion: {question}\n\nContexts:\n{context_str}\n\nAnswer:"

def main():
    ap = argparse.ArgumentParser(description="RAG запрос к LLM (Ollama) с контекстом из Qdrant")
    ap.add_argument("--question", required=True)
    ap.add_argument("--ollama_host", default="http://localhost:11434")
    ap.add_argument("--embed_model", default="nomic-embed-text", help="Модель эмбеддингов в Ollama")
    ap.add_argument("--llm_model", default="llama3.2", help="LLM модель в Ollama")
    ap.add_argument("--qdrant_host", default="localhost")
    ap.add_argument("--qdrant_port", type=int, default=6333)
    ap.add_argument("--collection", default="vllm_docs")
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    q_vec = embed_query(args.question, model=args.embed_model, host=args.ollama_host)

    qdrant = QdrantCollection(name=args.collection, host=args.qdrant_host, port=args.qdrant_port)
    results = qdrant.search(query_vector=q_vec, top_k=args.top_k)

    prompt = build_prompt(args.question, results)

    answer = generate_answer(prompt, model=args.llm_model, host=args.ollama_host)

    print("=== Answer ===")
    print(answer)
    print("\n=== Quotes ===")
    for i, r in enumerate(results, 1):
        heading = r.payload.get("heading", "")
        file_path = r.payload.get("file_path", "")
        page_number = r.payload.get("page_number")
        
        source_info = f"file={file_path}"
        if page_number is not None:
            source_info += f" page={page_number}"
        
        print(f"[{i}] heading={heading} {source_info}")

if __name__ == "__main__":
    main()