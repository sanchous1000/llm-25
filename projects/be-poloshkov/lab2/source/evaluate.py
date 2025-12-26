import json
import argparse
from pathlib import Path
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from openai import OpenAI

from config import load_config, Config


EVAL_QUESTIONS = [
    {
        "question": "What is the difference between var, let, and const in JavaScript?",
        "relevant_books": ["scope-closures"],
        "keywords": ["var", "let", "const", "hoisting", "block scope"]
    },
    {
        "question": "How does closure work in JavaScript?",
        "relevant_books": ["scope-closures"],
        "keywords": ["closure", "lexical scope", "function"]
    },
    {
        "question": "What is hoisting in JavaScript?",
        "relevant_books": ["scope-closures"],
        "keywords": ["hoisting", "variable", "declaration"]
    },
    {
        "question": "What are the different types in JavaScript?",
        "relevant_books": ["types-grammar", "get-started"],
        "keywords": ["type", "string", "number", "boolean", "object"]
    },
    {
        "question": "How does prototypal inheritance work?",
        "relevant_books": ["objects-classes"],
        "keywords": ["prototype", "inheritance", "object"]
    },
    {
        "question": "What is the this keyword in JavaScript?",
        "relevant_books": ["objects-classes", "scope-closures"],
        "keywords": ["this", "binding", "context"]
    },
    {
        "question": "How do JavaScript classes work under the hood?",
        "relevant_books": ["objects-classes"],
        "keywords": ["class", "prototype", "constructor"]
    },
    {
        "question": "What is lexical scope?",
        "relevant_books": ["scope-closures"],
        "keywords": ["lexical", "scope", "nested"]
    },
    {
        "question": "How does type coercion work in JavaScript?",
        "relevant_books": ["types-grammar"],
        "keywords": ["coercion", "type", "conversion"]
    },
    {
        "question": "What is the difference between == and === in JavaScript?",
        "relevant_books": ["types-grammar", "get-started"],
        "keywords": ["equality", "strict", "coercion"]
    },
    {
        "question": "How do arrow functions differ from regular functions?",
        "relevant_books": ["scope-closures", "get-started"],
        "keywords": ["arrow", "function", "this", "lexical"]
    },
    {
        "question": "What are iterators and generators in JavaScript?",
        "relevant_books": ["get-started", "objects-classes"],
        "keywords": ["iterator", "generator", "yield"]
    },
    {
        "question": "How does the module system work in JavaScript?",
        "relevant_books": ["scope-closures", "get-started"],
        "keywords": ["module", "import", "export"]
    },
    {
        "question": "What is the temporal dead zone?",
        "relevant_books": ["scope-closures"],
        "keywords": ["temporal", "dead", "zone", "TDZ", "let", "const"]
    },
    {
        "question": "How do you create private properties in JavaScript classes?",
        "relevant_books": ["objects-classes"],
        "keywords": ["private", "class", "property", "#"]
    },
]


@dataclass
class RetrievalResult:
    question: str
    retrieved_chunks: list[dict]
    relevant_books: list[str]
    keywords: list[str]
    precision_at_k: float
    recall_at_k: float
    mrr: float
    keyword_hits: int


def search_chunks(client: QdrantClient, model: SentenceTransformer, 
                  collection_name: str, query: str, top_k: int) -> list[dict]:
    query_embedding = model.encode(query).tolist()
    
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k,
        with_payload=True
    )
    
    return [
        {
            "score": r.score,
            "text": r.payload["text"],
            "book": r.payload["book"],
            "file": r.payload["file"],
            "section": r.payload["section"],
        }
        for r in results.points
    ]


def generate_answer(llm_client, model: str, question: str, chunks: list[dict], temperature: float = 0.3) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = f"[{chunk['book']}: {chunk['section']}]"
        context_parts.append(f"Source {i} {source}:\n{chunk['text']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    system_prompt = """You are a helpful assistant that answers questions about JavaScript based on the "You Don't Know JS" book series.
Answer only based on the provided context. Cite sources using [Book: section] format. Be concise."""
    
    user_prompt = f"""Context:

{context}

---

Question: {question}

Answer:"""
    
    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )
    
    return response.choices[0].message.content


def calculate_metrics(retrieved: list[dict], relevant_books: list[str], 
                      keywords: list[str], k: int) -> dict:
    # relevance by book match
    relevant_count = sum(1 for r in retrieved[:k] if r["book"] in relevant_books)
    
    precision_at_k = relevant_count / k if k > 0 else 0
    
    # recall - how many relevant books we found
    found_books = set(r["book"] for r in retrieved[:k])
    recall_at_k = len(found_books & set(relevant_books)) / len(relevant_books) if relevant_books else 0
    
    # MRR - reciprocal rank of first relevant result
    mrr = 0
    for i, r in enumerate(retrieved[:k]):
        if r["book"] in relevant_books:
            mrr = 1 / (i + 1)
            break
    
    # keyword hits
    all_text = " ".join(r["text"].lower() for r in retrieved[:k])
    keyword_hits = sum(1 for kw in keywords if kw.lower() in all_text)
    
    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "keyword_hits": keyword_hits,
        "keyword_total": len(keywords)
    }


def evaluate_retrieval(config: Config = None, k_values: list[int] = None, generate_answers: bool = True):
    if config is None:
        config = load_config()
    
    if k_values is None:
        k_values = [5, 10]
    
    client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)
    model = SentenceTransformer(config.embedding_model)
    
    llm_client = None
    if generate_answers:
        llm_client = OpenAI(base_url=config.llm_base_url, api_key="ollama")
    
    results = {k: [] for k in k_values}
    
    print(f"Evaluating {len(EVAL_QUESTIONS)} questions...")
    
    for i, q in enumerate(EVAL_QUESTIONS):
        print(f"  [{i+1}/{len(EVAL_QUESTIONS)}] {q['question'][:50]}...")
        
        retrieved = search_chunks(
            client, model, config.collection_name, 
            q["question"], max(k_values)
        )
        
        for k in k_values:
            metrics = calculate_metrics(
                retrieved, q["relevant_books"], q["keywords"], k
            )
            
            result_entry = {
                "question": q["question"],
                "metrics": metrics,
                "top_results": [
                    {"book": r["book"], "section": r["section"], "score": r["score"]}
                    for r in retrieved[:k]
                ]
            }
            
            if generate_answers and llm_client and k == min(k_values):
                answer = generate_answer(
                    llm_client, config.llm_model, 
                    q["question"], retrieved[:k], 
                    config.llm_temperature
                )
                result_entry["generated_answer"] = answer
            
            results[k].append(result_entry)
    
    summary = {}
    for k in k_values:
        precisions = [r["metrics"]["precision_at_k"] for r in results[k]]
        recalls = [r["metrics"]["recall_at_k"] for r in results[k]]
        mrrs = [r["metrics"]["mrr"] for r in results[k]]
        keyword_hits = [r["metrics"]["keyword_hits"] for r in results[k]]
        keyword_totals = [r["metrics"]["keyword_total"] for r in results[k]]
        
        summary[f"k={k}"] = {
            "precision_at_k": sum(precisions) / len(precisions),
            "recall_at_k": sum(recalls) / len(recalls),
            "mrr": sum(mrrs) / len(mrrs),
            "keyword_coverage": sum(keyword_hits) / sum(keyword_totals) if sum(keyword_totals) > 0 else 0,
        }
    
    return results, summary


def print_evaluation_report(summary: dict, config: Config):
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION REPORT")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  Chunk size: {config.chunk_size} tokens")
    print(f"  Overlap: {config.chunk_overlap} tokens")
    print(f"  Splitter: {config.splitter_type}")
    print(f"  Embedding: {config.embedding_model}")
    
    print("\nMetrics:")
    for k_label, metrics in summary.items():
        print(f"\n  {k_label}:")
        print(f"    Precision@k:      {metrics['precision_at_k']:.3f}")
        print(f"    Recall@k:         {metrics['recall_at_k']:.3f}")
        print(f"    MRR:              {metrics['mrr']:.3f}")
        print(f"    Keyword coverage: {metrics['keyword_coverage']:.3f}")


def save_evaluation_results(results: dict, summary: dict, config: Config):
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    report = {
        "config": {
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "splitter_type": config.splitter_type,
            "embedding_model": config.embedding_model,
            "llm_model": config.llm_model,
        },
        "summary": summary,
        "detailed_results": results
    }
    
    output_file = output_dir / f"eval_{config.chunk_size}_{config.splitter_type}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10], help="k values for metrics")
    
    args = parser.parse_args()
    
    config = load_config()
    results, summary = evaluate_retrieval(config, args.k)
    print_evaluation_report(summary, config)
    save_evaluation_results(results, summary, config)

