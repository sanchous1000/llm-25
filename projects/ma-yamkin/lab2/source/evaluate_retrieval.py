import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from utils import load_config, get_config_hash


def evaluate_faiss():
    config = load_config()
    config_hash = get_config_hash(config)
    artifacts_dir = Path(f"../artifacts/index_{config_hash}")
    eval_dir = Path(f"../eval")

    with open(f"../eval/questions.json", encoding="utf-8") as f:
        questions = json.load(f)
    with open(artifacts_dir / "metadata.json", encoding="utf-8") as f:
        chunks = json.load(f)
    index = faiss.read_index(str(artifacts_dir / "faiss.index"))
    embedder = SentenceTransformer(config["embedding"]["model"])

    chunk_arxiv_ids = [chunk["id"] for chunk in chunks]

    top_k = config["retrieval"]["top_k"]
    recalls = []
    precisions = []
    mrrs = []

    for q in questions:
        question = q["question"]
        relevant_ids = set(q["relevant_chunks"])

        query_vec = embedder.encode(question)
        faiss.normalize_L2(query_vec.reshape(1, -1))
        distances, indices = index.search(query_vec.reshape(1, -1).astype(np.float32), k=top_k)

        retrieved_ids = [chunk_arxiv_ids[i] for i in indices[0]]

        relevant_retrieved = set([rid for rid in retrieved_ids if rid in relevant_ids])
        num_relevant_total = len(relevant_ids)
        num_retrieved = len(retrieved_ids)
        num_relevant_retrieved = len(relevant_retrieved)

        # Recall@k
        recall = num_relevant_retrieved / num_relevant_total if num_relevant_total > 0 else 0
        recalls.append(recall)

        # Precision@k
        precision = num_relevant_retrieved / num_retrieved if num_retrieved > 0 else 0
        precisions.append(precision)

        # MRR
        mrr = 0.0
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in relevant_ids:
                mrr = 1.0 / rank
                break
        mrrs.append(mrr)

        print(f"Вопрос: {question[:50]}...")
        print(f"  Recall@{top_k}: {recall:.3f}, Precision@{top_k}: {precision:.3f}, MRR: {mrr:.3f}")

        # Итоговые метрики
        avg_recall = np.mean(recalls)
        avg_precision = np.mean(precisions)
        avg_mrr = np.mean(mrrs)

        print("\n" + "=" * 50)
        print(f"Средние метрики (k={top_k}):")
        print(f"  Recall@{top_k}:   {avg_recall:.3f}")
        print(f"  Precision@{top_k}: {avg_precision:.3f}")
        print(f"  MRR:              {avg_mrr:.3f}")

        # Сохранение метрик в ту же папку, что и вопросы
        metrics = {
            "config_hash": config_hash,
            "top_k": top_k,
            "recall": float(avg_recall),
            "precision": float(avg_precision),
            "mrr": float(avg_mrr),
            "per_query": [
                {
                    "question": q["question"],
                    "relevant_chunks": q["relevant_chunks"],
                    "recall": r,
                    "precision": p,
                    "mrr": m
                }
                for q, r, p, m in zip(questions, recalls, precisions, mrrs)
            ]
        }

        metrics_path = eval_dir / f"metrics_{config_hash}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print(f"\nМетрики сохранены в: {metrics_path}")


if __name__ == "__main__":
    evaluate_faiss()