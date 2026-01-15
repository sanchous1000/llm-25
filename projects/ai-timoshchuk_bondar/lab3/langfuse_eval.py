from __future__ import annotations
from dotenv import load_dotenv
from langfuse import get_client, Evaluation

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


load_dotenv()
langfuse = get_client()

DATASET_NAME = "fastapi_rag_eval"


def retrieval_metrics(
    retrieved_paths: list[str], gold_paths: list[str], k: int
) -> tuple[float, float, float]:
    topk = retrieved_paths[:k]
    gold = set(gold_paths)
    hits = [p for p in topk if p in gold]

    recall = (len(set(hits)) / len(gold)) if gold else 0.0
    precision = (len(set(hits)) / k) if k else 0.0

    # MRR@k
    rr = 0.0
    for rank, p in enumerate(topk, start=1):
        if p in gold:
            rr = 1.0 / rank
            break
    return recall, precision, rr


try:
    import faiss
except ImportError as e:
    raise ImportError("Install faiss-cpu: pip install faiss-cpu") from e

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

RE_WORD = re.compile(r"\w+", re.UNICODE)


def _tokenize_words(text: str) -> List[str]:
    return [w.lower() for w in RE_WORD.findall(text)]


def _load_config(project_root: Path) -> dict:
    cfg_path = project_root / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found at: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _run_id_from_cfg(cfg: dict) -> str:
    blob = yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:10]


def _load_chunks_jsonl(chunks_path: Path) -> List[dict]:
    chunks = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def _load_faiss_index_serialized(npy_path: Path):
    if not npy_path.exists():
        raise FileNotFoundError(f"FAISS serialized index not found: {npy_path}")
    buf = np.load(npy_path, allow_pickle=False)
    return faiss.deserialize_index(buf)


@dataclass
class RAGRuntime:
    project_root: Path
    cfg: dict
    run_id: str
    chunks: List[dict]
    embedder: SentenceTransformer
    faiss_index: Any
    bm25: Any  # BM25Okapi or None


_RUNTIME: Optional[RAGRuntime] = None


def get_runtime(project_root: Optional[Path] = None) -> RAGRuntime:
    """
    Lazy singleton runtime: грузит артефакты 1 раз на процесс.
    """
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME

    load_dotenv()

    if project_root is None:
        # Если файл лежит в rag/runtime.py -> project_root = на 2 уровня вверх
        project_root = Path(__file__).resolve().parents[1]

    cfg = _load_config(project_root)
    run_id = _run_id_from_cfg(cfg)

    art_dir = project_root / "artifacts" / run_id
    chunks_path = art_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"chunks.jsonl not found: {chunks_path} (run_id={run_id})"
        )

    chunks = _load_chunks_jsonl(chunks_path)

    dense_model = cfg["embedding"]["dense_model"]
    embedder = SentenceTransformer(dense_model)

    # FAISS index: используем сериализацию .npy (Windows + unicode path safe)
    faiss_npy = art_dir / "faiss_hnsw.npy"
    faiss_index = _load_faiss_index_serialized(faiss_npy)

    # BM25 (optional)
    bm25 = None
    mode = cfg["embedding"]["mode"]
    if mode in ("sparse", "hybrid"):
        if BM25Okapi is None:
            raise ImportError("Install rank-bm25: pip install rank-bm25")
        bm25_path = art_dir / "bm25.json"
        if not bm25_path.exists():
            raise FileNotFoundError(f"bm25.json not found: {bm25_path} (mode={mode})")
        tokenized = json.loads(bm25_path.read_text(encoding="utf-8"))["tokenized"]
        k1 = float(cfg["embedding"].get("bm25_k1", 1.2))
        b = float(cfg["embedding"].get("bm25_b", 0.75))
        bm25 = BM25Okapi(tokenized, k1=k1, b=b)

    _RUNTIME = RAGRuntime(
        project_root=project_root,
        cfg=cfg,
        run_id=run_id,
        chunks=chunks,
        embedder=embedder,
        faiss_index=faiss_index,
        bm25=bm25,
    )
    return _RUNTIME


def _dense_retrieve(
    rt: RAGRuntime, question: str, top_k: int
) -> Tuple[List[int], List[float]]:
    """
    Возвращает (indices, scores). score = преобразованный similarity (чем больше — тем лучше).
    """
    # E5 требует префикс query:
    qvec = rt.embedder.encode([f"query: {question}"], normalize_embeddings=True).astype(
        "float32"
    )
    D, I = rt.faiss_index.search(qvec, top_k)

    idxs = [int(i) for i in I[0] if int(i) >= 0]
    scores = [float(1.0 - (d * 0.5)) for d in D[0][: len(idxs)]]
    return idxs, scores


def _sparse_retrieve(
    rt: RAGRuntime, question: str, top_k: int
) -> Tuple[List[int], List[float]]:
    """
    BM25 ранжирование по всем чанкам.
    """
    if rt.bm25 is None:
        return [], []
    scores_all = rt.bm25.get_scores(_tokenize_words(question))
    idx = np.argsort(-np.asarray(scores_all))[:top_k]
    idxs = [int(i) for i in idx]
    scores = [float(scores_all[i]) for i in idxs]
    return idxs, scores


def _hybrid_merge(
    dense: Tuple[List[int], List[float]],
    sparse: Tuple[List[int], List[float]],
    alpha: float,
    top_k: int,
) -> List[Tuple[int, float]]:
    """
    Простая гибридизация: нормируем ранги в score.
    """
    dense_idxs, _ = dense
    sparse_idxs, _ = sparse

    score: Dict[int, float] = {}

    # rank-based score (устойчивее, чем сырые шкалы)
    for r, idx in enumerate(dense_idxs, start=1):
        score[idx] = score.get(idx, 0.0) + alpha * (1.0 / r)
    for r, idx in enumerate(sparse_idxs, start=1):
        score[idx] = score.get(idx, 0.0) + (1.0 - alpha) * (1.0 / r)

    merged = sorted(score.items(), key=lambda x: -x[1])[:top_k]
    return merged


def _build_messages(question: str, retrieved: List[dict]) -> List[dict]:
    """
    Важно: не заставляем модель печатать SOURCE и не позволяем ей перечислять контекст.
    Источники выводятся программно (не внутри текста модели).
    """
    system = (
        "Ты — ассистент по документации FastAPI.\n"
        "Отвечай на русском.\n"
        "Используй только предоставленные фрагменты.\n"
        'Если ответа нет — скажи: "В предоставленных фрагментах документации это не найдено.".\n'
        "Не перечисляй источники и не перепечатывай контекст.\n"
        "Ответ дай связным текстом (1–5 абзацев)."
    )

    ctx_lines = []
    for i, r in enumerate(retrieved, start=1):
        # Даем компактный контекст
        ctx_lines.append(f"[{i}] {r['snippet']}")

    user = (
        f"Вопрос: {question}\n\n"
        "Фрагменты документации:\n" + "\n\n".join(ctx_lines) + "\n\nОтвет:"
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def rag_inference_returning_sources(
    question: str,
    *,
    rag_cfg_override: Optional[dict] = None,
) -> Tuple[str, List[dict]]:
    """
    Возвращает:
      answer_text: str
      retrieved: list[dict] где dict содержит:
         - source: doc_path
         - chunk_id
         - score
         - snippet (кусок текста)
    """
    rt = get_runtime()
    cfg = rt.cfg

    # Поддержка override параметров (удобно для экспериментов)
    rag_cfg = {
        "mode": cfg["embedding"]["mode"],
        "top_k": int(cfg["retrieval"]["top_k"]),
        "hybrid_alpha": float(cfg["retrieval"].get("hybrid_alpha", 0.7)),
        "llm_model": os.getenv("OLLAMA_MODEL", "llama3.2:1b-instruct-q5_K_M"),
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 350,
    }
    if rag_cfg_override:
        rag_cfg.update(rag_cfg_override)

    mode = rag_cfg["mode"]
    top_k = int(rag_cfg["top_k"])

    # 1) Retrieval (dense/sparse/hybrid)
    if mode == "dense":
        idxs, scores = _dense_retrieve(rt, question, top_k)
        merged = list(zip(idxs, scores))
    elif mode == "sparse":
        idxs, scores = _sparse_retrieve(rt, question, top_k)
        merged = list(zip(idxs, scores))
    else:
        dense = _dense_retrieve(rt, question, top_k)
        sparse = _sparse_retrieve(rt, question, top_k)
        merged = _hybrid_merge(
            dense, sparse, alpha=float(rag_cfg["hybrid_alpha"]), top_k=top_k
        )

    # 2) Сбор retrieved объектов (дедуп по chunk_id не нужен, но полезен по doc_path)
    retrieved: List[dict] = []
    for idx, sc in merged:
        ch = rt.chunks[idx]
        retrieved.append(
            {
                "source": ch.get("doc_path"),
                "chunk_id": ch.get("chunk_id"),
                "score": float(sc),
                "snippet": ch.get("text", "")[:1200].strip(),  # ограничим контекст
            }
        )

    # Дедуп по source (уменьшает повторы и мусор в контексте)
    dedup = []
    seen = set()
    for r in retrieved:
        if r["source"] in seen:
            continue
        seen.add(r["source"])
        dedup.append(r)
    retrieved = dedup

    # 3) Вызов LLM (Ollama OpenAI-compatible)
    client = OpenAI(base_url=rag_cfg["ollama_base_url"], api_key="ollama")
    messages = _build_messages(question, retrieved)

    resp = client.chat.completions.create(
        model=rag_cfg["llm_model"],
        messages=messages,
        temperature=float(rag_cfg["temperature"]),
        top_p=float(rag_cfg["top_p"]),
        max_tokens=int(rag_cfg["max_tokens"]),
    )

    answer_text = resp.choices[0].message.content or ""
    return answer_text.strip(), retrieved


# ---- Task: прогон RAG (retrieval + generation)
def _get_input_question(item):
    inp = getattr(item, "input", None)
    if isinstance(inp, dict):
        return inp.get("question") or inp.get("q")
    if isinstance(inp, str):
        return inp
    # на случай, если SDK отдаёт dict вместо объекта
    if isinstance(item, dict):
        inp2 = item.get("input")
        if isinstance(inp2, dict):
            return inp2.get("question") or inp2.get("q")
        if isinstance(inp2, str):
            return inp2
    raise TypeError(f"Unsupported item.input type: {type(inp)}")


def _get_gold_paths(item, expected_output):
    exp = expected_output
    if exp is None:
        exp = getattr(item, "expected_output", None)

    # exp может быть dict или list
    if isinstance(exp, dict):
        gp = exp.get("gold_paths", [])
        return gp if isinstance(gp, list) else []
    if isinstance(exp, list):
        return exp
    return []


def rag_task(*, item, **kwargs):
    question = _get_input_question(item)
    gold_paths = _get_gold_paths(item, getattr(item, "expected_output", None))

    answer_text, retrieved = rag_inference_returning_sources(question)
    retrieved_paths = [r["source"] for r in retrieved]

    return {
        "answer": answer_text,
        "retrieved_paths": retrieved_paths,
        "gold_paths": gold_paths,
    }


def recall_at_5(*, output, **kwargs):
    r, _, _ = retrieval_metrics(output["retrieved_paths"], output["gold_paths"], k=5)
    return Evaluation(name="Recall@5", value=r)

def precision_at_5(*, output, **kwargs):
    _, p, _ = retrieval_metrics(output["retrieved_paths"], output["gold_paths"], k=5)
    return Evaluation(name="Precision@5", value=p)

def mrr_at_5(*, output, **kwargs):
    _, _, m = retrieval_metrics(output["retrieved_paths"], output["gold_paths"], k=5)
    return Evaluation(name="MRR@5", value=m)

def recall_at_10(*, output, **kwargs):
    r, _, _ = retrieval_metrics(output["retrieved_paths"], output["gold_paths"], k=10)
    return Evaluation(name="Recall@10", value=r)

def precision_at_10(*, output, **kwargs):
    _, p, _ = retrieval_metrics(output["retrieved_paths"], output["gold_paths"], k=10)
    return Evaluation(name="Precision@10", value=p)

def mrr_at_10(*, output, **kwargs):
    _, _, m = retrieval_metrics(output["retrieved_paths"], output["gold_paths"], k=10)
    return Evaluation(name="MRR@10", value=m)


def main():
    if not langfuse.auth_check():
        raise RuntimeError("Langfuse auth_check() failed")

    dataset = langfuse.get_dataset(DATASET_NAME)

    # В metadata можно положить конфиг retrieval из Лабы 2 (chunk_size, overlap, mode, run_id индекса, и т.д.)
    result = dataset.run_experiment(
        name="RAG eval (hybrid v1)",
        description="FastAPI docs RAG",
        task=rag_task,
        evaluators=[
            recall_at_5,
            precision_at_5,
            mrr_at_5,
            recall_at_10,
            precision_at_10,
            mrr_at_10,
        ],
        metadata={
            "mode": "hybrid",
            "top_k": 10,
            # "chunk_size": 400,
            # "overlap": 80,
            # "run_id": "...",
        },
    )

    print(result.format())
    langfuse.flush()


if __name__ == "__main__":
    main()
