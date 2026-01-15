import time
from langfuse import get_client, propagate_attributes
import os
from dotenv import load_dotenv
from pathlib import Path
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import yaml
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss

from rank_bm25 import BM25Okapi

RE_WORD = re.compile(r"\w+", re.UNICODE)

langfuse = get_client()



def tokenize_words(text: str) -> List[str]:
    return [w.lower() for w in RE_WORD.findall(text)]


def load_bm25_if_available(art_dir: Path, cfg: dict):
    """
    Пытается загрузить BM25 артефакт (bm25.json) и собрать BM25Okapi.
    Если файла нет или rank_bm25 не установлен — возвращает None.
    """
    bm25_path = art_dir / "bm25.json"
    if BM25Okapi is None:
        return None
    if not bm25_path.exists():
        return None

    payload = json.loads(bm25_path.read_text(encoding="utf-8"))
    tokenized = payload.get("tokenized")
    if not tokenized:
        return None

    k1 = float(cfg.get("embedding", {}).get("bm25_k1", 1.2))
    b = float(cfg.get("embedding", {}).get("bm25_b", 0.75))
    return BM25Okapi(tokenized, k1=k1, b=b)


def run_retrieval(
    question: str,
    *,
    cfg: dict,
    embedder: SentenceTransformer,
    index: faiss.Index,
    chunks: List[dict],
    bm25=None,
    top_k: Optional[int] = None,
) -> List[dict]:
    """
    Возвращает список retrieved-чанков в едином формате:
    [
      {
        "chunk_id": "...",
        "doc_path": "...",
        "text": "...",
        "meta": {...},
        "score": <float>,      # итоговый скор (чем больше, тем лучше)
        "retrieval_mode": "dense|sparse|hybrid",
        "rank": <int>          # 1..K
      }, ...
    ]

    Поддерживает:
      - dense (FAISS)
      - sparse (BM25) если есть bm25
      - hybrid (комбинация рангов) если есть bm25
    """
    mode = cfg.get("embedding", {}).get("mode", "dense").lower()
    if top_k is None:
        top_k = int(cfg.get("retrieval", {}).get("top_k", 8))

    # --- Dense candidates
    dense_indices: List[int] = []
    dense_dists: List[float] = []

    if mode in ("dense", "hybrid"):
        qvec = embedder.encode([f"query: {question}"], normalize_embeddings=True).astype("float32")
        D, I = index.search(qvec, top_k)
        dense_indices = [int(i) for i in I[0]]
        dense_dists = [float(d) for d in D[0]]

    # --- Sparse candidates
    sparse_indices: List[int] = []
    sparse_scores: List[float] = []

    if mode in ("sparse", "hybrid") and bm25 is not None:
        q_tokens = tokenize_words(question)
        scores = bm25.get_scores(q_tokens)  # numpy-like
        scores_arr = np.asarray(scores, dtype=np.float32)
        idx = np.argsort(-scores_arr)[:top_k]
        sparse_indices = [int(i) for i in idx]
        sparse_scores = [float(scores_arr[i]) for i in idx]

    # Если просили sparse/hybrid, но bm25 нет — падаем в dense
    if mode in ("sparse", "hybrid") and bm25 is None:
        mode = "dense"

    # --- Build final ranking
    if mode == "dense":
        # FAISS HNSWFlat обычно L2: меньше = лучше. Превращаем в "скор", где больше = лучше.
        # Простая монотонная функция:
        def dist_to_score(d: float) -> float:
            return 1.0 / (1.0 + max(d, 0.0))

        scored = []
        for rank, (idx, dist) in enumerate(zip(dense_indices, dense_dists), start=1):
            ch = dict(chunks[idx])  # копия
            ch["score"] = float(dist_to_score(dist))
            ch["retrieval_mode"] = "dense"
            ch["rank"] = rank
            scored.append(ch)
        return scored

    if mode == "sparse":
        scored = []
        for rank, (idx, s) in enumerate(zip(sparse_indices, sparse_scores), start=1):
            ch = dict(chunks[idx])
            ch["score"] = float(s)
            ch["retrieval_mode"] = "sparse"
            ch["rank"] = rank
            scored.append(ch)
        return scored

    # --- Hybrid: комбинируем ранги (устойчиво и просто)
    # score = alpha*(1/rank_dense) + (1-alpha)*(1/rank_sparse)
    alpha = float(cfg.get("retrieval", {}).get("hybrid_alpha", 0.7))

    score_map: Dict[int, float] = {}
    for r, idx in enumerate(dense_indices, start=1):
        score_map[idx] = score_map.get(idx, 0.0) + alpha * (1.0 / r)

    for r, idx in enumerate(sparse_indices, start=1):
        score_map[idx] = score_map.get(idx, 0.0) + (1.0 - alpha) * (1.0 / r)

    ranked = sorted(score_map.items(), key=lambda x: -x[1])[:top_k]

    out = []
    for rank, (idx, s) in enumerate(ranked, start=1):
        ch = dict(chunks[idx])
        ch["score"] = float(s)
        ch["retrieval_mode"] = "hybrid"
        ch["rank"] = rank
        out.append(ch)

    return out


def build_messages_with_context(
    question: str,
    retrieved: List[dict],
    *,
    max_contexts: int = 8,
) -> Tuple[List[dict], List[str]]:
    """
    Собирает messages для chat.completions и возвращает:
      - messages (system+user)
      - used_sources (список уникальных doc_path в порядке появления)
    """
    system = (
        "Ты — RAG-ассистент по документации FastAPI (RU).\n"
        "Правила:\n"
        "1) Отвечай на русском.\n"
        "2) Используй ТОЛЬКО контекст ниже. Не используй внешние знания и не придумывай.\n"
        "3) НЕ перепечатывай контекст и НЕ перечисляй SOURCE/источники в тексте ответа.\n"
        "4) Если ответа нет в контексте — скажи: \"В предоставленных фрагментах документации это не найдено.\".\n"
        "5) Формат ответа СТРОГО такой:\n"
        "Ответ: <1-5 абзацев>\n"
        "Цитаты: [n], [m] (только номера фрагментов, которые реально использовал)\n"
        "Никаких списков источников, никаких ссылок/URL, кроме номеров [n]."
    )

    ctxs = retrieved[:max_contexts]

    ctx_text = []
    used_sources = []
    seen = set()

    for i, c in enumerate(ctxs, start=1):
        src = c.get("doc_path", "unknown")
        if src not in seen:
            seen.add(src)
            used_sources.append(src)

        snippet = (c.get("text") or "").strip()
        # Можно добавить скор/ранк для дебага, но в контекст обычно не нужно
        ctx_text.append(f"### CONTEXT [{i}] (SOURCE={src})\n{snippet}\n")

    user = (
        f"Вопрос: {question}\n\n"
        "Контекст:\n"
        + "\n".join(ctx_text)
        + "\n\n"
        "Соблюдай формат."
    )

    return (
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        used_sources,
    )


def call_llm_openai_compatible(
    client: OpenAI,
    *,
    model: str,
    messages: List[dict],
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    extra_body: Optional[dict] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Унифицированный вызов OpenAI-совместимого API (Ollama).
    Возвращает:
      - text
      - usage dict (если доступно)
    """
    params = {}
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if extra_body is not None:
        params["extra_body"] = extra_body

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        **params,
    )

    text = resp.choices[0].message.content or ""
    usage = {}
    if getattr(resp, "usage", None) is not None:
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
    return text, usage





def answer_question_rag(question: str, *, user_id: str, session_id: str, rag_cfg: dict):
    """
    rag_cfg: твои параметры (top_k, mode, run_id индекса, модель, temperature, ...)
    """

    with langfuse.start_as_current_observation(as_type="span", name="rag_turn") as root:
        # Протаскиваем user/session на все дочерние шаги
        with propagate_attributes(user_id=user_id, session_id=session_id):
            root.update(input={"question": question, "rag_cfg": rag_cfg})

            # 1) Retrieval
            t0 = time.time()
            with root.start_as_current_observation(as_type="span", name="retrieval") as span_retr:
                # >>> твой поиск по FAISS/Qdrant
                # retrieved = [{"source":"advanced/index.md","score":0.42,"chunk_id":"...","snippet":"..."}, ...]
                retrieved = run_retrieval(question, rag_cfg)  # реализовано у тебя в лабе 2

                span_retr.update(
                    input={"question": question},
                    output={
                        "top_k": rag_cfg.get("top_k"),
                        "hits": [
                            {"source": r["source"], "score": r.get("score"), "chunk_id": r.get("chunk_id")}
                            for r in retrieved
                        ],
                        "latency_s": round(time.time() - t0, 4),
                    },
                )

            # 2) Сборка промпта (контекст + инструкции)
            with root.start_as_current_observation(as_type="span", name="prompt_build") as span_pb:
                messages, used_sources = build_messages_with_context(question, retrieved, rag_cfg)
                # used_sources: уникальные источники, которые реально положили в контекст
                span_pb.update(
                    output={
                        "n_contexts": len(retrieved),
                        "used_sources": used_sources,
                    }
                )

            # 3) Вызов LLM (generation)
            t1 = time.time()
            with root.start_as_current_observation(
                as_type="generation",
                name="llm_generate",
                model=rag_cfg["llm_model"],
            ) as gen:
                # >>> твой вызов Ollama (OpenAI-совместимый REST) из лабы 1
                llm_text, usage = call_llm_openai_compatible(messages, rag_cfg)

                gen.update(
                    input={
                        "messages": messages,
                        "params": {
                            "temperature": rag_cfg.get("temperature"),
                            "top_p": rag_cfg.get("top_p"),
                            "max_tokens": rag_cfg.get("max_tokens"),
                        },
                    },
                    output={"text": llm_text},
                    metadata={
                        "latency_s": round(time.time() - t1, 4),
                        "usage": usage,  # если есть (prompt_tokens, completion_tokens)
                    },
                )

            # 4) Жёстко формируем структурированный вывод (НЕ как промпт)
            final_answer = format_answer_with_citations(llm_text, retrieved)

            root.update(output={"final_answer": final_answer})
            langfuse.flush()
            return final_answer


def format_answer_with_citations(answer_text: str, retrieved: list[dict]) -> str:
    # Берём уникальные источники (без дублей), оставляем топ-N
    uniq = []
    seen = set()
    for r in retrieved:
        src = r["source"]
        if src not in seen:
            seen.add(src)
            uniq.append(src)
    uniq = uniq[:5]

    lines = []
    lines.append(answer_text.strip())
    lines.append("")
    lines.append("Источники:")
    for i, src in enumerate(uniq, 1):
        lines.append(f"  [{i}] {src}")
    return "\n".join(lines)


def load_config(project_root: Path) -> dict:
    return yaml.safe_load((project_root / "config.yaml").read_text(encoding="utf-8"))

def run_id_from_cfg(cfg: dict) -> str:
    import hashlib
    blob = yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:10]

def load_chunks(art_dir: Path):
    chunks = []
    with (art_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def build_prompt(question: str, ctxs: list[dict]) -> list[dict]:
    system = (
        "Ты — RAG-ассистент по документации FastAPI (RU).\n"
        "Правила:\n"
        "1) Отвечай на русском.\n"
        "2) Используй ТОЛЬКО контекст ниже. Не используй внешние знания и не придумывай.\n"
        "3) НЕ перепечатывай контекст и НЕ перечисляй SOURCE/источники в тексте ответа.\n"
        "4) Если ответа нет в контексте — скажи: \"В предоставленных фрагментах документации это не найдено.\".\n"
        "5) Формат ответа СТРОГО такой:\n"
        "Ответ: <1-5 абзацев>\n"
        "Цитаты: [n], [m] (только номера фрагментов, которые реально использовал)\n"
        "Никаких списков источников, никаких ссылок/URL, кроме номеров [n]."
    )

    ctx_text = []
    for i, c in enumerate(ctxs, start=1):
        snippet = c["text"].strip()
        ctx_text.append(
            f"### CONTEXT [{i}] (SOURCE={c['doc_path']})\n{snippet}\n"
        )

    user = (
        f"Вопрос: {question}\n\n"
        "Контекст:\n"
        + "\n".join(ctx_text)
        + "\n\n"
        "Соблюдай формат."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]



def main():
    load_dotenv()
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    llm_model = os.getenv("OLLAMA_MODEL", "llama3.2:1b-instruct-q5_K_M")

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root)
    run_id = run_id_from_cfg(cfg)
    art_dir = project_root / "artifacts" / run_id

    chunks = load_chunks(art_dir)
    chunk = np.load(art_dir / "faiss_hnsw.npy", allow_pickle=False)
    index = faiss.deserialize_index(chunk)

    dense_model_name = cfg["embedding"]["dense_model"]
    embedder = SentenceTransformer(dense_model_name)

    client = OpenAI(base_url=base_url, api_key="ollama")

    top_k = int(cfg["retrieval"]["top_k"])
    
    bm25 = load_bm25_if_available(art_dir, cfg)  # добавь после загрузки chunks/index/embedder

    while True:
        q = input("> ").strip()
        if not q:
            break

        retrieved = run_retrieval(
            q,
            cfg=cfg,
            embedder=embedder,
            index=index,
            chunks=chunks,
            bm25=bm25,
            top_k=top_k,
        )

        messages, used_sources = build_messages_with_context(q, retrieved, max_contexts=top_k)

        answer, usage = call_llm_openai_compatible(
            client,
            model=llm_model,
            messages=messages,
            temperature=0.2,
            top_p=0.9,
            max_tokens=600,
            # extra_body={"top_k": 40, "repeat_penalty": 1.1}  # опционально для Ollama
        )

        print("\n" + answer + "\n")
        print("Источники:")
        for i, src in enumerate(used_sources, start=1):
            print(f"  [{i}] {src}")
        if usage:
            print(f"\nusage={usage}")
        print()

if __name__ == "__main__":
    main()
