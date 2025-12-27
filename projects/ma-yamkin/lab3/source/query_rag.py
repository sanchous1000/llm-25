import yaml
from pathlib import Path
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langfuse import propagate_attributes


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def query_rag(question, langfuse, name, trace_id):
    config = load_config()

    artifacts_dir = Path(f"artifacts/{name}")

    if not artifacts_dir.exists():
        raise RuntimeError(
            f"Индекс не найден: {artifacts_dir}. Сначала запустите build_index.py и load_to_vector_store.py")

    print("Загрузка FAISS-индекса...")
    index = faiss.read_index(str(artifacts_dir / "faiss.index"))
    with open(artifacts_dir / "metadata.json", encoding="utf-8") as f:
        chunks = json.load(f)

    chunk_arxiv_ids = [chunk["id"] for chunk in chunks]

    embedder = SentenceTransformer(config["embedding"]["model"])
    top_k = config["retrieval"]["top_k"]

    query_vec = embedder.encode(question)
    faiss.normalize_L2(query_vec.reshape(1, -1))

    distances, indices = index.search(query_vec.reshape(1, -1).astype(np.float32), k=top_k)

    contexts = []
    sources = []
    for idx in indices[0]:
        chunk = chunks[idx]
        text = chunk["text"]
        arxiv_id = chunk.get("arxiv_id", "unknown")
        page = chunk.get("page", "unknown")

        contexts.append(text)
        sources.append((arxiv_id, page))

    if not contexts:
        print("Не найдено релевантных фрагментов.")
        return

    context_block = "\n\n".join(
        f"[Источник: arXiv:{arxiv_id}, стр. {page}]\n{ctx}"
        for (arxiv_id, page), ctx in zip(sources, contexts)
    )

    with propagate_attributes(user_id="user_12345"):
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="search-in-rag",
            input={"query": question},
            trace_context={"trace_id": trace_id},
            metadata={
                'sources': sources,
                'distances': distances
            }
        ) as root_span:
            root_span.update(output={"answer": context_block})

    return context_block, chunk_arxiv_ids, indices
