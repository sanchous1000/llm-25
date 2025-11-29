from typing import List

import requests
from tqdm import tqdm


def embed_query(text: str, model: str, host: str) -> List[float]:
    """Получение эмбеддинга через Ollama"""
    r = requests.post(f"{host}/api/embeddings", json={"model": model, "prompt": text}, timeout=120)
    r.raise_for_status()
    return r.json()["embedding"]


def embed_texts(texts: List[str], model: str, host: str) -> List[List[float]]:
    out = []
    for t in tqdm(texts, desc="embeddings"):
        out.append(embed_query(t, model, host))
        
    return out


def generate_answer(prompt: str, model: str, host: str) -> str:
    # Синхронный вызов без стриминга
    r = requests.post(f"{host}/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


