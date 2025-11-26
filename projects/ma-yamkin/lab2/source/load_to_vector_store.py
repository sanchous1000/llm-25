import numpy as np
import argparse
import json
from pathlib import Path
import faiss
from utils import load_config, get_config_hash


def main(rebuild: bool = False):
    config = load_config()
    config_hash = get_config_hash(config)
    artifacts_dir = Path(f"../artifacts/index_{config_hash}")

    faiss_path = artifacts_dir / "faiss.index"
    meta_path = artifacts_dir / "metadata.json"

    if faiss_path.exists() and meta_path.exists() and not rebuild:
        print(f"FAISS-индекс уже существует в {artifacts_dir}. Пропускаем.")
        return

    with open(artifacts_dir / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(artifacts_dir / "embeddings.npy")

    assert len(chunks) == embeddings.shape[0], "Несовпадение количества чанков и эмбеддингов"

    dim = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(dim, config["vector_store"].get("hnsw_m", 16))
    index.hnsw.efConstruction = config["vector_store"].get("hnsw_ef_construct", 100)
    index.hnsw.efSearch = config["vector_store"].get("hnsw_ef_search", 50)

    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, str(faiss_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ FAISS-индекс сохранён в {artifacts_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Пересоздать индекс")
    args = parser.parse_args()
    main(rebuild=args.rebuild)
