import argparse
from utils.qdrant import QdrantCollection


def main():
    ap = argparse.ArgumentParser(description="Загрузка индекса в векторное хранилище")
    ap.add_argument("--qdrant_host", default="localhost", help="Хост Qdrant")
    ap.add_argument("--qdrant_port", type=int, default=6333, help="Порт Qdrant")
    ap.add_argument("--collection", default="vllm_docs", help="Название коллекции")
    ap.add_argument(
        "--drop-and-reindex",
        action="store_true",
        help="Удалить коллекцию и переиндексировать"
    )
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="Пересобрать коллекцию (очистить и пересоздать)"
    )
    ap.add_argument(
        "--vec_size",
        type=int,
        help="Размерность векторов (требуется при создании новой коллекции)"
    )
    ap.add_argument(
        "--distance",
        default="cosine",
        choices=["cosine", "dot", "euclidean"],
        help="Метрика расстояния"
    )
    ap.add_argument(
        "--hnsw_m",
        type=int,
        default=16,
        help="Параметр M для HNSW индекса"
    )
    ap.add_argument(
        "--hnsw_ef_construction",
        type=int,
        default=100,
        help="Параметр ef_construction для HNSW"
    )
    ap.add_argument(
        "--hnsw_ef_search",
        type=int,
        default=50,
        help="Параметр ef_search для HNSW"
    )
    args = ap.parse_args()

    qdrant = QdrantCollection(name=args.collection, host=args.qdrant_host, port=args.qdrant_port)

    if args.drop_and_reindex or args.rebuild:
        print(f"Удаление коллекции '{args.collection}'...")
        try:
            qdrant.delete_collection()
            print("✓ Коллекция удалена")
        except Exception as e:
            print(f"Warning: {e}")

    if args.vec_size:
        hnsw_config = {
            "m": args.hnsw_m,
            "ef_construction": args.hnsw_ef_construction,
            "ef_search": args.hnsw_ef_search
        }
        print(f"Создание коллекции '{args.collection}'...")
        qdrant.ensure_exists(
            vec_size=args.vec_size,
            distance=args.distance,
            recreate=False,
            hnsw_config=hnsw_config
        )
        print(f"Коллекция создана с параметрами:")
        print(f"  Размерность: {args.vec_size}")
        print(f"  Метрика: {args.distance}")
        print(f"  HNSW: M={args.hnsw_m}, ef_construction={args.hnsw_ef_construction}, ef_search={args.hnsw_ef_search}")
    else:
        print("Для создания новой коллекции укажите --vec_size")
        print("Для загрузки данных используйте build_embeddings.py")

if __name__ == "__main__":
    main()

