import argparse
import uuid

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, PointStruct

try:
    from build_index import split_documents, load_documents
except ImportError:

    print("⚠️  Не удалось импортировать split_documents из build_index.py.")
    print("⚠️  Пожалуйста, убедитесь, что build_index.py находится в той же папке или PYTHONPATH.")
    exit(1)


def setup_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int, args):
    if client.collection_exists(collection_name):
        if args.rebuild:
            print(f"🗑️  Удаляю старую коллекцию '{collection_name}' (--rebuild)...")
            client.delete_collection(collection_name)
        else:
            print(f"ℹ️  Коллекция '{collection_name}' уже существует. Добавляем данные.")
            return

    print(f"🔨 Создается коллекция '{collection_name}'...")
    print(f"   ⚙️  Вектор: {vector_size} (Cosine)")
    print(f"   ⚙️  HNSW: M={args.hnsw_m}, ef_construction={args.hnsw_ef}")

    quantization_config = None
    if args.quantization:
        print("   ⚙️  Квантование: Scalar (Int8)")
        quantization_config = models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=True
            )
        )

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(
            m=args.hnsw_m,
            ef_construct=args.hnsw_ef, 
        ),
        quantization_config=quantization_config
    )

def main():
    parser = argparse.ArgumentParser(description="Загрузка подготовленных чанков в Qdrant")
    
    parser.add_argument("--source_dir", default="verl_sources", help="Папка с документами")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--add_header", action="store_true", help="Добавлять имя файла в текст")
    
    parser.add_argument("--emb_model", default="intfloat/e5-large-v2")
    
    parser.add_argument("--url", default="http://localhost:6333", help="URL Qdrant")
    parser.add_argument("--collection", default="verl_rag", help="Имя коллекции")
    parser.add_argument("--rebuild", action="store_true", help="Пересоздать коллекцию")
    parser.add_argument("--hnsw_m", type=int, default=16)
    parser.add_argument("--hnsw_ef", type=int, default=100)
    parser.add_argument("--quantization", action="store_true", help="Включить сжатие")

    args = parser.parse_args()  

    print("🔄 Этап 1: Подготовка данных...")
    docs = load_documents(args.source_dir)
    chunks = split_documents(docs, args.chunk_size, args.overlap, args.add_header)
    
    if not chunks:
        print("❌ Нет чанков для загрузки.")
        return

    print(f"🧠 Этап 2: Инициализация модели {args.emb_model}...")
    embeddings_model = HuggingFaceEmbeddings(model_name=args.emb_model)
    
    test_vec = embeddings_model.embed_query("test")
    vector_size = len(test_vec)

    client = QdrantClient(url=args.url)
    setup_qdrant_collection(client, args.collection, vector_size, args)

    batch_size = 64
    total_chunks = len(chunks)
    print(f"🚀 Этап 3: Загрузка {total_chunks} чанков в Qdrant...")

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        
        texts = [doc.page_content for doc in batch]
        vectors = embeddings_model.embed_documents(texts)
        
        points = []
        for j, (text, vector) in enumerate(zip(texts, vectors)):
            payload = batch[j].metadata.copy()
            payload["page_content"] = text
            
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, text))
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            ))
            
        client.upsert(
            collection_name=args.collection,
            points=points
        )
        print(f"   📦 Прогресс: {min(i + batch_size, total_chunks)} / {total_chunks}")

    print(f"\n✅ Успешно! Коллекция '{args.collection}' обновлена.")
    print(f"   📊 Теперь в базе: {client.count(args.collection).count} векторов")

if __name__ == "__main__":
    main()