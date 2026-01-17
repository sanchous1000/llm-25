"""
Проверяет, какие chunk_id находятся в Elasticsearch индексе.
"""
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

from config_utils import load_config
from es_utils import get_es_client


def main():
    lab2_dir = Path(__file__).parent.parent
    
    # Загружаем конфиг
    config_path = lab2_dir / "source/config.yaml"
    config = load_config(config_path)
    
    es_config = config.get("elasticsearch", {})
    
    # Подключаемся к ES
    try:
        es_client, es_url = get_es_client(es_config)
        print(f"Connected to Elasticsearch: {es_url}")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return
    
    index_name = es_config.get("index_name", "fastapi_docs")
    
    # Проверяем существование индекса
    if not es_client.indices.exists(index=index_name):
        print(f"\n[ERROR] Index '{index_name}' does not exist!")
        print("Run: python source/load_to_vector_store.py")
        return
    
    # Получаем статистику
    stats = es_client.indices.stats(index=index_name)
    doc_count = stats["indices"][index_name]["total"]["docs"]["count"]
    
    print(f"\nIndex: {index_name}")
    print(f"Documents in Elasticsearch: {doc_count}")
    
    # Загружаем текущую версию chunks
    index_dir = lab2_dir / "data/index"
    # Сортируем по времени модификации (самая новая последняя)
    version_dirs = [d for d in index_dir.iterdir() if d.is_dir()]
    version_dirs.sort(key=lambda d: d.stat().st_mtime)
    versions = [d.name for d in version_dirs]
    current_version = versions[-1]
    
    chunks_file = lab2_dir / "data/chunks" / current_version / "chunks.jsonl"
    local_count = sum(1 for _ in open(chunks_file, encoding="utf-8"))
    
    print(f"\nLocal version: {current_version}")
    print(f"Documents in local chunks: {local_count}")
    
    print(f"\n{'='*60}")
    if doc_count == local_count:
        print("[OK] Elasticsearch index matches local version!")
    else:
        print(f"[WARNING] Mismatch!")
        print(f"  Elasticsearch: {doc_count} documents")
        print(f"  Local: {local_count} documents")
        print(f"\nElasticsearch might have OLD version of index.")
        print("Solution: Reload index")
        print("  python source/load_to_vector_store.py --rebuild")
    
    # Проверяем sample chunk_id
    print(f"\n{'='*60}")
    print("Checking sample chunk_ids from expected_chunks.json...")
    
    expected_file = lab2_dir / "data/evaluation/expected_chunks.json"
    with open(expected_file, "r", encoding="utf-8") as f:
        expected_chunks = json.load(f)
    
    # Берем первые 3 expected chunk_id
    all_expected_ids = []
    for chunk_ids in expected_chunks.values():
        all_expected_ids.extend(chunk_ids)
    
    sample_ids = all_expected_ids[:5]
    
    for chunk_id in sample_ids:
        query = {"query": {"term": {"chunk_id": chunk_id}}}
        result = es_client.search(index=index_name, body=query, size=1)
        hits = result["hits"]["total"]["value"]
        
        status = "[OK]" if hits > 0 else "[MISSING]"
        print(f"  {status} {chunk_id}")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
