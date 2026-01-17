"""
Скрипт для очистки старых индексов в Elasticsearch.

Позволяет удалить старые индексы перед загрузкой новых данных.
"""
import argparse
from pathlib import Path

from elasticsearch import Elasticsearch

from config_utils import load_config
from es_utils import get_es_client


def list_indices(es_client: Elasticsearch) -> list[str]:
    """Получает список всех индексов."""
    indices = es_client.indices.get_alias(index="*")
    return list(indices.keys())


def delete_index(es_client: Elasticsearch, index_name: str) -> bool:
    """Удаляет индекс.
    
    Args:
        es_client: Клиент Elasticsearch.
        index_name: Имя индекса для удаления.
    
    Returns:
        True если индекс удален, False если не существовал.
    """
    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)
        return True
    return False


def delete_all_indices(es_client: Elasticsearch, pattern: str = "*") -> int:
    """Удаляет все индексы, соответствующие паттерну.
    
    Args:
        es_client: Клиент Elasticsearch.
        pattern: Паттерн для поиска индексов (например, "fastapi_*").
    
    Returns:
        Количество удаленных индексов.
    """
    indices = list_indices(es_client)
    matching_indices = [idx for idx in indices if pattern in idx]
    
    if not matching_indices:
        return 0
    
    for index_name in matching_indices:
        es_client.indices.delete(index=index_name)
    
    return len(matching_indices)


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="Clean Elasticsearch indices",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="source/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--index",
        type=str,
        help="Specific index name to delete",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Pattern to match indices (e.g., 'fastapi_*')",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all indices",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Delete ALL indices (use with caution!)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm deletion (required for --all)",
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    lab2_dir = script_dir.parent
    
    config_path = lab2_dir / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_config(config_path)
    es_config = config.get("elasticsearch", {})
    
    # Подключаемся к Elasticsearch
    es_client, es_url = get_es_client(es_config)
    print(f"Connected to Elasticsearch: {es_url}")
    
    print(f"Connected to Elasticsearch: {es_client.info()['cluster_name']}")
    
    # Список индексов
    if args.list:
        indices = list_indices(es_client)
        print(f"\nFound {len(indices)} indices:")
        for idx in indices:
            stats = es_client.indices.stats(index=idx)
            doc_count = stats["indices"][idx]["total"]["docs"]["count"]
            print(f"  - {idx} ({doc_count} documents)")
        return
    
    # Удаление всех индексов
    if args.all:
        if not args.confirm:
            print("ERROR: --all requires --confirm flag for safety")
            print("This will delete ALL indices in Elasticsearch!")
            return
        
        indices = list_indices(es_client)
        # Исключаем системные индексы
        user_indices = [idx for idx in indices if not idx.startswith('.')]
        
        if not user_indices:
            print("No user indices to delete")
            return
        
        print(f"\nWARNING: This will delete {len(user_indices)} indices:")
        for idx in user_indices:
            print(f"  - {idx}")
        
        response = input("\nType 'DELETE ALL' to confirm: ")
        if response != "DELETE ALL":
            print("Cancelled")
            return
        
        for index_name in user_indices:
            es_client.indices.delete(index=index_name)
            print(f"Deleted: {index_name}")
        
        print(f"\n[OK] Deleted {len(user_indices)} indices")
        return
    
    # Удаление по паттерну
    if args.pattern:
        indices = list_indices(es_client)
        matching = [idx for idx in indices if args.pattern in idx]
        
        if not matching:
            print(f"No indices matching pattern '{args.pattern}'")
            return
        
        print(f"\nFound {len(matching)} indices matching '{args.pattern}':")
        for idx in matching:
            print(f"  - {idx}")
        
        if not args.confirm:
            response = input("\nDelete these indices? (yes/no): ")
            if response.lower() != "yes":
                print("Cancelled")
                return
        
        for index_name in matching:
            es_client.indices.delete(index=index_name)
            print(f"Deleted: {index_name}")
        
        print(f"\n[OK] Deleted {len(matching)} indices")
        return
    
    # Удаление конкретного индекса
    if args.index:
        if delete_index(es_client, args.index):
            print(f"[OK] Deleted index: {args.index}")
        else:
            print(f"Index '{args.index}' does not exist")
        return
    
    # По умолчанию удаляем индекс из конфига
    index_name = es_config.get("index_name", "fastapi_docs")
    if delete_index(es_client, index_name):
        print(f"[OK] Deleted index: {index_name}")
    else:
        print(f"Index '{index_name}' does not exist")
    
    # Также удаляем алиасы
    aliases = es_client.indices.get_alias(name=f"{index_name}_v*")
    if aliases:
        for alias_name in aliases.keys():
            es_client.indices.delete_alias(index=alias_name, name=alias_name)
            print(f"[OK] Removed alias: {alias_name}")


if __name__ == "__main__":
    main()
