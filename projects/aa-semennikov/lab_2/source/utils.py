import json
import hashlib
import yaml
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def compute_config_hash(config):
    relevant_params = {
        'chunking': config['chunking'],
        'embeddings': {
            'model': config['embeddings']['dense']['model'],
            'dimension': config['embeddings']['dense'].get('dimension'),
        }
    }
    config_str = json.dumps(relevant_params, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()


def save_metadata(index_path, config, config_hash, num_documents, num_chunks):
    metadata = {
        'version': config_hash[:8],
        'config_hash': config_hash,
        'timestamp': datetime.now().isoformat(),
        'num_documents': num_documents,
        'num_chunks': num_chunks,
        'config': config,
        'statistics': {
            'avg_chunks_per_doc': num_chunks / num_documents if num_documents > 0 else 0,
        }
    }
    
    metadata_file = index_path / 'index_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_metadata(index_path):
    """
    Загружает метаданные индекса.
    
    Args:
        index_path: Путь к директории индекса
        
    Returns:
        Словарь с метаданными или None если файл не найден
    """
    metadata_file = index_path / 'index_metadata.json'
    
    if not metadata_file.exists():
        return None
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_if_rebuild_needed(config, index_path):
    """
    Проверяет, нужно ли перестраивать индекс.
    
    Args:
        config: Текущая конфигурация
        index_path: Путь к директории индекса
        
    Returns:
        True если нужна пересборка, False иначе
    """
    if not config['versioning']['auto_version']:
        # Версионирование отключено, всегда перестраиваем
        return True
    
    metadata = load_metadata(index_path)
    
    if metadata is None:
        # Метаданные не найдены, нужна сборка
        return True
    
    # Сравниваем хеши конфигурации
    current_hash = compute_config_hash(config)
    stored_hash = metadata.get('config_hash')
    
    if current_hash != stored_hash:
        return True
    
    return False


def print_summary(config, num_documents, num_chunks, output_path):
    print("\n" + "=" * 80)
    print("СВОДКА ПО ИНДЕКСУ")
    print("=" * 80)
    
    print(f"\nДокументы:")
    print(f"  Обработано документов: {num_documents}")
    print(f"  Создано чанков: {num_chunks}")
    print(f"  Среднее кол-во чанков на документ: {num_chunks/num_documents:.1f}")
    
    print(f"\nПараметры разбиения (Markdown):")
    print(f"  Размер чанка: {config['chunking']['chunk_size']} токенов")
    print(f"  Перекрытие: {config['chunking']['chunk_overlap']} токенов")
    print(f"  Уровни заголовков: {', '.join(config['chunking']['header_levels'])}")
    print(f"  Заголовки в тексте: {'Да' if config['chunking']['include_headers_in_text'] else 'Нет'}")
    
    print(f"\nПараметры векторизации (Dense):")
    print(f"  Модель: {config['embeddings']['dense']['model']}")
    print(f"  Устройство: {config['embeddings']['dense']['device']}")
    if config['embeddings']['dense'].get('dimension'):
        print(f"  Размерность: {config['embeddings']['dense']['dimension']}")
    
    print(f"\nВыходные файлы:")
    print(f"  Директория: {output_path}")
    print(f"  Чанки: {output_path / config['paths']['chunks_dir']}")
    print(f"  Эмбеддинги: {output_path / config['paths']['embeddings_dir']}")
    print(f"  Индекс: {output_path / config['paths']['index_dir']}")
    
    print("\n" + "=" * 80 + "\n")