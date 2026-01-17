"""
Переносит разметку expected_chunks с одной версии на другую.

Алгоритм:
1. Берёт expected chunk_id из версии A
2. Извлекает эмбеддинг этого чанка
3. Ищет в версии B чанки с наибольшим косинусным сходством
4. Сохраняет найденные chunk_id в новый файл expected_chunks
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_expected_chunks(expected_file: Path) -> Dict[str, List[str]]:
    """Загружает expected_chunks из JSON."""
    with open(expected_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_chunks_map(chunks_file: Path) -> Dict[str, dict]:
    """Загружает чанки как словарь {chunk_id: {...}} с индексами."""
    chunks_map = {}
    with open(chunks_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            chunk = json.loads(line)
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id:
                chunk["_index"] = idx  # Сохраняем индекс для доступа к эмбеддингу
                chunks_map[chunk_id] = chunk
    return chunks_map


def load_embeddings(embeddings_file: Path) -> np.ndarray:
    """Загружает эмбеддинги из .npy файла."""
    return np.load(embeddings_file)


def find_most_similar_chunks(
    source_embedding: np.ndarray,
    target_embeddings: np.ndarray,
    target_chunk_ids: List[str],
    top_k: int = 3,
    min_similarity: float = 0.75,
) -> List[tuple[str, float]]:
    """Находит наиболее похожие чанки по косинусной близости эмбеддингов.
    
    Args:
        source_embedding: Эмбеддинг исходного чанка (1D array)
        target_embeddings: Массив всех эмбеддингов целевой версии (N x dim)
        target_chunk_ids: Список chunk_id в том же порядке
        top_k: Сколько топовых результатов вернуть
        min_similarity: Минимальная косинусная близость для включения
    
    Returns:
        Список (chunk_id, similarity) отсортированный по убыванию similarity
    """
    # Вычисляем косинусное сходство со всеми чанками
    source_emb = source_embedding.reshape(1, -1)
    similarities = cosine_similarity(source_emb, target_embeddings)[0]
    
    # Находим топ-k индексов
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        sim = similarities[idx]
        if sim >= min_similarity:
            results.append((target_chunk_ids[idx], float(sim)))
    
    return results


def transfer_expected_chunks(
    version_source: str,
    version_target: str,
    expected_source_file: Path,
    expected_target_file: Path,
    lab2_dir: Path,
) -> None:
    """Переносит разметку с одной версии на другую."""
    
    print("="*80)
    print("ПЕРЕНОС РАЗМЕТКИ EXPECTED_CHUNKS")
    print("="*80)
    
    # Загружаем expected chunks для исходной версии
    print(f"\nЗагрузка expected_chunks из: {expected_source_file.name}")
    expected_source = load_expected_chunks(expected_source_file)
    print(f"  Вопросов: {len(expected_source)}")
    total_chunks = sum(len(v) for v in expected_source.values())
    print(f"  Всего expected chunk_id: {total_chunks}")
    
    # Загружаем чанки и эмбеддинги исходной версии
    print(f"\nЗагрузка чанков и эмбеддингов версии A: {version_source}")
    chunks_source_file = lab2_dir / f"data/chunks/{version_source}/chunks.jsonl"
    embeddings_source_file = lab2_dir / f"data/embeddings/{version_source}/embeddings.npy"
    chunks_source = load_chunks_map(chunks_source_file)
    embeddings_source = load_embeddings(embeddings_source_file)
    print(f"  Чанков: {len(chunks_source)}, эмбеддингов: {embeddings_source.shape}")
    
    # Загружаем чанки и эмбеддинги целевой версии
    print(f"\nЗагрузка чанков и эмбеддингов версии B: {version_target}")
    chunks_target_file = lab2_dir / f"data/chunks/{version_target}/chunks.jsonl"
    embeddings_target_file = lab2_dir / f"data/embeddings/{version_target}/embeddings.npy"
    chunks_target = load_chunks_map(chunks_target_file)
    embeddings_target = load_embeddings(embeddings_target_file)
    print(f"  Чанков: {len(chunks_target)}, эмбеддингов: {embeddings_target.shape}")
    
    # Создаём упорядоченный список chunk_id для целевой версии (в порядке строк файла)
    target_chunk_ids_list = []
    with open(chunks_target_file, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            chunk_id = chunk.get("chunk_id") or chunk.get("id") or ""
            target_chunk_ids_list.append(chunk_id)
    
    print(f"  Создан список chunk_id: {len(target_chunk_ids_list)} элементов")
    
    # Переносим разметку
    print("\nПеренос разметки...")
    expected_target = {}
    transfer_details = []  # Детальная информация о переносе
    
    stats = {
        "total_questions": len(expected_source),
        "total_expected_chunks": 0,
        "found_by_text": 0,
        "not_found": 0,
    }
    
    for question, expected_chunk_ids in expected_source.items():
        print(f"\n[?] {question[:60]}...")
        
        new_chunk_ids = set()
        question_transfers = []  # Переносы для текущего вопроса
        
        for chunk_id in expected_chunk_ids:
            stats["total_expected_chunks"] += 1
            
            # 1. Проверяем наличие в исходной версии
            if chunk_id not in chunks_source:
                print(f"  ✗ [{chunk_id}] НЕ НАЙДЕН в исходной версии")
                stats["not_found"] += 1
                question_transfers.append({
                    "source_chunk_id": chunk_id,
                    "target_chunk_id": None,
                    "similarity": None,
                    "status": "not_in_source",
                })
                continue
            
            # 2. Получаем эмбеддинг исходного чанка
            source_chunk = chunks_source[chunk_id]
            source_idx = source_chunk["_index"]
            source_embedding = embeddings_source[source_idx]
            
            # 3. Ищем наиболее похожие чанки в целевой версии по косинусной близости
            similar_chunks = find_most_similar_chunks(
                source_embedding,
                embeddings_target,
                target_chunk_ids_list,
                top_k=1,  # Берём только 1 лучший
                min_similarity=0.85,  # Повышаем порог для точности
            )
            
            if similar_chunks:
                # Берём только лучший результат
                best_id, best_sim = similar_chunks[0]
                if best_id and best_id.strip():  # Фильтруем пустые
                    new_chunk_ids.add(best_id)
                    stats["found_by_text"] += 1
                    print(f"  → [{chunk_id}] → {best_id} (sim={best_sim:.3f})")
                    question_transfers.append({
                        "source_chunk_id": chunk_id,
                        "target_chunk_id": best_id,
                        "similarity": float(best_sim),
                        "status": "found",
                    })
                else:
                    print(f"  ✗ [{chunk_id}] → найден пустой chunk_id")
                    stats["not_found"] += 1
                    question_transfers.append({
                        "source_chunk_id": chunk_id,
                        "target_chunk_id": "",
                        "similarity": float(best_sim) if best_sim else None,
                        "status": "empty_target",
                    })
            else:
                print(f"  ✗ [{chunk_id}] НЕ НАЙДЕН (косинусная близость < 0.85)")
                stats["not_found"] += 1
                question_transfers.append({
                    "source_chunk_id": chunk_id,
                    "target_chunk_id": None,
                    "similarity": None,
                    "status": "low_similarity",
                })
        
        expected_target[question] = list(new_chunk_ids)
        
        # Сохраняем детали для этого вопроса
        transfer_details.append({
            "question": question,
            "source_expected_count": len(expected_chunk_ids),
            "target_found_count": len(new_chunk_ids),
            "transfers": question_transfers,
        })
    
    # Статистика
    print("\n" + "="*80)
    print("СТАТИСТИКА ПЕРЕНОСА")
    print("="*80)
    print(f"Вопросов: {stats['total_questions']}")
    print(f"Всего expected chunks: {stats['total_expected_chunks']}")
    print(f"Найдено по косинусной близости (>0.75): {stats['found_by_text']}")
    print(f"Не найдено: {stats['not_found']}")
    print(f"Успешность: {stats['found_by_text'] / stats['total_expected_chunks'] * 100:.1f}%")
    
    # Сохраняем результат expected_chunks
    print(f"\nСохранение expected_chunks в: {expected_target_file}")
    with open(expected_target_file, "w", encoding="utf-8") as f:
        json.dump(expected_target, f, indent=2, ensure_ascii=False)
    
    # Сохраняем детальную информацию о переносе
    transfer_result_file = expected_target_file.parent / f"transfer_result_{version_source}_to_{version_target}.json"
    print(f"Сохранение деталей переноса в: {transfer_result_file}")
    
    transfer_result = {
        "version_source": version_source,
        "version_target": version_target,
        "timestamp": expected_source_file.name,
        "statistics": stats,
        "details": transfer_details,
    }
    
    with open(transfer_result_file, "w", encoding="utf-8") as f:
        json.dump(transfer_result, f, indent=2, ensure_ascii=False)
    
    print("\n✓ Готово!")
    print(f"\nФайлы созданы:")
    print(f"  - {expected_target_file.name}")
    print(f"  - {transfer_result_file.name}")
    print(f"\nТеперь можно запустить evaluate для версии {version_target}:")
    print(f"  python evaluate.py --version {version_target}")


def main():
    parser = argparse.ArgumentParser(
        description="Переносит разметку expected_chunks между версиями"
    )
    parser.add_argument(
        "--version-source",
        type=str,
        default="4530f7569b81",
        help="Исходная версия (откуда брать expected_chunks)",
    )
    parser.add_argument(
        "--version-target",
        type=str,
        default="e6233de3342d",
        help="Целевая версия (для которой создать expected_chunks)",
    )
    parser.add_argument(
        "--expected-source",
        type=str,
        default="data/evaluation/expected_chunks_4530f7569b81.json",
        help="Исходный файл expected_chunks",
    )
    parser.add_argument(
        "--expected-target",
        type=str,
        default="data/evaluation/expected_chunks_e6233de3342d.json",
        help="Целевой файл expected_chunks",
    )
    
    args = parser.parse_args()
    
    lab2_dir = Path(__file__).parent.parent
    
    expected_source_file = lab2_dir / args.expected_source
    expected_target_file = lab2_dir / args.expected_target
    
    if not expected_source_file.exists():
        print(f"Исходный файл не найден: {expected_source_file}")
        return
    
    transfer_expected_chunks(
        args.version_source,
        args.version_target,
        expected_source_file,
        expected_target_file,
        lab2_dir,
    )


if __name__ == "__main__":
    main()
