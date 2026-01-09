"""
Скрипт для автоматического заполнения ground_truth.json.
Находит релевантные чанки для каждого вопроса с помощью семантического поиска.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import get_device, load_questions, load_chunks
import argparse


def find_relevant_chunks(question: str,
                         chunks: List[Dict[str, Any]],
                         embedding_model: SentenceTransformer,
                         top_k: int = 5,
                         threshold: float = 0.7) -> List[int]:
    """
    Находит релевантные чанки для вопроса.
    
    Args:
        question: Вопрос
        chunks: Список всех чанков
        embedding_model: Модель для эмбеддингов
        top_k: Количество топ-результатов для рассмотрения
        threshold: Порог релевантности (cosine similarity)
    
    Returns:
        Список индексов релевантных чанков
    """
    # Получаем эмбеддинг вопроса
    question_embedding = embedding_model.encode([question], show_progress_bar=False)[0]
    
    # Загружаем dense эмбеддинги чанков
    chunks_dir = Path("chunks")
    version_dirs = sorted(chunks_dir.glob("v_*"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not version_dirs:
        return []
    
    dense_file = version_dirs[0] / "dense_embeddings.pkl"
    if not dense_file.exists():
        # Если нет сохраненных эмбеддингов, вычисляем на лету
        texts = [chunk["text"] for chunk in chunks]
        chunk_embeddings = embedding_model.encode(texts, show_progress_bar=True)
    else:
        with open(dense_file, 'rb') as f:
            chunk_embeddings = np.array(pickle.load(f))
    
    # Вычисляем cosine similarity
    question_embedding_norm = question_embedding / np.linalg.norm(question_embedding)
    chunk_embeddings_norm = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    
    similarities = np.dot(chunk_embeddings_norm, question_embedding_norm)
    
    # Находим топ-k результатов
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Фильтруем по порогу
    relevant_indices = []
    for idx in top_indices:
        if similarities[idx] >= threshold:
            relevant_indices.append(int(idx))
    
    # Если ничего не найдено по порогу, берем топ-3
    if not relevant_indices:
        relevant_indices = [int(idx) for idx in top_indices[:3]]
    
    return relevant_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Генерация ground truth')
    parser.add_argument('--questions-file', type=str, default='questions.md',
                       help='Файл с вопросами')
    parser.add_argument('--chunks-dir', type=str, default='chunks',
                       help='Директория с чанками')
    parser.add_argument('--dense-model', type=str, default='intfloat/multilingual-e5-large',
                       help='Модель для эмбеддингов')
    parser.add_argument('--device', type=str, default=None,
                       help='Устройство для вычислений')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Количество топ-результатов для рассмотрения')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Порог релевантности (cosine similarity)')
    parser.add_argument('--output', type=str, default='ground_truth.json',
                       help='Файл для сохранения ground truth')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Генерация ground truth")
    print("=" * 60)
    
    # Загружаем вопросы
    print(f"\nЗагрузка вопросов из {args.questions_file}...")
    questions = load_questions(args.questions_file)
    print(f"Загружено вопросов: {len(questions)}")
    
    # Загружаем чанки
    print(f"\nЗагрузка чанков из {args.chunks_dir}...")
    chunks = load_chunks(args.chunks_dir)
    
    # Загружаем модель эмбеддингов
    device = get_device(args.device) if args.device else None
    print(f"\nЗагрузка модели эмбеддингов: {args.dense_model} на {device}")
    embedding_model = SentenceTransformer(args.dense_model, device=device)
    
    # Находим релевантные чанки для каждого вопроса
    print(f"\nПоиск релевантных чанков для каждого вопроса...")
    ground_truth = {}
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Вопрос: {question['question'][:50]}...")
        relevant_indices = find_relevant_chunks(
            question['question'],
            chunks,
            embedding_model,
            top_k=args.top_k,
            threshold=args.threshold
        )
        ground_truth[str(question['id'])] = relevant_indices
        print(f"  Найдено релевантных чанков: {len(relevant_indices)} (индексы: {relevant_indices})")
    
    # Сохраняем ground truth
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("Генерация ground truth завершена!")
    print(f"Результаты сохранены в: {output_path}")
    print(f"Всего вопросов обработано: {len(questions)}")
    print("=" * 60)