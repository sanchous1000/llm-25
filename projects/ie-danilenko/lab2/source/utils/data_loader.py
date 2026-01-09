#!/usr/bin/env python3
"""
Функции для загрузки данных из различных источников.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_questions(questions_file: str) -> List[Dict[str, Any]]:
    """
    Загружает вопросы из файла.
    
    Args:
        questions_file: Путь к файлу с вопросами
    
    Returns:
        Список словарей с вопросами (id, question, relevant_chunks)
    """
    questions_path = Path(questions_file)
    if not questions_path.exists():
        raise FileNotFoundError(f"Файл с вопросами не найден: {questions_file}")
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    questions = []
    lines = content.split('\n')
    current_question = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('##'):
            continue  # Пропускаем заголовки разделов
        if re.match(r'^\d+\.', line):
            # Нашли вопрос
            question_text = re.sub(r'^\d+\.\s*', '', line)
            if question_text:
                questions.append({
                    "id": len(questions) + 1,
                    "question": question_text,
                    "relevant_chunks": []  # Будет заполнено из ground truth
                })
    
    return questions


def load_ground_truth(gt_file: str) -> Dict[int, List[int]]:
    """
    Загружает ground truth (релевантные чанки для каждого вопроса).
    
    Формат JSON:
    {
        "1": [0, 5, 12],  # Индексы релевантных чанков для вопроса 1
        "2": [3, 7],
        ...
    }
    
    Args:
        gt_file: Путь к файлу с ground truth
    
    Returns:
        Словарь с релевантными чанками для каждого вопроса
    """
    gt_path = Path(gt_file)
    if gt_path.exists():
        with open(gt_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_chunks(chunks_dir: str, version: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Загружает чанки из директории.
    
    Args:
        chunks_dir: Директория с чанками
        version: Версия (хеш конфигурации), если None - загружает последнюю
    
    Returns:
        Список чанков
    """
    chunks_path = Path(chunks_dir)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Директория {chunks_dir} не найдена")
    
    # Находим версию
    if version:
        version_dir = chunks_path / f"v_{version}"
    else:
        # Находим последнюю версию
        version_dirs = sorted(chunks_path.glob("v_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not version_dirs:
            raise FileNotFoundError(f"Не найдено версий в {chunks_dir}")
        version_dir = version_dirs[0]
    
    print(f"Загрузка чанков из версии: {version_dir}")
    
    chunks_file = version_dir / "chunks.json"
    if not chunks_file.exists():
        raise FileNotFoundError(f"Файл {chunks_file} не найден")
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Загружено чанков: {len(chunks)}")
    return chunks


def load_documents(input_dir: str) -> List[Dict[str, Any]]:
    """
    Загружает все Markdown документы из директории.
    
    Args:
        input_dir: Директория с Markdown файлами
    
    Returns:
        Список словарей с документами (content, metadata)
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Ошибка: директория {input_dir} не существует")
        return []
    
    documents = []
    
    # Загружаем все .md файлы
    md_files = list(input_path.glob("*.md"))
    print(f"Найдено .md файлов: {len(md_files)}")
    
    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Пропускаем пустые файлы
            if not content.strip():
                continue
            
            metadata = {
                "source_file": str(md_file),
                "filename": md_file.name
            }
            
            documents.append({
                "content": content,
                "metadata": metadata
            })
        except Exception as e:
            print(f"Ошибка при загрузке {md_file}: {e}")
    
    return documents

