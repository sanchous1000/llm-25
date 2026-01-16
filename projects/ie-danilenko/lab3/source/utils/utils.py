#!/usr/bin/env python3
"""
Общие утилиты для всех скриптов.
"""

import json
import hashlib
import re
from typing import Optional, List
import torch


def get_device(device_preference: Optional[str] = None) -> str:
    """
    Определяет доступное устройство для вычислений.
    
    Args:
        device_preference: Предпочтительное устройство (mps/cuda/cpu)
    
    Returns:
        Строка с названием устройства
    """
    if device_preference:
        # Проверяем доступность запрошенного устройства
        if device_preference == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            else:
                print("Warning: MPS недоступен, используется CPU")
                return "cpu"
        elif device_preference == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            else:
                print("Warning: CUDA недоступна, используется CPU")
                return "cpu"
        else:
            return device_preference
    
    # Автоматический выбор устройства
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_config_hash(config: dict) -> str:
    """
    Вычисляет хеш конфигурации для версионирования.
    
    Args:
        config: Словарь с конфигурацией
    
    Returns:
        Хеш конфигурации (первые 8 символов MD5)
    """
    # Создаем строковое представление конфигурации
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def tokenize(text: str) -> List[str]:
    """
    Простая токенизация для BM25.
    
    Args:
        text: Текст для токенизации
    
    Returns:
        Список токенов (слов в нижнем регистре)
    """
    # Приводим к нижнему регистру и разбиваем по словам
    words = re.findall(r'\b\w+\b', text.lower())
    return words

