"""Утилиты для работы с Ollama API для получения эмбеддингов."""
import requests
from typing import List


def embed_texts(texts: List[str], model: str = "nomic-embed-text", host: str = "http://localhost:11434") -> List[List[float]]:
    """
    Получить эмбеддинги для списка текстов.
    
    Args:
        texts: Список текстов для векторизации
        model: Название модели эмбеддингов
        host: URL сервера Ollama
        
    Returns:
        Список векторов эмбеддингов
    """
    vectors = []
    for text in texts:
        vector = embed_query(text, model=model, host=host)
        vectors.append(vector)
    return vectors


def embed_query(text: str, model: str = "nomic-embed-text", host: str = "http://localhost:11434") -> List[float]:
    """
    Получить эмбеддинг для одного текста.
    
    Args:
        text: Текст для векторизации
        model: Название модели эмбеддингов
        host: URL сервера Ollama
        
    Returns:
        Вектор эмбеддинга
    """
    url = f"{host}/api/embeddings"
    payload = {
        "model": model,
        "prompt": text
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["embedding"]

