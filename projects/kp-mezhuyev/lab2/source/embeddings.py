"""
Модуль для создания эмбеддингов текстов.

Поддерживает dense эмбеддинги через sentence-transformers или OpenAI.
"""
import os
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class BaseEmbedder(ABC):
    """Базовый класс для эмбеддеров."""
    
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Создает эмбеддинги для списка текстов.
        
        Args:
            texts: Список текстов для векторизации.
        
        Returns:
            Массив эмбеддингов (n_samples, n_dimensions).
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Возвращает размерность эмбеддингов."""
        pass


class DenseEmbedder(BaseEmbedder):
    """Создает плотные эмбеддинги с помощью sentence-transformers или OpenAI."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str = "cpu",
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
    ):
        """Инициализирует dense embedder.
        
        Args:
            model_name: Название модели или путь к модели.
            batch_size: Размер батча для обработки.
            device: Устройство для вычислений ('cpu' или 'cuda').
            openai_api_key: API ключ OpenAI (если используется OpenAI).
            openai_base_url: Базовый URL для OpenAI API.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        
        # Определяем, используется ли OpenAI
        self.use_openai = (
            "text-embedding" in model_name.lower() or
            "ada" in model_name.lower() or
            openai_api_key is not None
        )
        
        if self.use_openai:
            try:
                import openai
                self.openai_client = openai.OpenAI(
                    api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
                    base_url=openai_base_url,
                )
                self._dimension = None  # Будет определен при первом запросе
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAI embeddings. "
                    "Install it with: pip install openai"
                )
        else:
            # Используем sentence-transformers
            self.model = SentenceTransformer(model_name, device=device)
            self._dimension = self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: list[str]) -> np.ndarray:
        """Создает плотные эмбеддинги."""
        if self.use_openai:
            return self._embed_openai(texts)
        else:
            return self._embed_sentence_transformers(texts)
    
    def _embed_openai(self, texts: list[str]) -> np.ndarray:
        """Создает эмбеддинги через OpenAI API."""
        embeddings = []
        
        for i in tqdm(
            range(0, len(texts), self.batch_size),
            desc="Creating OpenAI embeddings",
        ):
            batch = texts[i:i + self.batch_size]
            
            response = self.openai_client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
            # Определяем размерность при первом батче
            if self._dimension is None and batch_embeddings:
                self._dimension = len(batch_embeddings[0])
        
        return np.array(embeddings, dtype=np.float32)
    
    def _embed_sentence_transformers(self, texts: list[str]) -> np.ndarray:
        """Создает эмбеддинги через sentence-transformers."""
        # Модели e5 требуют специального форматирования
        # Но если тексты уже отформатированы (с префиксом "passage:" или "query:"),
        # то передаем их как есть
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Нормализация для e5 моделей
        )
        return embeddings.astype(np.float32)
    
    def get_dimension(self) -> int:
        """Возвращает размерность эмбеддингов."""
        if self._dimension is None:
            # Для OpenAI определяем при первом запросе
            test_embedding = self.embed(["test"])
            self._dimension = test_embedding.shape[1]
        return self._dimension


def format_text_for_e5(text: str, prefix: str = "passage: ") -> str:
    """Форматирует текст для моделей e5 (passage/query)."""
    text = re.sub(r"\s+", " ", text.strip())
    return f"{prefix}{text}"


def get_embedder(
    vectorization_type: Literal["dense"],
    config: dict,
    corpus: list[str] | None = None,  # Не используется, оставлен для совместимости
) -> BaseEmbedder:
    """Создает dense эмбеддер.
    
    Args:
        vectorization_type: Тип векторизации (только 'dense').
        config: Словарь с конфигурацией.
        corpus: Не используется, оставлен для совместимости.
    
    Returns:
        Экземпляр DenseEmbedder.
    """
    if vectorization_type != "dense":
        raise ValueError(
            f"Only 'dense' embeddings are supported. "
            f"Got: {vectorization_type}"
        )
    
    embeddings_config = config.get("embeddings", {})
    dense_config = embeddings_config.get("dense", {})
    
    return DenseEmbedder(
        model_name=dense_config.get(
            "model",
            "intfloat/multilingual-e5-base",
        ),
        batch_size=dense_config.get("batch_size", 32),
        device=dense_config.get("device", "cpu"),
        openai_api_key=dense_config.get("openai_api_key") or os.getenv("OPENAI_API_KEY"),
        openai_base_url=dense_config.get("openai_base_url"),
    )
