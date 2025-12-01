import os
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch


class DenseEmbedder:
    
    def __init__(self, config):
        self.config = config
        self.model_name = config['model']
        self.batch_size = config.get('batch_size', 32)
        self.device = config.get('device', 'cpu')
        self.dimension = config.get('dimension')
        self._load_model()
    
    def _load_model(self):
        self._load_sentence_transformer()
    
    def _load_sentence_transformer(self):
        model_mapping = {
            'e5-large-v2': 'intfloat/e5-large-v2',
            'e5-base-v2': 'intfloat/e5-base-v2',
            'bge-base-en-v1.5': 'BAAI/bge-base-en-v1.5',
            'bge-large-en-v1.5': 'BAAI/bge-large-en-v1.5',
            'bge-m3': 'BAAI/bge-m3',
        }
            
        model_path = model_mapping.get(self.model_name, self.model_name)
        self.model = SentenceTransformer(model_path, device=self.device)
    
    def _embed_with_sentence_transformer(self, texts):
        # Для E5 моделей нужен префикс
        if 'e5' in self.model_name.lower():
            texts = [f"passage: {text}" for text in texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def embed_texts(self, texts):

        embeddings = self._embed_with_sentence_transformer(texts)

        return {
            'dense': embeddings,
            'model': self.model_name,
            'dimension': embeddings.shape[1],
            'count': len(embeddings)
        }