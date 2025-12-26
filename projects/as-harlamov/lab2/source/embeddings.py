import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import json

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from config import Config


class EmbeddingGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.dense_model = None
        self.bm25 = None

        if self.config.embeddings.type in ["dense", "hybrid"]:
            self.dense_model = SentenceTransformer(self.config.embeddings.model)

    def get_max_sequence_length(self) -> Optional[int]:
        if self.config.embeddings.type == "sparse":
            return None

        if hasattr(self.dense_model, 'get_max_seq_length'):
            return self.dense_model.get_max_seq_length()
        elif hasattr(self.dense_model, 'max_seq_length'):
            return self.dense_model.max_seq_length
        elif hasattr(self.dense_model, 'tokenizer') and hasattr(self.dense_model.tokenizer, 'model_max_length'):
            return self.dense_model.tokenizer.model_max_length

        return 512

    def count_tokens(self, text: str) -> int:
        encoded = self.dense_model.tokenizer.encode(text, add_special_tokens=False)
        return encoded.shape[0] if hasattr(encoded, 'shape') else len(encoded)

    def generate_dense_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.dense_model.encode(texts, show_progress_bar=False)

    def generate_sparse_embeddings(self, query: str, corpus: List[str]) -> List[float]:
        if not self.bm25:
            tokenized_corpus = [doc.split() for doc in corpus]
            self.bm25 = BM25Okapi(
                tokenized_corpus,
                k1=self.config.embeddings.bm25_k1,
                b=self.config.embeddings.bm25_b,
            )

        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        return scores.tolist()

    def generate_embeddings(
        self,
        chunks: List[Dict[str, Any]],
        corpus_texts: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        texts = [chunk["text"] for chunk in chunks]

        if corpus_texts is None:
            corpus_texts = texts

        results = []

        if self.config.embeddings.type == "dense":
            embeddings = self.generate_dense_embeddings(texts)
            for i, chunk in enumerate(chunks):
                results.append(
                    {
                        **chunk,
                        "embedding": embeddings[i].tolist(),
                        "embedding_type": "dense",
                    },
                )

        elif self.config.embeddings.type == "sparse":
            for chunk in chunks:
                results.append(
                    {
                        **chunk,
                        "embedding_type": "sparse",
                        "tokenized_text": chunk["text"].split(),
                    },
                )

        elif self.config.embeddings.type == "hybrid":
            dense_embeddings = self.generate_dense_embeddings(texts)
            for i, chunk in enumerate(chunks):
                results.append(
                    {
                        **chunk,
                        "embedding": dense_embeddings[i].tolist(),
                        "embedding_type": "hybrid",
                        "tokenized_text": chunk["text"].split(),
                    },
                )

        return results

    def save_embeddings(self, chunks_with_embeddings: List[Dict[str, Any]], output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)

        json_data = []
        for chunk in chunks_with_embeddings:
            chunk_copy = chunk.copy()
            if "embedding" in chunk_copy:
                chunk_copy["embedding"] = chunk["embedding"]
            json_data.append(chunk_copy)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

    def load_embeddings(self, input_path: Path) -> List[Dict[str, Any]]:
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)
