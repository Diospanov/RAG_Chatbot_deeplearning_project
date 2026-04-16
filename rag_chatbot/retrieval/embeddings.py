"""Embedding helpers based on sentence-transformers."""

from __future__ import annotations

import hashlib
import numpy as np


class EmbeddingService:
    """Thin wrapper around a sentence-transformers embedding model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.backend_name = "sentence-transformers"

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self._model = HashingEmbeddingBackend()
            self.backend_name = "hashing-fallback"
            return

        try:
            self._model = SentenceTransformer(model_name)
        except Exception:
            self._model = HashingEmbeddingBackend()
            self.backend_name = "hashing-fallback"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts and return float32 normalized vectors."""

        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        if self.backend_name == "sentence-transformers":
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embeddings.astype("float32")

        return self._model.encode(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single user query for retrieval."""

        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")

        return self.embed_texts([query])


class HashingEmbeddingBackend:
    """Deterministic local fallback when transformer embeddings are unavailable."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts with a simple normalized hashing trick."""

        vectors = np.zeros((len(texts), self.dimension), dtype="float32")
        for row_index, text in enumerate(texts):
            for token in text.lower().split():
                bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dimension
                vectors[row_index, bucket] += 1.0

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype("float32")
