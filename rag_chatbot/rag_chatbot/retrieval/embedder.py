"""
retrieval/embedder.py
---------------------
Text embedding using sentence-transformers.

Default model: all-MiniLM-L6-v2
  - 384-dimensional dense vectors
  - Fine-tuned BERT variant optimised for semantic similarity
  - Runs on CPU without a GPU

Can be swapped for any sentence-transformers compatible model via the
EMBEDDING_MODEL env var.
"""

from __future__ import annotations

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict = {}


def get_embedder(model_name: str | None = None):
    """Return (and cache) a SentenceTransformer embedder."""
    from sentence_transformers import SentenceTransformer

    model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    if model_name not in _MODEL_CACHE:
        logger.info("Loading embedding model: %s", model_name)
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)

    return _MODEL_CACHE[model_name]


def embed_texts(
    texts: List[str],
    model_name: str | None = None,
    batch_size: int = 64,
    show_progress: bool = True,
) -> List[List[float]]:
    """
    Encode a list of strings into dense vectors.

    Returns a list of float lists (not numpy arrays) for JSON/ChromaDB
    compatibility.
    """
    if not texts:
        return []

    embedder = get_embedder(model_name)
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,      # L2-normalize for cosine similarity
    )
    return embeddings.tolist()


def embed_query(query: str, model_name: str | None = None) -> List[float]:
    """Embed a single query string."""
    return embed_texts([query], model_name=model_name, show_progress=False)[0]
