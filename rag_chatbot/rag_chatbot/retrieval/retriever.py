"""
retrieval/retriever.py
----------------------
Semantic retrieval over the vector store.

Handles multi-source results and deduplication.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Retriever:
    """Top-k dense vector retrieval using cosine similarity."""

    def __init__(
        self,
        top_k: int | None = None,
        embedding_model: str | None = None,
    ):
        from retrieval.vector_store import VectorStore

        self.top_k = top_k or int(os.getenv("TOP_K", "5"))
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self._store = VectorStore(embedding_model=self.embedding_model)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        deduplicate: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Embed the query and return the top-k most relevant chunks.

        Returns a list of dicts:
            {
                "text": str,
                "metadata": { "source", "title", "date", "chunk_index", ... },
                "score": float,
            }
        """
        from retrieval.embedder import embed_query

        k = top_k or self.top_k
        q_emb = embed_query(query, model_name=self.embedding_model)
        results = self._store.query(q_emb, top_k=k)

        if deduplicate:
            results = _deduplicate(results)

        logger.debug("Retrieved %d chunks for query: %.60s…", len(results), query)
        return results

    def is_ready(self) -> bool:
        """Check whether the vector store has any indexed documents."""
        return self._store.count() > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deduplicate(results: List[Dict]) -> List[Dict]:
    """Remove near-duplicate chunks (same source + chunk_index)."""
    seen: set = set()
    out = []
    for r in results:
        meta = r.get("metadata", {})
        key = (meta.get("source", ""), meta.get("chunk_index", -1))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a context block for the LLM prompt.
    Each chunk is labelled with its source document for citation.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        source = meta.get("title") or meta.get("source", f"Document {i}")
        parts.append(f"[{i}] Source: {source}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)
