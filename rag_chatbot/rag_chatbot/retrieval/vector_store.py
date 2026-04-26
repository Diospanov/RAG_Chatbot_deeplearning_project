"""
retrieval/vector_store.py
-------------------------
ChromaDB-backed vector store.

Each stored record contains:
  - chunk text
  - embedding vector
  - source document metadata (source, title, date, file_type, chunk_index)
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100   # ChromaDB batch add limit


class VectorStore:
    """Thin wrapper around a ChromaDB collection."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
    ):
        import chromadb

        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "rag_docs")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
        logger.info(
            "VectorStore: collection=%r persist=%r count=%d",
            self.collection_name,
            self.persist_dir,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Embed and store a list of chunk dicts produced by the chunker."""
        from retrieval.embedder import embed_texts
        from tqdm import tqdm

        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        logger.info("Embedding %d chunks…", len(texts))
        embeddings = embed_texts(texts, model_name=self.embedding_model)

        # Build flat metadata dicts (ChromaDB requires string/int/float/bool values)
        ids, metas, docs, embeds = [], [], [], []
        for chunk, emb in zip(chunks, embeddings):
            uid = _chunk_id(chunk)
            flat_meta = _flatten_metadata(chunk.get("metadata", {}))
            ids.append(uid)
            metas.append(flat_meta)
            docs.append(chunk["text"])
            embeds.append(emb)

        # Add in batches
        for i in tqdm(range(0, len(ids), _BATCH_SIZE), desc="Indexing"):
            self._collection.upsert(
                ids=ids[i : i + _BATCH_SIZE],
                embeddings=embeds[i : i + _BATCH_SIZE],
                documents=docs[i : i + _BATCH_SIZE],
                metadatas=metas[i : i + _BATCH_SIZE],
            )

        logger.info("Indexed %d chunks into %r", len(ids), self.collection_name)

    def clear(self) -> None:
        """Delete all documents from the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Cleared collection %r", self.collection_name)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return the top-k most similar chunks.

        Each result:
            {
                "text": str,
                "metadata": dict,
                "score": float,   # cosine similarity (higher = better)
            }
        """
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self._collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text": doc,
                "metadata": meta,
                "score": 1.0 - dist,   # ChromaDB returns L2/cosine distance
            })

        return output

    def count(self) -> int:
        return self._collection.count()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_id(chunk: Dict[str, Any]) -> str:
    """Deterministic ID based on source + chunk_index."""
    meta = chunk.get("metadata", {})
    source = str(meta.get("source", ""))
    idx = str(meta.get("chunk_index", 0))
    raw = f"{source}::{idx}::{chunk['text'][:80]}"
    return hashlib.md5(raw.encode()).hexdigest()


def _flatten_metadata(meta: Dict) -> Dict:
    """Ensure all metadata values are ChromaDB-compatible primitives."""
    flat = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            flat[k] = v
        else:
            flat[k] = str(v)
    return flat
