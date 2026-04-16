"""Persistent FAISS indexing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rag_chatbot.schemas import DocumentChunk
from rag_chatbot.utils import read_json, write_json


@dataclass(slots=True)
class IndexArtifacts:
    """In-memory and on-disk artifacts needed for retrieval."""

    backend: str
    index: Any
    metadata: list[dict[str, Any]]


def build_vector_index(chunks: list[DocumentChunk], embeddings: np.ndarray) -> IndexArtifacts:
    """Build a vector index over normalized embeddings using FAISS when available."""

    if not chunks:
        raise ValueError("Cannot build an index without chunks.")
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk count and embedding count must match.")

    try:
        import faiss
    except ImportError:
        index = SimpleVectorIndex(embeddings)
        backend = "numpy"
    else:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        backend = "faiss"

    metadata = [chunk.to_dict() for chunk in chunks]
    return IndexArtifacts(backend=backend, index=index, metadata=metadata)


def build_faiss_index(chunks: list[DocumentChunk], embeddings: np.ndarray) -> IndexArtifacts:
    """Backward-compatible alias for older imports."""

    return build_vector_index(chunks, embeddings)


def save_index(artifacts: IndexArtifacts, index_path: Path, metadata_path: Path) -> None:
    """Persist the FAISS index and matching chunk metadata to disk."""

    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    if artifacts.backend == "faiss":
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("FAISS backend was selected, but the faiss package is not installed.") from exc
        faiss.write_index(artifacts.index, str(index_path))
    elif artifacts.backend == "numpy":
        with index_path.open("wb") as index_file:
            np.save(index_file, artifacts.index.vectors)
    else:
        raise ValueError(f"Unsupported index backend: {artifacts.backend}")

    write_json(
        metadata_path,
        {
            "backend": artifacts.backend,
            "chunks": artifacts.metadata,
        },
    )


def load_index(index_path: Path, metadata_path: Path) -> IndexArtifacts:
    """Load a saved FAISS index and its serialized metadata."""

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index was not found: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Index metadata file was not found: {metadata_path}")

    metadata_payload = read_json(metadata_path)
    if isinstance(metadata_payload, list):
        backend = "faiss"
        metadata = metadata_payload
    else:
        backend = str(metadata_payload.get("backend", "faiss"))
        metadata = list(metadata_payload.get("chunks", []))

    if backend == "faiss":
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "This index was saved with FAISS. Install a FAISS package or rebuild the index on this machine."
            ) from exc
        index = faiss.read_index(str(index_path))
    elif backend == "numpy":
        with index_path.open("rb") as index_file:
            vectors = np.load(index_file).astype("float32")
        index = SimpleVectorIndex(vectors)
    else:
        raise ValueError(f"Unsupported stored index backend: {backend}")

    return IndexArtifacts(backend=backend, index=index, metadata=metadata)


@dataclass(slots=True)
class SimpleVectorIndex:
    """Small NumPy-based fallback index for environments without FAISS."""

    vectors: np.ndarray

    def __post_init__(self) -> None:
        self.vectors = self.vectors.astype("float32")

    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return top-k inner-product matches using normalized embeddings."""

        if self.vectors.size == 0:
            empty_scores = np.empty((len(query_vectors), 0), dtype="float32")
            empty_indices = np.empty((len(query_vectors), 0), dtype="int64")
            return empty_scores, empty_indices

        scores = query_vectors @ self.vectors.T
        top_k = min(k, self.vectors.shape[0])
        sorted_indices = np.argsort(-scores, axis=1)[:, :top_k]
        sorted_scores = np.take_along_axis(scores, sorted_indices, axis=1)
        return sorted_scores.astype("float32"), sorted_indices.astype("int64")
