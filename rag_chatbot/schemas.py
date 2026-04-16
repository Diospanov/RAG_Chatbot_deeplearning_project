"""Dataclasses used across the project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SourceDocument:
    """Normalized representation of one ingested source document."""

    document_id: str
    source_path: str
    source_filename: str
    source_type: str
    title: str | None
    date: str | None
    text: str
    char_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the document into JSON-serializable data."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourceDocument":
        """Rebuild a document from serialized JSON data."""

        return cls(**payload)


@dataclass(slots=True)
class DocumentChunk:
    """Chunked slice of a source document with preserved metadata."""

    chunk_id: str
    document_id: str
    chunk_index: int
    strategy: str
    text: str
    token_count: int
    source_filename: str
    source_path: str
    source_type: str
    title: str | None
    date: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the chunk into JSON-serializable data."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentChunk":
        """Rebuild a chunk from serialized JSON data."""

        return cls(**payload)


@dataclass(slots=True)
class RetrievalResult:
    """Search result returned by the retriever."""

    rank: int
    score: float
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the retrieval result into JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class GenerationResult:
    """Grounded answer plus supporting context and provider details."""

    answer: str
    citations: list[str]
    provider: str
    model: str
    used_context_count: int
    context: list[RetrievalResult] = field(default_factory=list)
    refusal: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert the generation result into JSON-serializable data."""

        payload = asdict(self)
        payload["context"] = [item.to_dict() for item in self.context]
        return payload
