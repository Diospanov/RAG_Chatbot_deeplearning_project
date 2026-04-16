"""Chunking utilities for processed source documents."""

from __future__ import annotations

from collections.abc import Iterable
import re
from pathlib import Path

from rag_chatbot.schemas import DocumentChunk, SourceDocument
from rag_chatbot.utils import normalize_text, read_json, write_json


SUPPORTED_STRATEGIES = {"fixed", "sentence"}


def load_processed_documents(path: Path) -> list[SourceDocument]:
    """Load normalized source documents from JSON output of the ingestion step."""

    if not path.exists():
        raise FileNotFoundError(
            f"Processed documents file was not found: {path}. Run ingestion before chunking."
        )

    payload = read_json(path)
    if not payload:
        raise ValueError(f"Processed documents file is empty: {path}")

    return [SourceDocument.from_dict(item) for item in payload]


def save_chunks(chunks: list[DocumentChunk], path: Path) -> None:
    """Serialize generated chunks for reuse and inspection."""

    write_json(path, [chunk.to_dict() for chunk in chunks])


def chunk_documents(
    documents: Iterable[SourceDocument],
    strategy: str = "sentence",
    chunk_size: int = 256,
    overlap_ratio: float = 0.15,
) -> list[DocumentChunk]:
    """Split source documents into chunk objects while preserving metadata."""

    _validate_chunking_options(strategy=strategy, chunk_size=chunk_size, overlap_ratio=overlap_ratio)

    chunks: list[DocumentChunk] = []
    for document in documents:
        document_chunks = (
            _chunk_document_fixed(document, chunk_size, overlap_ratio)
            if strategy == "fixed"
            else _chunk_document_sentence(document, chunk_size, overlap_ratio)
        )
        chunks.extend(document_chunks)

    if not chunks:
        raise ValueError("Chunking produced no output. Check whether the documents contain readable text.")

    return chunks


def estimate_token_count(text: str) -> int:
    """Approximate token count using word-like segments for simple configuration."""

    return len(text.split())


def _validate_chunking_options(strategy: str, chunk_size: int, overlap_ratio: float) -> None:
    """Validate chunking settings against the project specification."""

    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported chunking strategy '{strategy}'. Use one of {sorted(SUPPORTED_STRATEGIES)}.")
    if not 100 <= chunk_size <= 512:
        raise ValueError("Chunk size must be between 100 and 512 tokens.")
    if not 0.10 <= overlap_ratio <= 0.25:
        raise ValueError("Chunk overlap ratio must be between 0.10 and 0.25.")


def _chunk_document_fixed(
    document: SourceDocument,
    chunk_size: int,
    overlap_ratio: float,
) -> list[DocumentChunk]:
    """Create overlapping fixed-size chunks from document text."""

    words = document.text.split()
    if not words:
        return []

    overlap_tokens = max(1, int(chunk_size * overlap_ratio))
    step = max(1, chunk_size - overlap_tokens)
    chunks: list[DocumentChunk] = []

    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            continue
        chunk_text = normalize_text(" ".join(window))
        chunks.append(_build_chunk(document, len(chunks), "fixed", chunk_text))
        if start + chunk_size >= len(words):
            break

    return chunks


def _chunk_document_sentence(
    document: SourceDocument,
    chunk_size: int,
    overlap_ratio: float,
) -> list[DocumentChunk]:
    """Create sentence-aware chunks that keep neighboring context together."""

    sentence_units = _prepare_sentence_units(document.text, chunk_size)
    if not sentence_units:
        return []

    overlap_tokens = max(1, int(chunk_size * overlap_ratio))
    chunks: list[DocumentChunk] = []
    current_units: list[str] = []
    current_tokens = 0

    for unit in sentence_units:
        unit_tokens = estimate_token_count(unit)
        if current_units and current_tokens + unit_tokens > chunk_size:
            chunks.append(_build_chunk(document, len(chunks), "sentence", " ".join(current_units)))
            current_units, current_tokens = _overlap_tail(current_units, overlap_tokens)
            if current_units and current_tokens + unit_tokens > chunk_size:
                current_units, current_tokens = [], 0

        current_units.append(unit)
        current_tokens += unit_tokens

    if current_units:
        chunks.append(_build_chunk(document, len(chunks), "sentence", " ".join(current_units)))

    return chunks


def _prepare_sentence_units(text: str, chunk_size: int) -> list[str]:
    """Split text into sentence-like units and break very long sentences when necessary."""

    normalized = normalize_text(text)
    if not normalized:
        return []

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n{2,}", normalized)
        if sentence and sentence.strip()
    ]

    units: list[str] = []
    for sentence in sentences:
        if estimate_token_count(sentence) <= chunk_size:
            units.append(sentence)
            continue

        words = sentence.split()
        for start in range(0, len(words), chunk_size):
            piece = normalize_text(" ".join(words[start : start + chunk_size]))
            if piece:
                units.append(piece)

    return units


def _overlap_tail(units: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Keep the tail end of a finished chunk to create overlap with the next chunk."""

    carried_units: list[str] = []
    carried_tokens = 0

    for unit in reversed(units):
        unit_tokens = estimate_token_count(unit)
        carried_units.insert(0, unit)
        carried_tokens += unit_tokens
        if carried_tokens >= overlap_tokens:
            break

    return carried_units, carried_tokens


def _build_chunk(document: SourceDocument, chunk_index: int, strategy: str, chunk_text: str) -> DocumentChunk:
    """Create a chunk object from chunk text and source document metadata."""

    normalized_chunk = normalize_text(chunk_text)
    metadata = {
        "source_filename": document.source_filename,
        "source_path": document.source_path,
        "source_type": document.source_type,
        "title": document.title,
        "date": document.date,
        "chunk_index": chunk_index,
        "chunking_strategy": strategy,
        **document.metadata,
    }

    return DocumentChunk(
        chunk_id=f"{document.document_id}_chunk_{chunk_index}",
        document_id=document.document_id,
        chunk_index=chunk_index,
        strategy=strategy,
        text=normalized_chunk,
        token_count=estimate_token_count(normalized_chunk),
        source_filename=document.source_filename,
        source_path=document.source_path,
        source_type=document.source_type,
        title=document.title,
        date=document.date,
        metadata=metadata,
    )
