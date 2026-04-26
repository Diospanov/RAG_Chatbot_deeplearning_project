"""
ingest/chunker.py
-----------------
Two chunking strategies:

  1. fixed   — Fixed-size chunks with token-based overlap
  2. sentence — Sentence-aware (recursive) splitting that respects sentence
               boundaries, then merges/splits to hit target size

Both strategies preserve and propagate source document metadata,
adding chunk_index and char offsets for traceability.
"""

from __future__ import annotations

import re
import logging
from typing import List, Dict, Any, Literal

logger = logging.getLogger(__name__)

ChunkStrategy = Literal["fixed", "sentence"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_documents(
    documents: List[Dict[str, Any]],
    strategy: ChunkStrategy = "sentence",
    chunk_size: int = 256,          # target tokens
    chunk_overlap: int = 50,        # overlap tokens
) -> List[Dict[str, Any]]:
    """
    Chunk a list of loaded documents.

    Returns a flat list of chunk dicts:
        {
            "text": str,
            "metadata": {
                ...original doc metadata...,
                "chunk_index": int,
                "chunk_strategy": str,
            }
        }
    """
    all_chunks = []
    for doc in documents:
        chunks = _chunk_document(doc, strategy, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
        logger.info(
            "%s → %d chunks (%s, size=%d, overlap=%d)",
            doc["metadata"].get("source", "?"),
            len(chunks),
            strategy,
            chunk_size,
            chunk_overlap,
        )
    logger.info("Total chunks: %d", len(all_chunks))
    return all_chunks


# ---------------------------------------------------------------------------
# Per-document chunking
# ---------------------------------------------------------------------------

def _chunk_document(
    doc: Dict[str, Any],
    strategy: ChunkStrategy,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    text = doc["text"]
    meta = dict(doc["metadata"])

    if strategy == "fixed":
        texts = _fixed_chunk(text, chunk_size, chunk_overlap)
    elif strategy == "sentence":
        texts = _sentence_chunk(text, chunk_size, chunk_overlap)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    chunks = []
    for i, t in enumerate(texts):
        chunk_meta = dict(meta)
        chunk_meta["chunk_index"] = i
        chunk_meta["chunk_strategy"] = strategy
        chunk_meta["chunk_size_setting"] = chunk_size
        chunks.append({"text": t, "metadata": chunk_meta})

    return chunks


# ---------------------------------------------------------------------------
# Strategy 1 — Fixed-size with overlap (character-based approximation)
# ---------------------------------------------------------------------------
# We approximate tokens as words for simplicity (avoids requiring tiktoken
# at index time). 1 token ≈ 0.75 words for English, so we scale accordingly.

_WORDS_PER_TOKEN = 0.75


def _tokens_to_words(tokens: int) -> int:
    return max(1, int(tokens * _WORDS_PER_TOKEN))


def _fixed_chunk(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into fixed-size word windows with overlap."""
    words = text.split()
    if not words:
        return []

    step_w = _tokens_to_words(chunk_size)
    overlap_w = _tokens_to_words(chunk_overlap)
    stride = max(1, step_w - overlap_w)

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + step_w, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 20:          # skip trivial fragments
            chunks.append(chunk)
        if end >= len(words):
            break
        start += stride

    return chunks


# ---------------------------------------------------------------------------
# Strategy 2 — Sentence-aware (recursive) splitting
# ---------------------------------------------------------------------------

# Sentence boundary detection (handles common abbreviations)
_SENTENCE_ENDINGS = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+")


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Normalize whitespace first
    text = re.sub(r"\s+", " ", text).strip()
    sentences = _SENTENCE_ENDINGS.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _count_words(text: str) -> int:
    return len(text.split())


def _sentence_chunk(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Sentence-aware chunking:
      1. Split text into paragraphs, then sentences.
      2. Greedily accumulate sentences until chunk_size words are reached.
      3. On overflow, start a new chunk, carrying back `overlap` sentences.
    """
    target_words = _tokens_to_words(chunk_size)
    overlap_words = _tokens_to_words(chunk_overlap)

    # Split into paragraphs first to respect natural breaks
    paragraphs = re.split(r"\n{2,}", text)
    sentences: List[str] = []
    for para in paragraphs:
        sents = _split_into_sentences(para)
        sentences.extend(sents)
        sentences.append("")  # paragraph boundary marker

    sentences = [s for s in sentences if True]  # keep empty as separators

    chunks: List[str] = []
    current_sents: List[str] = []
    current_words = 0

    for sent in sentences:
        sent_words = _count_words(sent) if sent else 0

        if current_words + sent_words > target_words and current_sents:
            # Flush current chunk
            chunk_text = " ".join(s for s in current_sents if s).strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Build overlap: keep sentences from the end until we hit overlap_words
            overlap_sents: List[str] = []
            overlap_w = 0
            for s in reversed(current_sents):
                w = _count_words(s)
                if overlap_w + w > overlap_words:
                    break
                overlap_sents.insert(0, s)
                overlap_w += w

            current_sents = overlap_sents
            current_words = overlap_w

        if sent:
            current_sents.append(sent)
            current_words += sent_words

    # Flush last chunk
    if current_sents:
        chunk_text = " ".join(s for s in current_sents if s).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks
