"""
eval/metrics.py
---------------
Evaluation metrics for the RAG pipeline.

Metrics implemented:
  - precision_at_k   — fraction of retrieved chunks containing the answer
  - recall_at_k      — fraction of relevant passages that were retrieved
  - answer_relevance — semantic similarity of generated answer to ground-truth
  - faithfulness     — fraction of answer sentences grounded in retrieved context
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def precision_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    ground_truth_passage: str,
    k: int = 5,
    threshold: float = 0.5,
) -> float:
    """
    Precision@k: fraction of top-k retrieved chunks that contain the answer.

    A chunk is considered relevant if its text overlaps sufficiently
    with the ground-truth passage (token Jaccard >= threshold).
    """
    top_k = retrieved_chunks[:k]
    if not top_k:
        return 0.0

    gt_tokens = _tokenize(ground_truth_passage)
    relevant = sum(
        1 for c in top_k
        if _jaccard(gt_tokens, _tokenize(c["text"])) >= threshold
    )
    return relevant / len(top_k)


def recall_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    ground_truth_passages: List[str],
    k: int = 5,
    threshold: float = 0.5,
) -> float:
    """
    Recall@k: fraction of ground-truth passages found in top-k results.
    """
    top_k = retrieved_chunks[:k]
    if not ground_truth_passages:
        return 1.0

    found = 0
    for gt in ground_truth_passages:
        gt_tokens = _tokenize(gt)
        for c in top_k:
            if _jaccard(gt_tokens, _tokenize(c["text"])) >= threshold:
                found += 1
                break

    return found / len(ground_truth_passages)


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

def answer_relevance(
    generated_answer: str,
    ground_truth_answer: str,
    embedder=None,
) -> float:
    """
    Semantic similarity between generated and ground-truth answers.
    Uses cosine similarity of sentence embeddings.
    Falls back to token overlap if embedder is unavailable.
    """
    try:
        if embedder is None:
            from retrieval.embedder import get_embedder
            embedder = get_embedder()

        import numpy as np
        a = embedder.encode(generated_answer, normalize_embeddings=True)
        b = embedder.encode(ground_truth_answer, normalize_embeddings=True)
        return float(np.dot(a, b))
    except Exception:
        # Fallback to Jaccard
        return _jaccard(_tokenize(generated_answer), _tokenize(ground_truth_answer))


def faithfulness(
    generated_answer: str,
    context: str,
) -> float:
    """
    Faithfulness: fraction of sentences in the answer that are supported
    by the retrieved context.

    A sentence is considered supported if its key terms appear in the context.
    """
    sentences = _split_sentences(generated_answer)
    if not sentences:
        return 1.0

    ctx_tokens = _tokenize(context)
    supported = 0
    for sent in sentences:
        sent_tokens = _tokenize(sent)
        if not sent_tokens:
            supported += 1
            continue
        # A sentence is supported if >50% of its meaningful tokens appear in context
        meaningful = [t for t in sent_tokens if len(t) > 3]
        if not meaningful:
            supported += 1
            continue
        overlap = sum(1 for t in meaningful if t in ctx_tokens)
        if overlap / len(meaningful) >= 0.5:
            supported += 1

    return supported / len(sentences)


def has_citation(answer: str) -> bool:
    """Check whether the answer contains a source citation."""
    # Match patterns like (Source: X), [Source], [1], etc.
    patterns = [
        r"\(Source:",
        r"\[Source",
        r"\[\d+\]",
        r"\(see .+\)",
    ]
    return any(re.search(p, answer, re.IGNORECASE) for p in patterns)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set:
    """Lowercase word tokens for overlap computation."""
    return set(re.findall(r"\b[a-z]{2,}\b", text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]
