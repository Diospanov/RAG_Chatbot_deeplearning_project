"""Evaluation metrics for retrieval and grounded generation placeholders."""

from __future__ import annotations

from typing import Any

from rag_chatbot.schemas import RetrievalResult


def precision_at_k(results: list[RetrievalResult], expected_sources: list[str], k: int = 5) -> float:
    """Compute precision@k based on retrieved source filenames."""

    if k <= 0:
        raise ValueError("k must be at least 1.")
    if not results:
        return 0.0

    normalized_expected = {source.lower() for source in expected_sources}
    retrieved_top_k = results[:k]
    relevant_count = sum(
        1
        for result in retrieved_top_k
        if result.metadata.get("source_filename", "").lower() in normalized_expected
    )
    return relevant_count / k


def faithfulness_placeholder(answer: str, context: list[RetrievalResult]) -> dict[str, Any]:
    """Return a placeholder object for future faithfulness integration."""

    return {
        "implemented": False,
        "score": None,
        "notes": "Replace this placeholder with RAGAS or a custom factuality check if needed.",
        "answer_length": len(answer),
        "context_count": len(context),
    }


def answer_relevance_placeholder(question: str, answer: str) -> dict[str, Any]:
    """Return a placeholder object for answer relevance evaluation."""

    return {
        "implemented": False,
        "score": None,
        "notes": "Replace this placeholder with RAGAS or a semantic similarity metric if needed.",
        "question_length": len(question),
        "answer_length": len(answer),
    }
