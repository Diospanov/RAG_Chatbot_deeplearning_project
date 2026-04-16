"""Grounded generation service with OpenAI support and deterministic fallback behavior."""

from __future__ import annotations

import re

from rag_chatbot.config import AppConfig
from rag_chatbot.generation.prompts import (
    REQUIRED_REFUSAL_SENTENCE,
    build_system_prompt,
    build_user_prompt,
)
from rag_chatbot.schemas import GenerationResult, RetrievalResult


class GroundedGenerator:
    """Generate answers strictly from retrieved context."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.system_prompt = build_system_prompt()

    def generate(self, question: str, retrieved_context: list[RetrievalResult]) -> GenerationResult:
        """Generate a grounded answer or return the required refusal sentence."""

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        citations = _collect_citations(retrieved_context)
        if not retrieved_context or not _context_supports_question(cleaned_question, retrieved_context):
            return self._refusal_result(retrieved_context)

        if self.config.openai_api_key:
            return self._generate_with_openai(cleaned_question, retrieved_context, citations)

        return self._generate_with_fallback(retrieved_context, citations)

    def _generate_with_openai(
        self,
        question: str,
        retrieved_context: list[RetrievalResult],
        citations: list[str],
    ) -> GenerationResult:
        """Use the OpenAI API when credentials are available."""

        try:
            from openai import OpenAI
        except ImportError:
            return self._generate_with_fallback(retrieved_context, citations)

        client = OpenAI(api_key=self.config.openai_api_key)
        try:
            response = client.chat.completions.create(
                model=self.config.openai_model_name,
                temperature=0,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": build_user_prompt(question, retrieved_context)},
                ],
            )
        except Exception:
            return self._generate_with_fallback(retrieved_context, citations)
        answer = (response.choices[0].message.content or "").strip()
        normalized_answer = _normalize_model_answer(answer, citations)
        refusal = normalized_answer == REQUIRED_REFUSAL_SENTENCE
        return GenerationResult(
            answer=normalized_answer,
            citations=[] if refusal else citations,
            provider="openai",
            model=self.config.openai_model_name,
            used_context_count=len(retrieved_context),
            context=retrieved_context,
            refusal=refusal,
        )

    def _generate_with_fallback(
        self,
        retrieved_context: list[RetrievalResult],
        citations: list[str],
    ) -> GenerationResult:
        """Provide a beginner-friendly extractive fallback when no LLM is configured."""

        supporting_points: list[str] = []
        for result in retrieved_context[:3]:
            source = result.metadata.get("source_filename", "unknown_source")
            sentence = _first_meaningful_sentence(result.text)
            if not sentence:
                continue
            supporting_points.append(f"{sentence} [{source}]")

        if not supporting_points:
            return self._refusal_result(retrieved_context)

        answer = " ".join(supporting_points)
        return GenerationResult(
            answer=answer,
            citations=citations,
            provider="fallback",
            model="extractive_stub",
            used_context_count=len(retrieved_context),
            context=retrieved_context,
            refusal=False,
        )

    def _refusal_result(self, retrieved_context: list[RetrievalResult]) -> GenerationResult:
        """Return the exact refusal sentence required by the specification."""

        return GenerationResult(
            answer=REQUIRED_REFUSAL_SENTENCE,
            citations=[],
            provider="rule_based",
            model="strict_refusal",
            used_context_count=len(retrieved_context),
            context=retrieved_context,
            refusal=True,
        )


def _collect_citations(results: list[RetrievalResult]) -> list[str]:
    """Collect unique source filenames from retrieval results in ranked order."""

    citations: list[str] = []
    for result in results:
        source = result.metadata.get("source_filename")
        if source and source not in citations:
            citations.append(source)
    return citations


def _context_supports_question(question: str, retrieved_context: list[RetrievalResult]) -> bool:
    """Use a conservative lexical overlap check to decide whether context is usable."""

    question_terms = {
        token
        for token in re.findall(r"[A-Za-z0-9]{3,}", question.lower())
        if token
        not in {
            "what",
            "when",
            "where",
            "which",
            "who",
            "whom",
            "whose",
            "why",
            "how",
            "does",
            "do",
            "did",
            "the",
            "this",
            "that",
            "with",
            "from",
            "into",
            "about",
            "there",
            "their",
            "have",
            "has",
            "had",
            "your",
            "they",
            "them",
        }
    }
    if not question_terms:
        return bool(retrieved_context)

    combined_context = " ".join(result.text.lower() for result in retrieved_context)
    overlap = [term for term in question_terms if term in combined_context]
    return len(overlap) >= 1


def _normalize_model_answer(answer: str, citations: list[str]) -> str:
    """Keep the model output strict and ensure citations are visible on supported answers."""

    cleaned = answer.strip()
    if not cleaned:
        return REQUIRED_REFUSAL_SENTENCE
    if cleaned == REQUIRED_REFUSAL_SENTENCE:
        return cleaned

    has_citation = any(f"[{citation}]" in cleaned for citation in citations)
    if citations and not has_citation:
        citation_suffix = " ".join(f"[{citation}]" for citation in citations[:2])
        return f"{cleaned} {citation_suffix}".strip()
    return cleaned


def _first_meaningful_sentence(text: str) -> str:
    """Take the first non-trivial sentence-like span from a chunk."""

    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    for part in parts:
        cleaned = part.strip()
        if len(cleaned.split()) >= 5:
            return cleaned
    return text.strip()
