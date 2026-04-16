"""Convenience helpers for generation with retrieved context."""

from __future__ import annotations

from rag_chatbot.config import AppConfig
from rag_chatbot.generation.service import GroundedGenerator
from rag_chatbot.retrieval.retriever import FaissRetriever
from rag_chatbot.schemas import GenerationResult, RetrievalResult


def answer_question(
    question: str,
    retrieved_context: list[RetrievalResult] | None = None,
    retriever: FaissRetriever | None = None,
    config: AppConfig | None = None,
    strategy: str = "sentence",
    top_k: int = 5,
) -> GenerationResult:
    """Retrieve context when needed and generate a grounded answer."""

    config = config or AppConfig()
    generator = GroundedGenerator(config=config)

    if retrieved_context is None:
        retriever = retriever or FaissRetriever.from_disk(config=config, strategy=strategy)
        retrieved_context = retriever.retrieve(question, top_k=top_k)

    return generator.generate(question=question, retrieved_context=retrieved_context)
