"""Evaluation pipeline for retrieval and grounded answer generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_chatbot.config import AppConfig
from rag_chatbot.eval.metrics import (
    answer_relevance_placeholder,
    faithfulness_placeholder,
    precision_at_k,
)
from rag_chatbot.generation.service import GroundedGenerator
from rag_chatbot.schemas import GenerationResult, RetrievalResult
from rag_chatbot.utils import read_json


@dataclass(slots=True)
class EvaluationRecord:
    """Evaluation result for one QA example."""

    question: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    precision_at_5: float
    answer: str
    citations: list[str]
    provider: str
    model: str
    faithfulness: dict[str, Any]
    answer_relevance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the record into JSON-serializable data."""

        return asdict(self)


def evaluate_retrieval(
    dataset_path: Path | None = None,
    config: AppConfig | None = None,
    strategy: str = "sentence",
    top_k: int = 5,
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Run retrieval evaluation over a dataset and append a JSONL experiment log."""

    from rag_chatbot.retrieval.retriever import FaissRetriever

    config = config or AppConfig()
    dataset_path = dataset_path or config.evaluation_dataset_path
    records_data = read_json(dataset_path)
    if not isinstance(records_data, list) or not records_data:
        raise ValueError(f"Evaluation dataset is empty or malformed: {dataset_path}")

    metric_k = 5
    retrieve_k = max(top_k, metric_k)

    retriever = FaissRetriever.from_disk(config=config, strategy=strategy)
    generator = GroundedGenerator(config=config)

    records: list[EvaluationRecord] = []
    for item in records_data:
        question = str(item["question"]).strip()
        expected_sources = [str(source) for source in item.get("expected_sources", [])]
        retrieved_context = retriever.retrieve(question, top_k=retrieve_k)
        generation = generator.generate(question=question, retrieved_context=retrieved_context)
        records.append(
            _build_record(
                question=question,
                expected_sources=expected_sources,
                retrieved_context=retrieved_context,
                generation=generation,
                metric_k=metric_k,
            )
        )

    average_precision = sum(record.precision_at_5 for record in records) / len(records)
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path.resolve()),
        "strategy": strategy,
        "top_k": retrieve_k,
        "num_examples": len(records),
        "average_precision_at_5": average_precision,
        "records": [record.to_dict() for record in records],
    }
    _append_experiment_log(log_path or config.experiment_log_path, summary)
    return summary


def _build_record(
    question: str,
    expected_sources: list[str],
    retrieved_context: list[RetrievalResult],
    generation: GenerationResult,
    metric_k: int,
) -> EvaluationRecord:
    """Build one evaluation record from retrieval and generation outputs."""

    retrieved_sources = [
        str(result.metadata.get("source_filename", "unknown_source")) for result in retrieved_context[:metric_k]
    ]
    return EvaluationRecord(
        question=question,
        expected_sources=expected_sources,
        retrieved_sources=retrieved_sources,
        precision_at_5=precision_at_k(retrieved_context, expected_sources, k=metric_k),
        answer=generation.answer,
        citations=generation.citations,
        provider=generation.provider,
        model=generation.model,
        faithfulness=faithfulness_placeholder(generation.answer, retrieved_context),
        answer_relevance=answer_relevance_placeholder(question, generation.answer),
    )


def _append_experiment_log(path: Path, summary: dict[str, Any]) -> None:
    """Append one JSON object per line for easy experiment tracking."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log_file:
        import json

        log_file.write(json.dumps(summary, ensure_ascii=False) + "\n")
