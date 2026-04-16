"""End-to-end ingestion pipeline for raw source documents."""

from __future__ import annotations

from pathlib import Path

from rag_chatbot.config import AppConfig
from rag_chatbot.ingest.loaders import load_documents
from rag_chatbot.schemas import SourceDocument
from rag_chatbot.utils import write_json


def run_ingestion_pipeline(config: AppConfig | None = None, output_path: Path | None = None) -> list[SourceDocument]:
    """Load supported raw files, normalize them, and save the processed corpus."""

    config = config or AppConfig()
    config.ensure_directories()

    documents = load_documents(
        raw_dir=config.raw_data_dir,
        supported_extensions=config.supported_extensions,
    )

    destination = output_path or config.processed_documents_path
    write_json(destination, [document.to_dict() for document in documents])
    return documents
