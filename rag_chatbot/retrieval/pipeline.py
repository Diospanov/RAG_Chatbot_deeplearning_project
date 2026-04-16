"""End-to-end chunking and indexing pipeline for retrieval."""

from __future__ import annotations

from rag_chatbot.config import AppConfig
from rag_chatbot.retrieval.chunking import chunk_documents, load_processed_documents, save_chunks
from rag_chatbot.retrieval.embeddings import EmbeddingService
from rag_chatbot.retrieval.indexing import build_vector_index, save_index
from rag_chatbot.schemas import DocumentChunk


def build_retrieval_pipeline(
    config: AppConfig | None = None,
    strategy: str = "sentence",
    chunk_size: int | None = None,
    overlap_ratio: float | None = None,
) -> list[DocumentChunk]:
    """Create chunks, embed them, build a FAISS index, and save all artifacts."""

    config = config or AppConfig()
    config.ensure_directories()
    if not config.processed_documents_path.exists():
        from rag_chatbot.ingest.pipeline import run_ingestion_pipeline

        run_ingestion_pipeline(config=config)

    documents = load_processed_documents(config.processed_documents_path)
    chunk_size = chunk_size or config.default_chunk_size
    overlap_ratio = overlap_ratio if overlap_ratio is not None else config.default_chunk_overlap

    chunks = chunk_documents(
        documents=documents,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap_ratio=overlap_ratio,
    )
    save_chunks(chunks, config.chunk_output_path(strategy))

    embedding_service = EmbeddingService(config.embedding_model_name)
    embeddings = embedding_service.embed_texts([chunk.text for chunk in chunks])
    artifacts = build_vector_index(chunks, embeddings)
    save_index(
        artifacts=artifacts,
        index_path=config.faiss_index_path(strategy),
        metadata_path=config.index_metadata_path(strategy),
    )

    return chunks
