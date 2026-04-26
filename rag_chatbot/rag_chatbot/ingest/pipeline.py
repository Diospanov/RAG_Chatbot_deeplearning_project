"""
ingest/pipeline.py
------------------
Orchestrates the full ingestion pipeline:
  Load → Chunk → Embed → Index

Run as a script:
    python -m ingest.pipeline --docs-dir data/docs --chunk-strategy sentence
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(
    docs_dir: str | Path,
    chunk_strategy: str = "sentence",
    chunk_size: int = 256,
    chunk_overlap: int = 50,
    clear_existing: bool = False,
) -> int:
    """
    Run ingestion end-to-end. Returns the number of chunks indexed.
    """
    from ingest.loader import load_directory
    from ingest.chunker import chunk_documents
    from retrieval.vector_store import VectorStore

    # 1. Load documents
    logger.info("=== STEP 1: Loading documents from %s ===", docs_dir)
    documents = load_directory(docs_dir)
    if not documents:
        logger.error("No documents found in %s", docs_dir)
        return 0
    logger.info("Loaded %d documents", len(documents))

    # 2. Chunk
    logger.info("=== STEP 2: Chunking (strategy=%s) ===", chunk_strategy)
    chunks = chunk_documents(
        documents,
        strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    logger.info("Produced %d chunks", len(chunks))

    # 3. Embed + Index
    logger.info("=== STEP 3: Embedding and indexing ===")
    store = VectorStore()
    if clear_existing:
        store.clear()

    store.add_chunks(chunks)
    count = store.count()
    logger.info("Vector store now contains %d chunks", count)

    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ingest documents into the RAG vector store")
    p.add_argument(
        "--docs-dir",
        default="data/docs",
        help="Directory containing documents to ingest (default: data/docs)",
    )
    p.add_argument(
        "--chunk-strategy",
        choices=["fixed", "sentence"],
        default="sentence",
        help="Chunking strategy (default: sentence)",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Target chunk size in tokens (default: 256)",
    )
    p.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in tokens (default: 50)",
    )
    p.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing vector store before ingesting",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    n = run_pipeline(
        docs_dir=args.docs_dir,
        chunk_strategy=args.chunk_strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        clear_existing=args.clear,
    )
    if n == 0:
        sys.exit(1)
    logger.info("Pipeline complete. %d chunks indexed.", n)
