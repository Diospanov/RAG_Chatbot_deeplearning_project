"""Command-line entry point for the RAG chatbot project."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_chatbot.config import AppConfig
from rag_chatbot.utils import normalize_text


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the project."""

    parser = argparse.ArgumentParser(description="Run the RAG chatbot project pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Load PDF and DOCX files from data/raw.")
    ingest_parser.set_defaults(handler=handle_ingest)

    index_parser = subparsers.add_parser("index", help="Chunk documents and build the FAISS index.")
    _add_index_arguments(index_parser)
    index_parser.set_defaults(handler=handle_index)

    ask_parser = subparsers.add_parser("ask", help="Ask a grounded question against the indexed corpus.")
    ask_parser.add_argument("question", help="User question to answer from retrieved context.")
    _add_query_arguments(ask_parser)
    ask_parser.set_defaults(handler=handle_ask)

    eval_parser = subparsers.add_parser("evaluate", help="Run retrieval evaluation on a QA dataset.")
    _add_query_arguments(eval_parser)
    eval_parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to the evaluation dataset JSON file.",
    )
    eval_parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional path for the JSONL experiment log.",
    )
    eval_parser.set_defaults(handler=handle_evaluate)

    return parser


def _add_index_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach chunking configuration arguments to a parser."""

    parser.add_argument(
        "--strategy",
        choices=["sentence", "fixed"],
        default="sentence",
        help="Chunking strategy used to build the retrieval index.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size in approximate tokens. Must be between 100 and 512.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.15,
        help="Chunk overlap ratio. Must be between 0.10 and 0.25.",
    )


def _add_query_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach retrieval configuration arguments to a parser."""

    _add_index_arguments(parser)
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve for each question.",
    )


def handle_ingest(args: argparse.Namespace) -> int:
    """Run document ingestion and print a short summary."""

    from rag_chatbot.ingest.pipeline import run_ingestion_pipeline

    config = AppConfig()
    documents = run_ingestion_pipeline(config=config)
    print(f"Ingested {len(documents)} document(s) into {config.processed_documents_path}.")
    return 0


def handle_index(args: argparse.Namespace) -> int:
    """Build chunk artifacts and the FAISS index."""

    from rag_chatbot.retrieval.pipeline import build_retrieval_pipeline

    config = AppConfig()
    _ensure_processed_documents(config)
    chunks = build_retrieval_pipeline(
        config=config,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap_ratio=args.overlap,
    )
    print(
        f"Built {len(chunks)} chunk(s) using '{args.strategy}' strategy. "
        f"Index saved to {config.faiss_index_path(args.strategy)}."
    )
    return 0


def handle_ask(args: argparse.Namespace) -> int:
    """Answer one user question against the indexed corpus."""

    from rag_chatbot.generation.pipeline import answer_question

    config = AppConfig()
    _ensure_index_ready(
        config=config,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    result = answer_question(
        question=args.question,
        config=config,
        strategy=args.strategy,
        top_k=args.top_k,
    )

    print("\nAnswer")
    print(result.answer)
    if result.citations:
        print("\nCitations")
        for citation in result.citations:
            print(f"- {citation}")
    if result.context:
        print("\nRetrieved Chunks")
        for item in result.context:
            source = item.metadata.get("source_filename", "unknown_source")
            print(f"- Rank {item.rank} | Score {item.score:.4f} | {source}")
            print(f"  {normalize_text(item.text)[:220]}")
    return 0


def handle_evaluate(args: argparse.Namespace) -> int:
    """Run the evaluation pipeline and print the summary."""

    from rag_chatbot.eval.pipeline import evaluate_retrieval

    config = AppConfig()
    _ensure_index_ready(
        config=config,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    summary = evaluate_retrieval(
        dataset_path=args.dataset,
        config=config,
        strategy=args.strategy,
        top_k=args.top_k,
        log_path=args.log_path,
    )
    print("\nEvaluation Summary")
    print(f"Examples: {summary['num_examples']}")
    print(f"Average precision@5: {summary['average_precision_at_5']:.4f}")
    print(f"Log written to: {args.log_path or config.experiment_log_path}")
    return 0


def _ensure_processed_documents(config: AppConfig) -> None:
    """Run ingestion automatically when the processed corpus is missing."""

    from rag_chatbot.ingest.pipeline import run_ingestion_pipeline

    if config.processed_documents_path.exists():
        return
    run_ingestion_pipeline(config=config)


def _ensure_index_ready(
    config: AppConfig,
    strategy: str,
    chunk_size: int,
    overlap: float,
) -> None:
    """Ensure retrieval artifacts exist before answering or evaluating."""

    if config.faiss_index_path(strategy).exists() and config.index_metadata_path(strategy).exists():
        return
    handle_index(
        argparse.Namespace(
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    )


def main() -> int:
    """Parse command-line arguments and dispatch the selected command."""

    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
