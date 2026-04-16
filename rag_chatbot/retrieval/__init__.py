"""Chunking, embedding, indexing, and retrieval tools."""

__all__ = ["FaissRetriever", "build_retrieval_pipeline"]


def __getattr__(name: str):
    """Lazily expose heavier retrieval objects without import-time side effects."""

    if name == "build_retrieval_pipeline":
        from .pipeline import build_retrieval_pipeline

        return build_retrieval_pipeline
    if name == "FaissRetriever":
        from .retriever import FaissRetriever

        return FaissRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
