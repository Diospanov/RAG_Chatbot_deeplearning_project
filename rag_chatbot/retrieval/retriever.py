"""Retriever built on top of embeddings and a persisted FAISS index."""

from __future__ import annotations

from rag_chatbot.config import AppConfig
from rag_chatbot.retrieval.embeddings import EmbeddingService
from rag_chatbot.retrieval.indexing import load_index
from rag_chatbot.schemas import RetrievalResult


class FaissRetriever:
    """Query a saved FAISS index and return top-k matching chunks."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        index,
        metadata: list[dict],
    ) -> None:
        self.embedding_service = embedding_service
        self.index = index
        self.metadata = metadata

    @classmethod
    def from_disk(cls, config: AppConfig | None = None, strategy: str = "sentence") -> "FaissRetriever":
        """Restore a retriever from previously saved index artifacts."""

        config = config or AppConfig()
        artifacts = load_index(
            index_path=config.faiss_index_path(strategy),
            metadata_path=config.index_metadata_path(strategy),
        )
        embedding_service = EmbeddingService(config.embedding_model_name)
        return cls(embedding_service=embedding_service, index=artifacts.index, metadata=artifacts.metadata)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve the top-k most similar chunks for a user query."""

        if top_k <= 0:
            raise ValueError("top_k must be at least 1.")
        if not self.metadata:
            return []

        query_embedding = self.embedding_service.embed_query(query)
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.metadata)))

        results: list[RetrievalResult] = []
        for rank, (score, index_position) in enumerate(zip(scores[0], indices[0]), start=1):
            if index_position < 0:
                continue

            chunk_payload = self.metadata[index_position]
            result_metadata = {
                "source_filename": chunk_payload["source_filename"],
                "source_path": chunk_payload["source_path"],
                "source_type": chunk_payload["source_type"],
                "title": chunk_payload["title"],
                "date": chunk_payload["date"],
                "chunk_index": chunk_payload["chunk_index"],
                "strategy": chunk_payload["strategy"],
                "token_count": chunk_payload["token_count"],
                **chunk_payload.get("metadata", {}),
            }
            results.append(
                RetrievalResult(
                    rank=rank,
                    score=float(score),
                    chunk_id=chunk_payload["chunk_id"],
                    document_id=chunk_payload["document_id"],
                    text=chunk_payload["text"],
                    metadata=result_metadata,
                )
            )

        return results
