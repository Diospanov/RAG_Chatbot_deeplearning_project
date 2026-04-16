"""Minimal Streamlit chat interface for the RAG chatbot."""

from __future__ import annotations

import streamlit as st

from rag_chatbot.config import AppConfig
from rag_chatbot.generation.service import GroundedGenerator
from rag_chatbot.retrieval.pipeline import build_retrieval_pipeline
from rag_chatbot.retrieval.retriever import FaissRetriever


DEFAULT_STRATEGY = "sentence"
DEFAULT_TOP_K = 5


def main() -> None:
    """Run the Streamlit chat application."""

    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon=":books:",
        layout="wide",
    )
    _inject_styles()

    config = AppConfig()
    config.ensure_directories()

    _initialize_session_state()

    st.title("RAG Chatbot")
    st.caption("Ask questions about your indexed documents and inspect the retrieved evidence.")

    with st.sidebar:
        st.header("Settings")
        strategy = st.selectbox(
            "Chunking strategy",
            options=["sentence", "fixed"],
            index=0,
            help="Choose the indexed chunking strategy to query.",
        )
        top_k = st.slider(
            "Retrieved chunks",
            min_value=1,
            max_value=5,
            value=DEFAULT_TOP_K,
            help="Number of chunks to retrieve for each question.",
        )
        show_chunks = st.checkbox("Show retrieved chunks", value=True)
        st.divider()
        st.subheader("Index status")
        _render_index_status(config, strategy)

    _render_chat_history(show_chunks=show_chunks)

    question = st.chat_input("Ask a question about the indexed documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving documents and generating a grounded answer..."):
            response = _run_query(question=question, strategy=strategy, top_k=top_k, config=config)
        _render_response(response, show_chunks=show_chunks)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response["answer"],
            "citations": response["citations"],
            "chunks": response["chunks"],
            "provider": response["provider"],
            "model": response["model"],
            "error": response["error"],
        }
    )


def _initialize_session_state() -> None:
    """Create persistent session keys used by the chat interface."""

    if "messages" not in st.session_state:
        st.session_state.messages = []


def _render_chat_history(show_chunks: bool) -> None:
    """Render previous user and assistant messages in the chat transcript."""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if message.get("error"):
                    st.error(message["error"])
                _render_citations(message.get("citations", []))
                if show_chunks and message.get("chunks"):
                    _render_chunks(message["chunks"])
                _render_provider_info(message.get("provider"), message.get("model"))


def _render_index_status(config: AppConfig, strategy: str) -> None:
    """Show whether retrieval artifacts exist and offer a one-click build path."""

    index_ready = _artifacts_exist(config, strategy)
    if index_ready:
        st.success("Vector index and metadata are available.")
        return

    st.warning("No saved retrieval index was found for this strategy.")
    st.caption(
        "You can build the index from the processed documents after running ingestion. "
        "This may take a moment on the first run."
    )
    if st.button("Build retrieval index", use_container_width=True):
        with st.spinner("Building chunks, embeddings, and the vector index..."):
            try:
                chunks = build_retrieval_pipeline(config=config, strategy=strategy)
            except Exception as exc:
                st.error(f"Could not build the retrieval index: {exc}")
            else:
                st.success(f"Index built successfully with {len(chunks)} chunks.")
                _clear_cached_resources()


def _run_query(question: str, strategy: str, top_k: int, config: AppConfig) -> dict[str, object]:
    """Retrieve supporting chunks and generate a grounded answer for the UI."""

    try:
        retriever = _get_retriever(strategy=strategy, _config=config)
        generator = _get_generator(_config=config)
        retrieved_context = retriever.retrieve(question, top_k=top_k)
        result = generator.generate(question=question, retrieved_context=retrieved_context)
    except Exception as exc:
        return {
            "answer": "The app could not answer the question yet.",
            "citations": [],
            "chunks": [],
            "provider": "ui",
            "model": "error",
            "error": str(exc),
        }

    return {
        "answer": result.answer,
        "citations": result.citations,
        "chunks": [item.to_dict() for item in result.context],
        "provider": result.provider,
        "model": result.model,
        "error": None,
    }


@st.cache_resource(show_spinner=False)
def _get_retriever(strategy: str, _config: AppConfig) -> FaissRetriever:
    """Cache the retriever so repeated questions stay fast."""

    return FaissRetriever.from_disk(config=_config, strategy=strategy)


@st.cache_resource(show_spinner=False)
def _get_generator(_config: AppConfig) -> GroundedGenerator:
    """Cache the generation service across interactions."""

    return GroundedGenerator(config=_config)


def _clear_cached_resources() -> None:
    """Clear cached services after rebuilding the index."""

    _get_retriever.clear()
    _get_generator.clear()


def _artifacts_exist(config: AppConfig, strategy: str) -> bool:
    """Check whether the saved retrieval artifacts needed by the UI already exist."""

    return config.faiss_index_path(strategy).exists() and config.index_metadata_path(strategy).exists()


def _render_response(response: dict[str, object], show_chunks: bool) -> None:
    """Render one assistant response, including citations and retrieved context."""

    st.markdown(str(response["answer"]))
    if response.get("error"):
        st.error(str(response["error"]))
    _render_citations(response.get("citations", []))
    if show_chunks and response.get("chunks"):
        _render_chunks(response["chunks"])
    _render_provider_info(str(response.get("provider")), str(response.get("model")))


def _render_citations(citations: list[str]) -> None:
    """Display source citations for the answer."""

    if not citations:
        return

    citation_text = "  \n".join(f"- `{citation}`" for citation in citations)
    st.markdown("**Citations**")
    st.markdown(citation_text)


def _render_chunks(chunks: list[dict]) -> None:
    """Display retrieved chunks inside expandable sections."""

    st.markdown("**Retrieved Chunks**")
    for chunk in chunks:
        source = chunk.get("metadata", {}).get("source_filename", "unknown_source")
        rank = chunk.get("rank", "?")
        score = float(chunk.get("score", 0.0))
        title = chunk.get("metadata", {}).get("title") or "Untitled"
        chunk_index = chunk.get("metadata", {}).get("chunk_index", "?")
        label = f"Rank {rank} | {source} | score {score:.4f}"
        with st.expander(label):
            st.caption(f"Title: {title} | Chunk index: {chunk_index}")
            st.write(chunk.get("text", ""))


def _render_provider_info(provider: str | None, model: str | None) -> None:
    """Show which backend produced the answer."""

    if not provider or not model:
        return
    st.caption(f"Generated with `{provider}` using `{model}`")


def _inject_styles() -> None:
    """Add a few light custom styles so the default app feels more intentional."""

    st.markdown(
        """
        <style>
            :root {
                --sand: #f5efe4;
                --clay: #d17b49;
                --ink: #1f2430;
                --leaf: #5b7c64;
            }
            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(209, 123, 73, 0.18), transparent 30%),
                    linear-gradient(180deg, #fffaf2 0%, #f7f0e5 100%);
            }
            .block-container {
                max-width: 980px;
                padding-top: 2rem;
                padding-bottom: 3rem;
            }
            h1, h2, h3 {
                color: var(--ink);
                letter-spacing: 0.02em;
            }
            [data-testid="stSidebar"] {
                background: rgba(255, 250, 242, 0.92);
                border-right: 1px solid rgba(31, 36, 48, 0.08);
            }
            [data-testid="stChatMessage"] {
                border-radius: 16px;
                border: 1px solid rgba(31, 36, 48, 0.08);
                background: rgba(255, 255, 255, 0.68);
                backdrop-filter: blur(4px);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
