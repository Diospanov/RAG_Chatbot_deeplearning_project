"""Project configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs) -> bool:
        """Fallback no-op when python-dotenv is not installed yet."""

        return False

PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
load_dotenv(WORKSPACE_ROOT / ".env")
load_dotenv(PACKAGE_ROOT / ".env")


@dataclass(slots=True)
class AppConfig:
    """Centralized filesystem and environment configuration."""

    project_root: Path = field(default_factory=lambda: PACKAGE_ROOT)
    workspace_root: Path = field(default_factory=lambda: WORKSPACE_ROOT)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    index_data_dir: Path = field(init=False)
    processed_documents_path: Path = field(init=False)
    evaluation_dir: Path = field(init=False)
    evaluation_dataset_path: Path = field(init=False)
    experiment_log_path: Path = field(init=False)
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_model_name: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    default_chunk_size: int = 256
    default_chunk_overlap: float = 0.15
    supported_extensions: tuple[str, ...] = (".pdf", ".docx")

    def __post_init__(self) -> None:
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.processed_data_dir = self.project_root / "data" / "processed"
        self.index_data_dir = self.project_root / "data" / "index"
        self.processed_documents_path = self.processed_data_dir / "documents.json"
        self.evaluation_dir = self.project_root / "eval"
        self.evaluation_dataset_path = self.evaluation_dir / "dataset_sample.json"
        self.experiment_log_path = self.evaluation_dir / "experiments" / "experiment_log.jsonl"

    @property
    def openai_api_key(self) -> str | None:
        """Expose the API key for later project modules without hardcoding it."""

        return os.getenv("OPENAI_API_KEY")

    def ensure_directories(self) -> None:
        """Create required directories if they do not exist yet."""

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.index_data_dir.mkdir(parents=True, exist_ok=True)
        (self.evaluation_dir / "experiments").mkdir(parents=True, exist_ok=True)

    def chunk_output_path(self, strategy: str) -> Path:
        """Return the chunk JSON path for the selected chunking strategy."""

        return self.processed_data_dir / f"chunks_{strategy}.json"

    def faiss_index_path(self, strategy: str) -> Path:
        """Return the FAISS index path for the selected chunking strategy."""

        return self.index_data_dir / f"{strategy}_index.faiss"

    def index_metadata_path(self, strategy: str) -> Path:
        """Return the serialized metadata path that matches the FAISS index."""

        return self.index_data_dir / f"{strategy}_metadata.json"
