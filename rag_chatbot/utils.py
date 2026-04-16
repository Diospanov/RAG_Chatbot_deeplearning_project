"""Small utility helpers shared by project modules."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    """Collapse repeated whitespace while keeping paragraph boundaries readable."""

    cleaned = text.replace("\x00", " ")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def make_document_id(filename: str) -> str:
    """Create a simple stable identifier from a source filename."""

    stem = Path(filename).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return slug or "document"


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with UTF-8 encoding and readable indentation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read JSON content from disk using UTF-8 encoding."""

    return json.loads(path.read_text(encoding="utf-8"))
