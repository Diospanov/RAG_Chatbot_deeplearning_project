"""
ingest/loader.py
----------------
Multi-format document loader.
Supports: PDF, plain text, Markdown, HTML, DOCX.

Each loaded document is returned as a dict:
    {
        "text": str,          # raw extracted text
        "metadata": {
            "source": str,    # filename or URL
            "title": str,     # document title (best-effort)
            "date": str,      # modification date or extracted date
            "file_type": str, # pdf | txt | md | html | docx
        }
    }
"""

from __future__ import annotations

import os
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_documents(paths: List[str | Path]) -> List[Dict[str, Any]]:
    """Load a list of file paths and return a list of document dicts."""
    documents = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            logger.warning("File not found: %s", path)
            continue
        try:
            doc = _load_single(path)
            if doc and doc["text"].strip():
                documents.append(doc)
                logger.info("Loaded %s (%d chars)", path.name, len(doc["text"]))
            else:
                logger.warning("Empty content from %s", path.name)
        except Exception as exc:
            logger.error("Failed to load %s: %s", path.name, exc)
    return documents


def load_directory(
    directory: str | Path,
    extensions: tuple[str, ...] = (".pdf", ".txt", ".md", ".html", ".htm", ".docx"),
    recursive: bool = True,
) -> List[Dict[str, Any]]:
    """Walk a directory and load all supported documents."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    paths = []
    glob_fn = directory.rglob if recursive else directory.glob
    for ext in extensions:
        paths.extend(glob_fn(f"*{ext}"))

    paths = sorted(set(paths))
    logger.info("Found %d files in %s", len(paths), directory)
    return load_documents(paths)


# ---------------------------------------------------------------------------
# Per-format loaders
# ---------------------------------------------------------------------------

def _load_single(path: Path) -> Dict[str, Any]:
    ext = path.suffix.lower()
    loaders = {
        ".pdf":  _load_pdf,
        ".txt":  _load_text,
        ".md":   _load_markdown,
        ".html": _load_html,
        ".htm":  _load_html,
        ".docx": _load_docx,
    }
    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader(path)


def _base_metadata(path: Path, file_type: str) -> Dict[str, str]:
    mtime = os.path.getmtime(path)
    return {
        "source": str(path),
        "title": path.stem.replace("_", " ").replace("-", " ").title(),
        "date": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d"),
        "file_type": file_type,
    }


# --- PDF ---

def _load_pdf(path: Path) -> Dict[str, Any]:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Install pypdf: pip install pypdf")

    reader = PdfReader(str(path))
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    text = "\n\n".join(text_parts)

    meta = _base_metadata(path, "pdf")
    # Attempt to extract title from PDF metadata
    if reader.metadata and reader.metadata.title:
        meta["title"] = reader.metadata.title
    if reader.metadata and reader.metadata.creation_date:
        try:
            meta["date"] = reader.metadata.creation_date.strftime("%Y-%m-%d")
        except Exception:
            pass

    return {"text": text, "metadata": meta}


# --- Plain text ---

def _load_text(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {"text": text, "metadata": _base_metadata(path, "txt")}


# --- Markdown ---

def _load_markdown(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8", errors="replace")

    # Extract title from first H1
    title_match = re.search(r"^#\s+(.+)", raw, re.MULTILINE)
    meta = _base_metadata(path, "md")
    if title_match:
        meta["title"] = title_match.group(1).strip()

    # Strip markdown syntax for plain text (basic)
    try:
        import markdown
        from bs4 import BeautifulSoup
        html = markdown.markdown(raw)
        text = BeautifulSoup(html, "html.parser").get_text(separator="\n")
    except ImportError:
        # Fallback: strip common markdown symbols
        text = re.sub(r"[#*_`~>]", "", raw)

    return {"text": text, "metadata": meta}


# --- HTML ---

def _load_html(path: Path) -> Dict[str, Any]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Install beautifulsoup4: pip install beautifulsoup4")

    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts and styles
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    meta = _base_metadata(path, "html")
    title_tag = soup.find("title")
    if title_tag:
        meta["title"] = title_tag.get_text().strip()

    return {"text": text, "metadata": meta}


# --- DOCX ---

def _load_docx(path: Path) -> Dict[str, Any]:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)

    meta = _base_metadata(path, "docx")
    # First paragraph is often the title
    if paragraphs:
        first = paragraphs[0].strip()
        if len(first) < 120:
            meta["title"] = first

    return {"text": text, "metadata": meta}
