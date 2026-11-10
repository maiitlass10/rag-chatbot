"""Turn uploaded files (PDF, DOCX, TXT, MD) into page-level text."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")


@dataclass(frozen=True)
class Page:
    source: str  # file name
    page: int | None  # 1-based page number (PDF only)
    text: str


class UnsupportedFileError(ValueError):
    pass


def load_bytes(name: str, data: bytes) -> list[Page]:
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        return _load_pdf(name, data)
    if ext == ".docx":
        return _load_docx(name, data)
    if ext in (".txt", ".md"):
        return [Page(name, None, data.decode("utf-8", errors="replace"))]
    raise UnsupportedFileError(f"Unsupported file type: {ext or name}")


def load_path(path: str | Path) -> list[Page]:
    path = Path(path)
    return load_bytes(path.name, path.read_bytes())


def _load_pdf(name: str, data: bytes) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, pdf_page in enumerate(reader.pages, start=1):
        text = (pdf_page.extract_text() or "").strip()
        if text:
            pages.append(Page(name, i, text))
    return pages


def _load_docx(name: str, data: bytes) -> list[Page]:
    from docx import Document

    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return [Page(name, None, "\n".join(parts))] if parts else []
