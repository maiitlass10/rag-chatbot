"""Split pages into overlapping chunks along paragraph / sentence boundaries."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

from .loaders import Page

# Sentence ends: Latin and Arabic punctuation (؟ ، ؛ are Arabic forms).
_SENTENCE_END = re.compile(r"(?<=[.!?؟؛])\s+|\n+")


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page: int | None
    text: str

    def label(self) -> str:
        return f"{self.source} (p. {self.page})" if self.page else self.source

    def to_dict(self) -> dict:
        return asdict(self)


def _split_units(text: str) -> list[str]:
    return [u.strip() for u in _SENTENCE_END.split(text) if u and u.strip()]


def _hard_wrap(unit: str, size: int) -> list[str]:
    """Break a single over-long sentence on word boundaries."""
    out, current = [], ""
    for word in unit.split():
        if current and len(current) + 1 + len(word) > size:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        out.append(current)
    return out


def chunk_page(page: Page, size: int = 900, overlap: int = 150) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")

    units: list[str] = []
    for unit in _split_units(page.text):
        units.extend(_hard_wrap(unit, size) if len(unit) > size else [unit])

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for unit in units:
        if current and length + len(unit) + 1 > size:
            chunks.append(" ".join(current))
            # Carry trailing sentences forward as overlap for context continuity.
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(current):
                if tail_len + len(prev) + 1 > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            current, length = tail, tail_len
        current.append(unit)
        length += len(unit) + 1
    if current:
        chunks.append(" ".join(current))

    return [
        Chunk(
            id=hashlib.sha1(f"{page.source}|{page.page}|{i}|{text}".encode()).hexdigest()[:16],
            source=page.source,
            page=page.page,
            text=text,
        )
        for i, text in enumerate(chunks)
    ]


def chunk_pages(pages: list[Page], size: int = 900, overlap: int = 150) -> list[Chunk]:
    return [c for p in pages for c in chunk_page(p, size, overlap)]
