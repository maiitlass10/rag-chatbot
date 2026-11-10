"""Central configuration, overridable through environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    model: str = field(default_factory=lambda: os.environ.get("RAG_MODEL", "claude-opus-5-5"))
    effort: str = field(default_factory=lambda: os.environ.get("RAG_EFFORT", "medium"))
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "RAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
        )
    )
    index_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("RAG_INDEX_DIR", ".rag_index"))
    )
    chunk_size: int = field(default_factory=lambda: _env_int("RAG_CHUNK_SIZE", 900))
    chunk_overlap: int = field(default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 150))
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 6))


def get_settings() -> Settings:
    return Settings()
