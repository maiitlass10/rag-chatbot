"""Embedding backends. The default runs locally and supports 100+ languages incl. Arabic."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model (default: multilingual-e5-small).

    E5 models expect "passage: " / "query: " prefixes; vectors are L2-normalised so a
    dot product equals cosine similarity.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._e5 = "e5" in model_name.lower()

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        prefix = "passage: " if self._e5 else ""
        return self._encode([prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        prefix = "query: " if self._e5 else ""
        return self._encode([prefix + text])[0]


class HashingEmbedder:
    """Dependency-free fallback (hashed bag of character n-grams).

    Lower quality than a neural model, but deterministic and offline - used in tests
    and when sentence-transformers is not installed.
    """

    def __init__(self, dim: int = 512, n: int = 3):
        self.dim, self.n = dim, n

    def _vec(self, text: str) -> np.ndarray:
        import zlib

        from .text import normalize

        v = np.zeros(self.dim, dtype=np.float32)
        t = f" {normalize(text)} "
        for i in range(max(len(t) - self.n + 1, 1)):
            v[zlib.crc32(t[i : i + self.n].encode()) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def default_embedder(model_name: str) -> Embedder:
    try:
        return SentenceTransformerEmbedder(model_name)
    except (ImportError, OSError):
        # Not installed, or the model could not be downloaded (offline).
        return HashingEmbedder()


def embedder_name(embedder: Embedder) -> str:
    if isinstance(embedder, SentenceTransformerEmbedder):
        return embedder.model_name
    if isinstance(embedder, HashingEmbedder):
        return f"hashing-{embedder.dim}-{embedder.n}"
    return type(embedder).__name__
