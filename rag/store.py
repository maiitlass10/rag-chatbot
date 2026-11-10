"""Hybrid retrieval index: BM25 keyword search + dense vectors, fused with RRF.

Why hybrid? Dense embeddings capture meaning ("salary" ~ "راتب") while BM25 nails exact
terms such as names, codes and numbers. Reciprocal Rank Fusion combines both rankings
without having to calibrate their very different score scales.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder, embedder_name
from .text import tokenize

RRF_K = 60  # standard constant from Cormack et al. (2009)


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    dense_rank: int | None
    keyword_rank: int | None


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens: list[Counter] = []
        self.doc_len: list[int] = []
        self.df: Counter = Counter()

    def add(self, texts: list[str]) -> None:
        for text in texts:
            tf = Counter(tokenize(text))
            self.doc_tokens.append(tf)
            self.doc_len.append(sum(tf.values()))
            self.df.update(tf.keys())

    def remove(self, indices: set[int]) -> None:
        keep = [i for i in range(len(self.doc_tokens)) if i not in indices]
        tokens = [self.doc_tokens[i] for i in keep]
        self.doc_tokens, self.doc_len, self.df = [], [], Counter()
        for tf in tokens:
            self.doc_tokens.append(tf)
            self.doc_len.append(sum(tf.values()))
            self.df.update(tf.keys())

    def scores(self, query: str) -> np.ndarray:
        n = len(self.doc_tokens)
        out = np.zeros(n, dtype=np.float32)
        if n == 0:
            return out
        avgdl = (sum(self.doc_len) / n) or 1.0
        for term in set(tokenize(query)):
            df = self.df.get(term)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self.doc_tokens):
                f = tf.get(term)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / avgdl)
                    out[i] += idf * f * (self.k1 + 1) / denom
        return out


class HybridIndex:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None
        self.bm25 = BM25()

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def add(self, chunks: list[Chunk]) -> int:
        existing = {c.id for c in self.chunks}
        new = [c for c in chunks if c.id not in existing]
        if not new:
            return 0
        vecs = self.embedder.embed_documents([c.text for c in new])
        self.vectors = vecs if self.vectors is None else np.vstack([self.vectors, vecs])
        self.chunks.extend(new)
        self.bm25.add([c.text for c in new])
        return len(new)

    def remove_source(self, source: str) -> int:
        drop = {i for i, c in enumerate(self.chunks) if c.source == source}
        if not drop:
            return 0
        keep = [i for i in range(len(self.chunks)) if i not in drop]
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if keep and self.vectors is not None else None
        self.bm25.remove(drop)
        return len(drop)

    def search(self, query: str, k: int = 6, candidates: int = 30) -> list[SearchResult]:
        if not self.chunks:
            return []
        dense = self.vectors @ self.embedder.embed_query(query)
        keyword = self.bm25.scores(query)

        dense_order = np.argsort(-dense)[:candidates]
        keyword_order = [i for i in np.argsort(-keyword)[:candidates] if keyword[i] > 0]

        fused: dict[int, float] = {}
        dense_rank = {int(i): r for r, i in enumerate(dense_order, start=1)}
        keyword_rank = {int(i): r for r, i in enumerate(keyword_order, start=1)}
        for ranks in (dense_rank, keyword_rank):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r)

        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [
            SearchResult(self.chunks[i], fused[i], dense_rank.get(i), keyword_rank.get(i))
            for i in best
        ]

    # ---- persistence ---------------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "embedder": embedder_name(self.embedder),
            "chunks": [c.to_dict() for c in self.chunks],
        }
        (directory / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False), "utf-8")
        np.save(
            directory / "vectors.npy",
            self.vectors if self.vectors is not None else np.zeros((0, 0)),
        )

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder) -> HybridIndex:
        directory = Path(directory)
        index = cls(embedder)
        meta_path = directory / "chunks.json"
        if not meta_path.exists():
            return index
        meta = json.loads(meta_path.read_text("utf-8"))
        if meta.get("embedder") != embedder_name(embedder):
            # Vectors from a different model are not comparable - start fresh.
            return index
        index.chunks = [Chunk(**c) for c in meta["chunks"]]
        vectors = np.load(directory / "vectors.npy")
        index.vectors = vectors if len(index.chunks) else None
        index.bm25.add([c.text for c in index.chunks])
        return index
