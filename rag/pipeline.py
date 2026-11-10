"""High-level facade tying loading, chunking, indexing and answering together."""

from __future__ import annotations

from pathlib import Path

from .chunking import chunk_pages
from .config import Settings, get_settings
from .embeddings import Embedder, default_embedder
from .llm import Answer, AnswerStream, ClaudeAnswerer
from .loaders import load_bytes
from .store import HybridIndex, SearchResult

NO_DOCS_MESSAGE = (
    "ما في مستندات مرفوعة بعد - ارفع ملف أولاً. / No documents indexed yet - upload a file first."
)


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        answerer: ClaudeAnswerer | None = None,
    ):
        self.settings = settings or get_settings()
        self.embedder = embedder or default_embedder(self.settings.embedding_model)
        self.index = HybridIndex.load(self.settings.index_dir, self.embedder)
        self._answerer = answerer

    @property
    def answerer(self) -> ClaudeAnswerer:
        # Created lazily so indexing works without an API key.
        if self._answerer is None:
            self._answerer = ClaudeAnswerer(self.settings.model, self.settings.effort)
        return self._answerer

    def ingest(self, name: str, data: bytes) -> int:
        """Index a file; re-uploading a file with the same name replaces it."""
        pages = load_bytes(name, data)
        chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        self.index.remove_source(name)
        added = self.index.add(chunks)
        self.index.save(self.settings.index_dir)
        return added

    def ingest_path(self, path: str | Path) -> int:
        path = Path(path)
        return self.ingest(path.name, path.read_bytes())

    def remove(self, source: str) -> int:
        removed = self.index.remove_source(source)
        self.index.save(self.settings.index_dir)
        return removed

    def retrieve(self, question: str) -> list[SearchResult]:
        return self.index.search(question, k=self.settings.top_k)

    def ask(
        self, question: str, history: list[dict] | None = None, on_text=None
    ) -> tuple[Answer, list[SearchResult]]:
        results = self.retrieve(question)
        if not results:
            return Answer(text=NO_DOCS_MESSAGE), []
        return self.answerer.answer(question, results, history, on_text=on_text), results

    def ask_stream(
        self, question: str, history: list[dict] | None = None
    ) -> tuple[AnswerStream | None, list[SearchResult]]:
        results = self.retrieve(question)
        if not results:
            return None, []
        return self.answerer.stream(question, results, history), results
