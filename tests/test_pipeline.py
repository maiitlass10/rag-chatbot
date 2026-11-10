from pathlib import Path

from rag.config import Settings
from rag.embeddings import HashingEmbedder
from rag.llm import Answer
from rag.pipeline import NO_DOCS_MESSAGE, RAGPipeline

SAMPLES = Path(__file__).resolve().parents[1] / "sample_docs"


class FakeAnswerer:
    def __init__(self):
        self.calls = []

    def answer(self, question, results, history=None, on_text=None):
        self.calls.append((question, results))
        if on_text:
            on_text("ok")
        return Answer(text="ok")


def make(tmp_path, answerer=None):
    settings = Settings(index_dir=tmp_path / "idx", chunk_size=300, chunk_overlap=50, top_k=3)
    return RAGPipeline(settings, embedder=HashingEmbedder(), answerer=answerer)


def test_ask_without_documents(tmp_path):
    answer, results = make(tmp_path, FakeAnswerer()).ask("hello?")
    assert answer.text == NO_DOCS_MESSAGE and results == []


def test_ingest_persist_and_ask(tmp_path):
    fake = FakeAnswerer()
    p = make(tmp_path, fake)
    assert p.ingest_path(SAMPLES / "hr_policy_ar.md") > 0
    # Re-ingesting the same file replaces it instead of duplicating.
    n = len(p.index)
    p.ingest_path(SAMPLES / "hr_policy_ar.md")
    assert len(p.index) == n

    # A fresh pipeline reloads the saved index from disk.
    p2 = make(tmp_path, fake)
    assert len(p2.index) == n
    answer, results = p2.ask("بدل التدريب كم؟")
    assert answer.text == "ok" and len(results) == 3
    assert any("1500" in r.chunk.text for r in results)
