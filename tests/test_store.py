from pathlib import Path

from rag.chunking import chunk_pages
from rag.embeddings import HashingEmbedder
from rag.loaders import load_path
from rag.store import HybridIndex

SAMPLES = Path(__file__).resolve().parents[1] / "sample_docs"


def build_index() -> HybridIndex:
    index = HybridIndex(HashingEmbedder())
    for f in sorted(SAMPLES.glob("*.md")):
        index.add(chunk_pages(load_path(f), size=300, overlap=50))
    return index


def test_arabic_query_finds_arabic_policy():
    top = build_index().search("كم يوم الإجازة السنوية؟", k=1)[0]
    assert top.chunk.source == "hr_policy_ar.md"
    assert "21" in top.chunk.text


def test_keyword_match_on_exact_term():
    top = build_index().search("AES-256 encryption", k=1)[0]
    assert "AES-256" in top.chunk.text


def test_add_is_idempotent_and_remove_source():
    index = build_index()
    n = len(index)
    chunks = chunk_pages(load_path(SAMPLES / "product_faq_en.md"), size=300, overlap=50)
    assert index.add(chunks) == 0
    removed = index.remove_source("product_faq_en.md")
    assert removed > 0 and len(index) == n - removed
    assert index.sources == ["hr_policy_ar.md"]
    assert all(r.chunk.source == "hr_policy_ar.md" for r in index.search("refund"))


def test_save_and_load_roundtrip(tmp_path):
    index = build_index()
    index.save(tmp_path)
    loaded = HybridIndex.load(tmp_path, HashingEmbedder())
    assert len(loaded) == len(index)
    q = "refund annual subscription"
    assert loaded.search(q, k=1)[0].chunk.id == index.search(q, k=1)[0].chunk.id


def test_load_with_different_embedder_starts_fresh(tmp_path):
    build_index().save(tmp_path)
    assert len(HybridIndex.load(tmp_path, HashingEmbedder(dim=256))) == 0


def test_empty_index_search():
    assert HybridIndex(HashingEmbedder()).search("anything") == []
