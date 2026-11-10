import pytest

from rag.chunking import chunk_page
from rag.loaders import Page


def test_short_page_is_single_chunk():
    chunks = chunk_page(Page("a.txt", None, "جملة أولى. جملة ثانية."), size=200, overlap=20)
    assert len(chunks) == 1
    assert chunks[0].source == "a.txt"


def test_long_text_respects_size_and_overlaps():
    sentences = [f"Sentence number {i} has some words in it." for i in range(60)]
    chunks = chunk_page(Page("b.txt", 3, " ".join(sentences)), size=300, overlap=80)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(c.page == 3 for c in chunks)
    # Consecutive chunks share at least one sentence (overlap).
    first_tail = chunks[0].text.split(". ")[-1]
    assert first_tail.rstrip(".") in chunks[1].text


def test_overlong_sentence_is_wrapped():
    chunks = chunk_page(Page("c.txt", None, "كلمة " * 500), size=100, overlap=10)
    assert all(len(c.text) <= 100 for c in chunks)


def test_ids_are_stable_and_unique():
    page = Page("d.txt", None, "One. Two. Three. " * 50)
    a = chunk_page(page, 120, 20)
    b = chunk_page(page, 120, 20)
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_page(Page("e.txt", None, "x"), size=10, overlap=10)
