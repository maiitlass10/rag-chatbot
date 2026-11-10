from types import SimpleNamespace as NS

from rag.chunking import Chunk
from rag.llm import build_messages, extract_citations, to_answer
from rag.store import SearchResult


def result(text, source="a.pdf", page=2):
    return SearchResult(Chunk("id", source, page, text), 0.1, 1, 1)


def test_build_messages_sends_documents_with_citations():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    msgs = build_messages("سؤال؟", [result("نص المستند")], history)
    assert msgs[:2] == history
    content = msgs[-1]["content"]
    assert content[0]["type"] == "document"
    assert content[0]["title"] == "a.pdf (p. 2)"
    assert content[0]["citations"] == {"enabled": True}
    assert content[-1] == {"type": "text", "text": "سؤال؟"}


def test_history_is_truncated():
    history = [{"role": r, "content": str(i)} for i in range(20) for r in ("user", "assistant")]
    assert len(build_messages("q", [result("t")], history)) == 12 + 1


def cite(title, text):
    return NS(document_title=title, document_index=0, cited_text=text)


def test_extract_citations_dedupes():
    msg = NS(
        content=[
            NS(type="thinking"),
            NS(
                type="text",
                text="21 days",
                citations=[cite("hr.md", "21 يوم"), cite("hr.md", "21 يوم ")],
            ),
            NS(type="text", text=".", citations=None),
        ]
    )
    cites = extract_citations(msg)
    assert len(cites) == 1 and cites[0].source == "hr.md"


def test_to_answer_handles_refusal():
    final = NS(stop_reason="refusal", content=[], model="m")
    assert to_answer(final).refused


def test_to_answer_joins_text():
    final = NS(
        stop_reason="end_turn",
        model="m",
        content=[
            NS(type="text", text="A", citations=None),
            NS(type="text", text="B", citations=None),
        ],
    )
    assert to_answer(final).text == "AB"
