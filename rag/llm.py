"""Answer generation with Claude, grounded in retrieved chunks with native citations.

Each retrieved chunk is sent as its own `document` block with citations enabled, so
Claude returns the exact passages it relied on - no fragile "[1]" parsing needed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import anthropic

from .store import SearchResult

SYSTEM_PROMPT = """You are a careful assistant that answers questions using ONLY the documents provided in the user's message.

- Reply in the same language and dialect the user writes in (Arabic or English).
- Base every claim on the documents and cite them. If the documents do not contain the answer, say so plainly instead of guessing, and suggest what document might help.
- Be concise: lead with the direct answer, then supporting detail. Use short bullet points when listing several facts.
- Earlier turns of the conversation are context for follow-up questions; the documents are the source of truth."""

MAX_HISTORY_TURNS = 6
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class Citation:
    source: str
    cited_text: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    model: str | None = None


def build_messages(
    question: str, results: list[SearchResult], history: list[dict] | None = None
) -> list[dict]:
    """Prior Q/A turns as plain text, then the current question with document blocks."""
    messages: list[dict] = []
    for turn in (history or [])[-MAX_HISTORY_TURNS * 2 :]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    content: list[dict] = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": r.chunk.text},
            "title": r.chunk.label(),
            "citations": {"enabled": True},
        }
        for r in results
    ]
    content.append({"type": "text", "text": question})
    messages.append({"role": "user", "content": content})
    return messages


def extract_citations(message) -> list[Citation]:
    """Collect unique (source, quote) pairs from the response's text blocks."""
    seen: set[tuple[str, str]] = set()
    out: list[Citation] = []
    for block in message.content:
        if block.type != "text":
            continue
        for c in getattr(block, "citations", None) or []:
            key = (c.document_title or f"doc {c.document_index}", c.cited_text.strip())
            if key not in seen:
                seen.add(key)
                out.append(Citation(*key))
    return out


class AnswerStream:
    """Iterate to receive text deltas; afterwards `.answer` holds the complete Answer."""

    def __init__(self, stream_manager):
        self._manager = stream_manager
        self.answer: Answer | None = None

    def __iter__(self) -> Iterator[str]:
        with self._manager as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
        self.answer = to_answer(final)


def to_answer(final) -> Answer:
    if final.stop_reason == "refusal":
        return Answer(
            text="عذراً، لا أستطيع المساعدة في هذا الطلب. / Sorry, I can't help with this request.",
            refused=True,
            model=final.model,
        )
    text = "".join(b.text for b in final.content if b.type == "text")
    return Answer(text=text, citations=extract_citations(final), model=final.model)


class ClaudeAnswerer:
    def __init__(
        self, model: str, effort: str = "medium", client: anthropic.Anthropic | None = None
    ):
        self.model = model
        self.effort = effort
        self.client = client or anthropic.Anthropic()

    def stream(
        self, question: str, results: list[SearchResult], history: list[dict] | None = None
    ) -> AnswerStream:
        manager = self.client.beta.messages.stream(
            model=self.model,
            max_tokens=64000,
            # Stable system prompt is cached across questions.
            system=[
                {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
            ],
            messages=build_messages(question, results, history),
            output_config={"effort": self.effort},
            # If a safety classifier declines, the API retries on a suitable fallback model.
            betas=[FALLBACK_BETA],
            extra_body={"fallbacks": "default"},
        )
        return AnswerStream(manager)

    def answer(
        self,
        question: str,
        results: list[SearchResult],
        history: list[dict] | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        stream = self.stream(question, results, history)
        for text in stream:
            if on_text:
                on_text(text)
        assert stream.answer is not None
        return stream.answer
