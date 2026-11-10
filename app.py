"""Streamlit chat UI.  Run with:  streamlit run app.py"""

from __future__ import annotations

import os
import re

import anthropic
import streamlit as st
from dotenv import load_dotenv

from rag.loaders import SUPPORTED_EXTENSIONS, UnsupportedFileError
from rag.pipeline import NO_DOCS_MESSAGE, RAGPipeline

load_dotenv()
st.set_page_config(page_title="Docs Chat · مساعد المستندات", page_icon="📚", layout="wide")

# Right-to-left rendering so Arabic answers read naturally.
st.markdown(
    """<style>
    [data-testid="stChatMessageContent"] p, [data-testid="stChatMessageContent"] li {
        unicode-bidi: plaintext; text-align: start;
    }
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading embedding model…")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


pipeline = get_pipeline()
HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
st.session_state.setdefault("history", [])  # [{"role", "content", "citations"?}]

# ---------------- Sidebar: documents ----------------
with st.sidebar:
    st.header("📂 المستندات / Documents")
    uploads = st.file_uploader(
        "PDF, DOCX, TXT, MD",
        type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
    if uploads and st.button("➕ فهرسة / Index", use_container_width=True):
        for f in uploads:
            try:
                with st.spinner(f"Indexing {f.name}…"):
                    n = pipeline.ingest(f.name, f.getvalue())
                st.success(f"{f.name}: {n} chunks")
            except UnsupportedFileError as e:
                st.error(str(e))

    st.divider()
    st.caption(f"{len(pipeline.index)} chunks · {len(pipeline.index.sources)} files")
    for source in pipeline.index.sources:
        col1, col2 = st.columns([5, 1])
        col1.write(f"📄 {source}")
        if col2.button("🗑️", key=f"rm-{source}", help="Remove"):
            pipeline.remove(source)
            st.rerun()

    st.divider()
    if st.button("🧹 محادثة جديدة / New chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    if not HAS_KEY:
        st.info(
            "🔎 وضع البحث فقط: ما في مفتاح API، فرح يعرض أنسب المقاطع من ملفاتك. "
            "حط ANTHROPIC_API_KEY بملف .env لتفعيل الأجوبة الذكية.\n\n"
            "Search-only mode: add ANTHROPIC_API_KEY to .env for AI answers."
        )

# ---------------- Chat ----------------
st.title("📚 اسأل مستنداتك · Chat with your documents")


def search_only_answer(question: str) -> tuple[str, list[dict]]:
    """Free mode without an API key: show the best-matching passages."""
    results = pipeline.retrieve(question)
    if not results:
        return NO_DOCS_MESSAGE, []
    lines = ["**أنسب المقاطع من مستنداتك / Most relevant passages:**", ""]
    for i, r in enumerate(results[:3], start=1):
        # Strip markdown heading marks so passages render as plain quoted text.
        clean = re.sub(r"#+\s*", "", r.chunk.text)
        text = clean if len(clean) <= 500 else clean[:500] + "…"
        lines.append(f"**{i}. {r.chunk.label()}**\n\n> {text}\n")
    return "\n".join(lines), []


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"📎 المصادر / Sources ({len(citations)})"):
        for c in citations:
            st.markdown(f"**{c['source']}**")
            st.caption(f"“{c['cited_text']}”")


for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        render_citations(turn.get("citations", []))

if question := st.chat_input("اكتب سؤالك هون… / Ask a question…"):
    with st.chat_message("user"):
        st.markdown(question)

    history = [{"role": t["role"], "content": t["content"]} for t in st.session_state.history]
    with st.chat_message("assistant"):
        if not HAS_KEY:
            answer_text, citations = search_only_answer(question)
            st.markdown(answer_text)
            st.session_state.history.append({"role": "user", "content": question})
            st.session_state.history.append(
                {"role": "assistant", "content": answer_text, "citations": citations}
            )
            st.stop()
        try:
            stream, results = pipeline.ask_stream(question, history)
            if stream is None:
                st.info(NO_DOCS_MESSAGE)
                answer_text, citations = NO_DOCS_MESSAGE, []
            else:
                st.write_stream(stream)
                answer = stream.answer
                answer_text = answer.text
                citations = [c.__dict__ for c in answer.citations]
                render_citations(citations)
                with st.expander("🔎 Retrieved chunks (debug)"):
                    for r in results:
                        st.markdown(
                            f"**{r.chunk.label()}** · RRF {r.score:.4f} · "
                            f"dense #{r.dense_rank} · keyword #{r.keyword_rank}"
                        )
                        st.caption(r.chunk.text[:400] + ("…" if len(r.chunk.text) > 400 else ""))
        except anthropic.AuthenticationError:
            st.error("Invalid or missing ANTHROPIC_API_KEY.")
            st.stop()
        except anthropic.RateLimitError:
            st.error("Rate limited - wait a moment and try again.")
            st.stop()
        except anthropic.APIConnectionError:
            st.error("Network error reaching the Claude API.")
            st.stop()
        except anthropic.APIStatusError as e:
            st.error(f"API error ({e.status_code}): {e.message}")
            st.stop()

    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append(
        {"role": "assistant", "content": answer_text, "citations": citations}
    )
