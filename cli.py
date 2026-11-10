"""Command-line interface.

python cli.py ingest docs/*.pdf
python cli.py ask "ما هي شروط الإجازة السنوية؟"
python cli.py chat
python cli.py search "annual leave"      # retrieval only, no API key needed
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from rag.pipeline import RAGPipeline


def _print_answer(pipeline: RAGPipeline, question: str, history: list[dict]) -> str:
    answer, _ = pipeline.ask(question, history, on_text=lambda t: print(t, end="", flush=True))
    if not answer.text.endswith("\n"):
        print()
    if answer.citations:
        print("\nSources:")
        for c in answer.citations:
            quote = c.cited_text if len(c.cited_text) < 160 else c.cited_text[:157] + "…"
            print(f"  • {c.source}: “{quote}”")
    return answer.text


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Chat with your documents")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingest", help="index files")
    p_ing.add_argument("paths", nargs="+")
    p_ask = sub.add_parser("ask", help="ask one question")
    p_ask.add_argument("question")
    p_search = sub.add_parser("search", help="show retrieved chunks only")
    p_search.add_argument("query")
    sub.add_parser("chat", help="interactive chat")
    sub.add_parser("list", help="list indexed files")
    args = parser.parse_args(argv)

    pipeline = RAGPipeline()
    if args.cmd == "ingest":
        for path in args.paths:
            print(f"{path}: {pipeline.ingest_path(path)} chunks")
    elif args.cmd == "list":
        for s in pipeline.index.sources:
            print(s)
    elif args.cmd == "search":
        for r in pipeline.retrieve(args.query):
            print(f"[{r.score:.4f}] {r.chunk.label()}  dense#{r.dense_rank} kw#{r.keyword_rank}")
            print(f"    {r.chunk.text[:200]}")
    elif args.cmd == "ask":
        _print_answer(pipeline, args.question, [])
    elif args.cmd == "chat":
        history: list[dict] = []
        print("Type your question (Ctrl+D to quit)")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if q:
                text = _print_answer(pipeline, q, history)
                history += [{"role": "user", "content": q}, {"role": "assistant", "content": text}]
    return 0


if __name__ == "__main__":
    sys.exit(main())
