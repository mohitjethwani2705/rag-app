"""Terminal interface.

  python -m rag.cli ingest <file-or-folder> ...
  python -m rag.cli ask "question"
  python -m rag.cli chat          # interactive, remembers the conversation
  python -m rag.cli docs          # list indexed documents
"""
import argparse
import sys
from pathlib import Path

from .core import db
from .core.config import settings
from .ingest.loaders.files import SUPPORTED


def _print_sources(citations: list[dict]) -> None:
    for c in citations:
        page = f", p.{c['page']}" if c["page"] else ""
        print(f"  [{c['n']}] {c['title']}{page}  (score {c['score']})")


def _answer(question: str, history: list[dict]) -> str:
    from .graph.build import get_retrieval_graph
    from .graph.nodes import NO_DOCS_ANSWER, build_prompt, citations_for
    from .core.models import get_llm

    state = get_retrieval_graph().invoke({"question": question, "chat_history": history})
    if not state.get("docs") and state.get("intent") != "small_talk":
        print(NO_DOCS_ANSWER)
        return NO_DOCS_ANSWER
    answer = ""
    for chunk in get_llm().stream(build_prompt(state)):  # stream tokens as they arrive
        answer += chunk.content
        print(chunk.content, end="", flush=True)
    print()
    cites = citations_for(state, answer)
    if cites:
        print("\nSources:")
        _print_sources(cites)
    if not state.get("sufficient"):
        print("\n(note: retrieved context scored low; the answer may not be grounded)")
    return answer


def cmd_ingest(paths: list[str]) -> None:
    from .ingest.pipeline import ingest_file

    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED) if p.is_dir() else [p]
    if not files:
        sys.exit("No supported files found (pdf, docx, txt, md).")
    for f in files:
        try:
            r = ingest_file(f)
            print(f"{f.name}: " + ("unchanged, skipped" if r["skipped"] else f"{r['chunks']} chunks"))
        except Exception as e:
            print(f"{f.name}: FAILED - {e}")


def cmd_chat() -> None:
    if not settings.llm_configured:
        sys.exit("No LLM configured. Set GROQ_API_KEY in .env")
    print(f"Chatting via {settings.provider}. Type 'exit' to quit.\n")
    history: list[dict] = []
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if q.lower() in {"exit", "quit"}:
            return
        if not q:
            continue
        print("bot> ", end="", flush=True)
        a = _answer(q, history)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        print()


def main() -> None:
    ap = argparse.ArgumentParser(prog="rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chat")
    sub.add_parser("docs")
    i = sub.add_parser("ingest")
    i.add_argument("paths", nargs="+")
    a = sub.add_parser("ask")
    a.add_argument("question")
    args = ap.parse_args()

    db.init_db()
    if args.cmd == "ingest":
        cmd_ingest(args.paths)
    elif args.cmd == "ask":
        if not settings.llm_configured:
            sys.exit("No LLM configured. Set GROQ_API_KEY in .env")
        _answer(args.question, [])
    elif args.cmd == "chat":
        cmd_chat()
    elif args.cmd == "docs":
        with db.connect() as c:
            for r in c.execute("SELECT title, chunk_count, ingested_at FROM documents"):
                print(f"{r['title']}  ({r['chunk_count']} chunks, {r['ingested_at']})")


if __name__ == "__main__":
    main()
