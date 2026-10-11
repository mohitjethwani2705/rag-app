"""Evaluation harness.

  python -m eval.run                 # retrieval only: dense vs BM25 vs hybrid vs hybrid+rerank (no LLM calls)
  python -m eval.run --full          # + generation, LLM-judged correctness, abstention, score-gate calibration
  python -m eval.run --full --rewrite --grade    # also turn on the optional agentic nodes

Runs against an isolated index in data/eval-<tag>, never your real one. A question counts as a retrieval
"hit" when one of its evidence strings appears in a retrieved chunk (independent of chunk boundaries).
"""
import argparse
import json
import re
import shutil
import statistics
import time
from pathlib import Path

from rag.core.config import settings

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def has_evidence(docs: list[dict], evidence: list[str]) -> bool:
    return any(norm(e) in norm(d["text"]) for d in docs for e in evidence)


def first_hit_rank(docs: list[dict], evidence: list[str]) -> int | None:
    for i, d in enumerate(docs, 1):
        if any(norm(e) in norm(d["text"]) for e in evidence):
            return i
    return None


def retry(fn, tries=4):
    for attempt in range(tries):
        try:
            return fn()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(4 * (attempt + 1))


def parse_json_flag(text: str, key: str) -> bool | None:
    m = re.search(r"\{.*?\}", text, re.S)
    try:
        return bool(json.loads(m.group())[key]) if m else None
    except Exception:
        return None


def best_threshold(pos: list[float], neg: list[float]) -> tuple[float, float]:
    """Cutoff that best separates answerable (pos) from unanswerable (neg) top scores."""
    cands = sorted(set(pos + neg))
    mids = [(a + b) / 2 for a, b in zip(cands, cands[1:])] or cands
    def bal_acc(t):
        tpr = sum(p >= t for p in pos) / len(pos)
        tnr = sum(n < t for n in neg) / len(neg)
        return (tpr + tnr) / 2
    best = max(mids, key=bal_acc)
    return best, bal_acc(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also run generation + judge (uses LLM)")
    ap.add_argument("--rewrite", action="store_true")
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=150, help="tokens; small so retrieval is non-trivial")
    ap.add_argument("--overlap", type=int, default=25)
    ap.add_argument("--judge-model", default="openai/gpt-oss-120b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="default", help="isolates the index dir so runs can go in parallel")
    args = ap.parse_args()

    eval_dir = ROOT / "data" / f"eval-{args.tag}"
    shutil.rmtree(eval_dir, ignore_errors=True)
    settings.data_dir, settings.collection = eval_dir, "eval"
    settings.chunk_size, settings.chunk_overlap = args.chunk_size, args.overlap
    settings.enable_query_rewrite, settings.enable_llm_grading = args.rewrite, args.grade

    from rag.core import db
    from rag.core.models import embed_query
    from rag.ingest.pipeline import ingest_file
    from rag.retrieval.hybrid import _bm25, _tok, hybrid_search, rerank
    from rag.retrieval.store import get_store

    db.init_db()
    for f in sorted((HERE / "docs").iterdir()):
        ingest_file(f)
    print(f"Indexed {get_store().count()} chunks (chunk_size={args.chunk_size} tokens)\n")

    qs = [json.loads(l) for l in (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l]
    if args.limit:
        qs = qs[: args.limit]
    answerable = [q for q in qs if q["answerable"]]
    K = settings.final_k

    # ---------------- retrieval ----------------
    def dense(q, k):
        return get_store().query(embed_query(q), k)

    def bm25(q, k):
        b, chunks = _bm25()
        sc = b.get_scores(_tok(q))
        return [chunks[i] for i in sorted(range(len(chunks)), key=lambda i: -sc[i])[:k] if sc[i] > 0]

    methods = {
        "dense only": lambda q: dense(q, K),
        "BM25 only": lambda q: bm25(q, K),
        "hybrid (RRF)": lambda q: hybrid_search(q)[:K],
        "hybrid + rerank": lambda q: rerank(q, [dict(d) for d in hybrid_search(q)]),
    }
    print(f"RETRIEVAL  (hit@{K} over {len(answerable)} answerable questions)")
    results = {}
    for name, fn in methods.items():
        ranks = [first_hit_rank(fn(q["question"]), q["evidence"]) for q in answerable]
        hits = sum(r is not None for r in ranks)
        mrr = statistics.mean(1 / r if r else 0 for r in ranks)
        results[name] = dict(hit=hits / len(answerable), mrr=mrr)
        print(f"  {name:<18} hit@{K} {hits:>2}/{len(answerable)} = {hits / len(answerable):5.1%}   MRR {mrr:.3f}")
    cand = sum(has_evidence(hybrid_search(q["question"]), q["evidence"]) for q in answerable)
    print(f"  candidate recall: hybrid top-{settings.retrieve_k} contains the evidence for {cand}/{len(answerable)}")
    misses = [q for q in answerable if not has_evidence(
        rerank(q["question"], [dict(d) for d in hybrid_search(q["question"])]), q["evidence"])]
    for q in misses:
        print(f"  MISS {q['id']}: {q['question']}")

    if not args.full:
        return

    # ---------------- generation ----------------
    from langchain_groq import ChatGroq

    from rag.graph.build import get_graph

    if not settings.llm_configured:
        raise SystemExit("\n--full needs an LLM: set GROQ_API_KEY in .env")
    judge = ChatGroq(model=args.judge_model, api_key=settings.groq_api_key, temperature=0)
    graph = get_graph()
    print(f"\nGENERATION  (model={settings.groq_model}, rewrite={args.rewrite}, llm_grading={args.grade})")

    rows = []
    for q in qs:
        t0 = time.time()
        st = retry(lambda: graph.invoke({"question": q["question"], "chat_history": []}))
        latency = time.time() - t0
        ans = st["answer"]
        if q["answerable"]:
            p = (f"Question: {q['question']}\nReference answer: {q['expected']}\nCandidate answer: {ans}\n\n"
                 "Does the candidate answer the question consistently with the reference (same key facts, "
                 'no contradictions)? Reply with JSON only: {"correct": true} or {"correct": false}.')
            ok = parse_json_flag(retry(lambda: judge.invoke(p)).content, "correct")
        else:
            p = (f"Question: {q['question']}\nCandidate answer: {ans}\n\nThe source documents do NOT contain the "
                 "answer. Does the candidate say it cannot find the information (instead of inventing or "
                 'guessing an answer)? Reply with JSON only: {"abstained": true} or {"abstained": false}.')
            ok = parse_json_flag(retry(lambda: judge.invoke(p)).content, "abstained")
        top = st["docs"][0]["score"] if st.get("docs") else None
        cited = bool(re.search(r"\[\d+\]", ans))
        rows.append(dict(id=q["id"], question=q["question"], answerable=q["answerable"], ok=ok,
                         answer=ans, top_score=top, sufficient=st.get("sufficient"),
                         cited=cited, latency=latency))
        print(f"  {q['id']} {'PASS' if ok else 'FAIL' if ok is False else '??? '}  {q['question'][:60]}")

    A = [r for r in rows if r["answerable"]]
    N = [r for r in rows if not r["answerable"]]
    print("\nSUMMARY")
    print(f"  answerable correct   {sum(r['ok'] is True for r in A):>2}/{len(A)} = {sum(r['ok'] is True for r in A) / len(A):5.1%}")
    if N:
        print(f"  unanswerable abstain {sum(r['ok'] is True for r in N):>2}/{len(N)} = {sum(r['ok'] is True for r in N) / len(N):5.1%}")
    print(f"  answers with citation {sum(r['cited'] for r in A):>2}/{len(A)}")
    print(f"  median latency        {statistics.median(r['latency'] for r in rows):.1f}s")

    pos = [r["top_score"] for r in A if r["top_score"] is not None]
    neg = [r["top_score"] for r in N if r["top_score"] is not None]
    if pos and neg:
        t, acc = best_threshold(pos, neg)
        cur = settings.min_rerank_score
        print(f"\nSCORE GATE  (top reranker logit; current MIN_RERANK_SCORE={cur})")
        print(f"  answerable   min {min(pos):7.2f}  median {statistics.median(pos):7.2f}")
        print(f"  unanswerable max {max(neg):7.2f}  median {statistics.median(neg):7.2f}")
        print(f"  current cutoff: flags {sum(p < cur for p in pos)}/{len(pos)} good questions, "
              f"catches {sum(n < cur for n in neg)}/{len(neg)} unanswerable")
        print(f"  suggested cutoff {t:.2f} (balanced accuracy {acc:.0%})")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.tag}.json"
    path.write_text(json.dumps(dict(args=vars(args), retrieval=results, rows=rows), indent=2), encoding="utf-8")
    for r in rows:
        if r["ok"] is not True:
            print(f"\n  FAILED {r['id']}: {r['question']}\n    answer: {r['answer'][:200]!r}  top_score={r['top_score']}")
    print(f"\nSaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
