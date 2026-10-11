"""Hybrid retrieval: dense (vector) + BM25 keyword, merged with reciprocal rank fusion."""
import re

from ..core.config import settings
from ..core.models import embed_query, rerank_scores
from .store import get_store

_bm25_cache: dict = {"count": -1, "bm25": None, "chunks": []}


def _tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def _bm25():
    from rank_bm25 import BM25Okapi

    store = get_store()
    n = store.count()
    if _bm25_cache["count"] != n:  # rebuild only when the index changed
        chunks = store.all_chunks()
        _bm25_cache.update(
            count=n, chunks=chunks, bm25=BM25Okapi([_tok(c["text"]) for c in chunks]) if chunks else None
        )
    return _bm25_cache["bm25"], _bm25_cache["chunks"]


def hybrid_search(query: str, k: int | None = None) -> list[dict]:
    k = k or settings.retrieve_k
    dense = get_store().query(embed_query(query), k)
    bm25, chunks = _bm25()
    sparse = []
    if bm25:
        scores = bm25.get_scores(_tok(query))
        top = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
        sparse = [chunks[i] for i in top if scores[i] > 0]

    fused: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for ranking in (dense, sparse):
        for rank, c in enumerate(ranking):
            fused[c["id"]] = fused.get(c["id"], 0.0) + 1.0 / (60 + rank)  # RRF, k=60
            by_id[c["id"]] = c
    return [by_id[i] for i in sorted(fused, key=fused.get, reverse=True)[:k]]


def rerank(query: str, docs: list[dict], top_k: int | None = None) -> list[dict]:
    if not docs:
        return []
    top_k = top_k or settings.final_k
    if settings.use_jina:
        docs = docs[:settings.jina_rerank_candidates]  # docs arrive in fused-rank order
    scores = rerank_scores(query, [d["text"] for d in docs])
    for d, s in zip(docs, scores):
        d["score"] = float(s)
    return sorted(docs, key=lambda d: d["score"], reverse=True)[:top_k]
