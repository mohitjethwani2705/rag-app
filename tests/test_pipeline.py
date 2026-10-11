"""End-to-end ingest + retrieval test using temp storage. Needs the embedding/reranker models
(downloaded on first run). Does not need an LLM."""
import pytest

from rag.core import db
from rag.core.config import settings


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "collection", "test")
    from rag.retrieval import hybrid, store

    store.get_store.cache_clear()
    hybrid._bm25_cache.update(count=-1, bm25=None, chunks=[])
    db.init_db()
    yield
    store.get_store.cache_clear()


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_ingest_retrieve_and_dedupe(tmp_path):
    from rag.ingest.pipeline import ingest_file
    from rag.retrieval.hybrid import hybrid_search, rerank

    a = _write(tmp_path, "cats.txt", "Cats are obligate carnivores. They sleep up to sixteen hours a day.")
    b = _write(tmp_path, "rockets.txt", "The Saturn V rocket carried Apollo astronauts to the Moon.")
    r1 = ingest_file(a)
    ingest_file(b)
    assert r1["chunks"] >= 1 and not r1["skipped"]

    # unchanged re-ingest is skipped, not duplicated
    assert ingest_file(a)["skipped"] is True

    docs = rerank("What carried astronauts to the Moon?", hybrid_search("What carried astronauts to the Moon?"))
    assert docs[0]["metadata"]["title"] == "rockets.txt"

    # changed content replaces old chunks
    a.write_text("Dogs are loyal companions.", encoding="utf-8")
    assert ingest_file(a)["skipped"] is False
    hits = hybrid_search("obligate carnivores")
    assert all("carnivores" not in h["text"] for h in hits)
