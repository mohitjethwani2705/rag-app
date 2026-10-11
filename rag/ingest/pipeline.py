import hashlib
import traceback
import uuid
from pathlib import Path

from ..core import db
from ..core.config import settings
from ..core.models import embed_documents
from ..retrieval.store import get_store
from .chunk import chunk_docs
from .loaders.files import FileLoader


def _set_job(job_id: str, status: str, detail: str = "") -> None:
    with db.connect() as c:
        c.execute(
            "INSERT INTO jobs(id,status,detail) VALUES(?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET status=?, detail=?, updated_at=CURRENT_TIMESTAMP",
            (job_id, status, detail, status, detail),
        )


def new_job() -> str:
    job_id = uuid.uuid4().hex[:12]
    _set_job(job_id, "queued")
    return job_id


def ingest_file(path: Path, job_id: str | None = None) -> dict:
    """Load -> dedupe -> chunk -> embed -> upsert. Safe to re-run on the same file."""
    job_id = job_id or new_job()
    try:
        _set_job(job_id, "running", f"loading {path.name}")
        docs = FileLoader(path).load()
        if not docs:
            raise ValueError("No extractable text (scanned PDF? OCR is not enabled)")
        source_id = docs[0].source_id
        content_hash = hashlib.sha256("".join(d.text for d in docs).encode()).hexdigest()

        with db.connect() as c:
            row = c.execute(
                "SELECT content_hash, embedding_model FROM documents WHERE source_id=?",
                (source_id,),
            ).fetchone()
        if row and row["content_hash"] == content_hash and row["embedding_model"] == settings.embedding_model:
            _set_job(job_id, "done", "unchanged, skipped")
            return {"job_id": job_id, "source_id": source_id, "chunks": 0, "skipped": True}

        chunks = chunk_docs(docs)
        _set_job(job_id, "running", f"embedding {len(chunks)} chunks")
        embeddings = embed_documents([c.text for c in chunks])

        store = get_store()
        store.delete_source(source_id)  # replace, never append
        store.upsert(
            [c.id for c in chunks], [c.text for c in chunks], embeddings, [c.metadata for c in chunks]
        )
        with db.connect() as c:
            c.execute(
                "INSERT OR REPLACE INTO documents"
                "(source_id,title,uri,source_type,content_hash,chunk_count,embedding_model) "
                "VALUES(?,?,?,?,?,?,?)",
                (source_id, docs[0].title, docs[0].uri, docs[0].source_type,
                 content_hash, len(chunks), settings.embedding_model),
            )
        _set_job(job_id, "done", f"{len(chunks)} chunks")
        return {"job_id": job_id, "source_id": source_id, "chunks": len(chunks), "skipped": False}
    except Exception as e:
        _set_job(job_id, "failed", f"{e}\n{traceback.format_exc(limit=3)}")
        raise
