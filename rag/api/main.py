import base64
import json
import re
import secrets
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from ..core import db
from ..core.config import settings
from ..graph.build import get_graph, get_retrieval_graph
from ..graph.nodes import NO_DOCS_ANSWER, build_prompt, citations_for
from ..core.models import JinaBudgetExceeded, get_llm, jina_tokens_used
from ..ingest.loaders.files import SUPPORTED
from ..ingest.pipeline import ingest_file, new_job
from ..retrieval.store import get_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="RAG API", lifespan=lifespan)


def _rate_limit_message(exc: Exception) -> str:
    """Friendly text for a provider rate limit (e.g. Groq's free-tier daily token cap)."""
    wait = ""
    try:
        m = re.search(r"try again in ([\dhms.]+)", str(exc))
        wait = f" Try again in about {m.group(1).rstrip('.')}." if m else ""
    except Exception:
        pass
    return "The AI service is rate limited right now (free-tier usage cap)." + wait


try:
    from groq import RateLimitError as _GroqRateLimit
except ImportError:  # groq is only needed when LLM_PROVIDER=groq
    _GroqRateLimit = ()

if _GroqRateLimit:
    @app.exception_handler(_GroqRateLimit)
    async def _on_rate_limit(request: Request, exc: Exception):
        return JSONResponse(status_code=429, content={"detail": _rate_limit_message(exc)})


@app.middleware("http")
async def _require_password(request: Request, call_next):
    if settings.app_password and request.url.path != "/healthz":
        ok = False
        header = request.headers.get("authorization", "")
        if header.lower().startswith("basic "):
            try:
                _, _, pw = base64.b64decode(header[6:]).decode().partition(":")
                ok = secrets.compare_digest(pw.encode(), settings.app_password.encode())
            except Exception:
                pass
        if not ok:
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="RAG"'})
    return await call_next(request)


@app.exception_handler(JinaBudgetExceeded)
async def _on_jina_budget(request: Request, exc: Exception):
    return JSONResponse(status_code=429, content={"detail": str(exc)})


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


class ChatRequest(BaseModel):
    question: str
    chat_history: list[dict] = []


def _safe_ingest(path: Path, job_id: str):
    try:
        ingest_file(path, job_id)
    except Exception:
        pass  # failure is recorded on the job row


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "chunks": get_store().count(),
        "llm_configured": settings.llm_configured,
        "llm_provider": settings.provider,
        "embedding_model": settings.jina_embedding_model if settings.use_jina else settings.embedding_model,
        "jina_tokens_used": jina_tokens_used() if settings.use_jina else None,
        "jina_token_budget": settings.jina_token_budget or None,
    }


@app.post("/ingest", status_code=202)
async def ingest(background: BackgroundTasks, file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(400, f"Unsupported type. Allowed: {sorted(SUPPORTED)}")
    dest = settings.upload_dir / name
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    with dest.open("wb") as f:
        while chunk := file.file.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File too large (max {settings.max_upload_mb} MB)")
            f.write(chunk)
    job_id = new_job()
    background.add_task(_safe_ingest, dest, job_id)
    return {"job_id": job_id, "file": name}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    with db.connect() as c:
        row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Unknown job")
    return dict(row)


@app.get("/documents")
def documents():
    with db.connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM documents ORDER BY ingested_at DESC")]


@app.delete("/documents/{source_id}")
def delete_document(source_id: str):
    get_store().delete_source(source_id)
    with db.connect() as c:
        c.execute("DELETE FROM documents WHERE source_id=?", (source_id,))
    return {"deleted": source_id}


def _require_llm():
    if not settings.llm_configured:
        raise HTTPException(503, "LLM not configured: set GROQ_API_KEY, or HF_MODEL_ID and HF_TOKEN, in .env")


@app.post("/chat")
async def chat(req: ChatRequest):
    _require_llm()
    state = {"question": req.question, "chat_history": req.chat_history}
    result = await run_in_threadpool(get_graph().invoke, state)
    return {
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "sufficient": result.get("sufficient"),
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE: `token` events, then a `citations` event, then `done`."""
    _require_llm()

    async def events():
        state = await run_in_threadpool(
            get_retrieval_graph().invoke,
            {"question": req.question, "chat_history": req.chat_history},
        )
        if not state.get("docs") and state.get("intent") != "small_talk":
            yield f"event: token\ndata: {json.dumps(NO_DOCS_ANSWER)}\n\n"
            yield "event: done\ndata: {}\n\n"
            return
        answer = ""
        try:
            async for chunk in get_llm().astream(build_prompt(state)):
                if not chunk.content:  # reasoning models emit empty-content chunks while thinking
                    continue
                answer += chunk.content
                yield f"event: token\ndata: {json.dumps(chunk.content)}\n\n"
        except Exception as e:
            msg = _rate_limit_message(e) if _GroqRateLimit and isinstance(e, _GroqRateLimit) else str(e)
            yield f"event: error\ndata: {json.dumps(msg)}\n\n"
            return
        yield f"event: citations\ndata: {json.dumps(citations_for(state, answer))}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
