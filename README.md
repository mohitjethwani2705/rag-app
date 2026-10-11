# RAG application

Chat with your documents. FastAPI + LangGraph, with answers that cite their sources.

- **Decision API intent gate:** every message is classified by the OpenRouter decision API (`inception/mercury-decide:free`) as small talk or a real question. Small talk skips retrieval.
- **Hybrid retrieval:** vector search + BM25 keyword search, merged with reciprocal rank fusion, then reranked.
- **Hosted models:** Jina for embeddings and reranking, Groq for generation. No local models, so memory stays small (~125 MB idle). Without a `JINA_API_KEY` it falls back to local CPU models (`bge-small` + `bge-reranker-base`, needs `sentence-transformers`).
- **Storage:** Chroma + SQLite, plain files under `data/`.

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then fill in the keys below
uvicorn rag.api.main:app --reload
```

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | answer generation (free tier is capped per day) |
| `JINA_API_KEY` | hosted embeddings + reranking; unset = local models |
| `OPENROUTER_API_KEY` | intent decision API; unset = every message is treated as a question |
| `APP_PASSWORD` | optional shared password (browser login box, any username); set it on any public deployment |
| `MAX_UPLOAD_MB` | upload size cap (default 10) |
| `JINA_TOKEN_BUDGET` | optional hard stop on Jina tokens (0 = unlimited); usage is shown in `/healthz` |

Hugging Face generation is still supported (`LLM_PROVIDER=hf`, `HF_TOKEN`, `HF_MODEL_ID`) but Groq is the default when `GROQ_API_KEY` is set.

## Web UI
Run the server, then open http://localhost:8000 - upload files on the left, chat on the right (streaming answers, expandable sources).

## Terminal
```
python -m rag.cli ingest path/to/file_or_folder
python -m rag.cli ask "your question"
python -m rag.cli chat        # interactive, remembers the conversation
python -m rag.cli docs        # list indexed documents
```

## API
| Endpoint | Purpose |
|---|---|
| `POST /ingest` (multipart `file`) | pdf/docx/txt/md; returns a `job_id`, runs in background |
| `GET /jobs/{id}` | queued / running / done / failed |
| `GET /documents`, `DELETE /documents/{source_id}` | manage the index |
| `POST /chat` | `{question, chat_history}` -> answer + citations |
| `POST /chat/stream` | SSE: `token` events, `citations`, `done` |
| `GET /healthz` | chunk count, LLM status, Jina token usage (always open, no password) |

Ingest and retrieval work without an LLM configured; `/chat` returns 503 until one is. Provider rate limits return a readable 429.

## Graph
`intent -> rewrite_query -> retrieve (hybrid + rerank) -> grade -> generate`.
Small talk goes straight from `intent` to `generate`. Query rewrite and LLM grading are off by default
(`ENABLE_QUERY_REWRITE`, `ENABLE_LLM_GRADING`); the reranker score gate is the default sufficiency signal
(`JINA_MIN_RERANK_SCORE`, 0-1 scale, or `MIN_RERANK_SCORE` for local models). Both thresholds are uncalibrated starting values: tune them on your own documents.

## Deploy on Render (free)
1. Push to GitHub, then Render -> New -> Blueprint -> pick the repo (reads `render.yaml`, installs `requirements-prod.txt`).
2. Enter `GROQ_API_KEY`, `JINA_API_KEY`, `OPENROUTER_API_KEY`, `APP_PASSWORD` when prompted.

Free tier limits: no persistent disk (the index and uploads reset on every restart or redeploy, so re-upload documents) and the service sleeps after ~15 minutes idle. Use a paid plan with a disk, or another host with a volume, to keep data.

## Swapping the vector store
Implement the `VectorStore` protocol in `rag/retrieval/store.py` (e.g. pgvector) and change `get_store()`.
Switching embedding models (including local <-> Jina) changes the vector size and makes the old index unusable: delete `data/` and re-ingest.

## Evaluation
`eval/` holds a 40-question set over sample docs and a runner (`python -u -m eval.run`) that scores answers with an LLM judge, retrieval hits, and refusals on unanswerable questions. The sample corpus is tiny, so treat the numbers as a smoke test, not a benchmark.

## Tests
`pytest` (the test uses the local embedding and reranker models, so it needs `sentence-transformers` from `requirements.txt` and downloads the models on first run).
