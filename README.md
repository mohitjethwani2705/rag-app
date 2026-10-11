# RAG application

FastAPI + LangGraph. Groq (default if `GROQ_API_KEY` is set) or Hugging Face Inference API for generation; embeddings and reranking run locally on CPU. Chroma + SQLite, both plain files under `data/`.

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then set GROQ_API_KEY (or HF_TOKEN + HF_MODEL_ID)
uvicorn rag.api.main:app --reload
```

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
| `GET /healthz` | chunk count, whether the LLM is configured |

Ingest and retrieval work without an LLM configured; `/chat` returns 503 until an LLM is configured. Provider: `LLM_PROVIDER=auto|groq|hf`; Groq model via `GROQ_MODEL`.

## Graph
`rewrite_query -> retrieve (hybrid + rerank) -> grade -> generate`. Rewrite and LLM grading are off by default
(`ENABLE_QUERY_REWRITE`, `ENABLE_LLM_GRADING`); the reranker score gate (`MIN_RERANK_SCORE`) is the default sufficiency signal.
Turn the flags on only if the eval set (next phase) shows they help with your model.

## Swapping the vector store
Implement the `VectorStore` protocol in `rag/retrieval/store.py` (e.g. pgvector) and change `get_store()`.
Changing `EMBEDDING_MODEL` requires re-ingesting: vectors from different models are not comparable. The ingest
pipeline detects this and re-embeds.

## Tests
`pytest` (downloads the embedding and reranker models on first run).
