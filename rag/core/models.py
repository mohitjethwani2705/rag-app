"""Single place where models are constructed. Swap models via .env only."""
import json
import threading
from functools import lru_cache

from .config import settings

_usage_lock = threading.Lock()


@lru_cache
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model, device="cpu")


@lru_cache
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model, device="cpu")


class JinaBudgetExceeded(RuntimeError):
    pass


def _usage_file():
    return settings.data_dir / "jina_usage.json"


def jina_tokens_used() -> int:
    try:
        return int(json.loads(_usage_file().read_text())["total_tokens"])
    except Exception:
        return 0


def _record_usage(tokens: int) -> None:
    with _usage_lock:
        _usage_file().parent.mkdir(parents=True, exist_ok=True)
        _usage_file().write_text(json.dumps({"total_tokens": jina_tokens_used() + tokens}))


def _jina(path: str, payload: dict) -> dict:
    import httpx

    budget = settings.jina_token_budget
    if budget and jina_tokens_used() >= budget:
        raise JinaBudgetExceeded(f"Jina token budget of {budget:,} reached; raise JINA_TOKEN_BUDGET to continue.")
    r = httpx.post(
        f"https://api.jina.ai/v1/{path}",
        headers={"Authorization": f"Bearer {settings.jina_api_key}"},
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    body = r.json()
    _record_usage(int(body.get("usage", {}).get("total_tokens", 0)))
    return body


def _jina_embed(texts: list[str], task: str) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), 64):
        data = _jina("embeddings", {
            "model": settings.jina_embedding_model, "task": task,
            "normalized": True, "input": texts[i:i + 64],
        })["data"]
        out += [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
    return out


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """Relevance score per text, in input order."""
    if settings.use_jina:
        res = _jina("rerank", {
            "model": settings.jina_reranker_model, "query": query,
            "documents": [t[:settings.jina_rerank_max_chars] for t in texts], "top_n": len(texts),
        })["results"]
        scores = [0.0] * len(texts)
        for r in res:
            scores[r["index"]] = float(r["relevance_score"])
        return scores
    import torch

    # Raw logits, not sigmoid: sigmoid squashes everything below ~-5 to ~0 and hides the difference
    # between "weakly relevant" and "unrelated".
    return [float(s) for s in get_reranker().predict(
        [(query, t) for t in texts], activation_fn=torch.nn.Identity()
    )]


def embed_documents(texts: list[str]) -> list[list[float]]:
    if settings.use_jina:
        return _jina_embed(texts, "retrieval.passage")
    return get_embedder().encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
    ).tolist()


def embed_query(text: str) -> list[float]:
    if settings.use_jina:
        return _jina_embed([text], "retrieval.query")[0]
    # bge models expect this instruction prefix on queries (not on passages)
    prefix = "Represent this sentence for searching relevant passages: "
    return get_embedder().encode(prefix + text, normalize_embeddings=True).tolist()


@lru_cache
def get_llm():
    """Chat model (Groq or Hugging Face). Raises a clear error if unconfigured."""
    if not settings.llm_configured:
        raise RuntimeError("No LLM configured: set GROQ_API_KEY, or HF_MODEL_ID + HF_TOKEN, in .env")
    if settings.provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_new_tokens,
        )

    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    kwargs = dict(
        huggingfacehub_api_token=settings.hf_token or None,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        task="text-generation",
        streaming=True,
    )
    if settings.hf_endpoint_url:
        endpoint = HuggingFaceEndpoint(endpoint_url=settings.hf_endpoint_url, **kwargs)
    else:
        endpoint = HuggingFaceEndpoint(repo_id=settings.hf_model_id, **kwargs)
    return ChatHuggingFace(llm=endpoint)
