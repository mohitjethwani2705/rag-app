import json
import re

from ..core.config import settings
from ..core.intent import SMALL_TALK, classify_intent
from ..core.models import get_llm
from ..retrieval.hybrid import hybrid_search, rerank
from .state import RAGState

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "End every sentence that states a fact from the context with a citation marker like [1] or [2]. "
    "If the context does not contain the answer, say you could not find it in the documents. "
    "Never invent facts or citations."
)

NO_DOCS_ANSWER = "I couldn't find anything relevant to that in the ingested documents."


SMALL_TALK_PROMPT = (
    "You are a friendly assistant for a document Q&A app. The user's message is small talk "
    "(thanks, greeting, feedback). Reply naturally in one or two short sentences, with no citations, "
    "and invite them to ask about their documents if relevant."
)


def intent(state: RAGState) -> RAGState:
    return {"intent": classify_intent(state["question"], state.get("chat_history"))}


def route_after_intent(state: RAGState) -> str:
    return "small_talk" if state.get("intent") == SMALL_TALK else "rewrite_query"


def rewrite_query(state: RAGState) -> RAGState:
    q = state["question"]
    history = state.get("chat_history") or []
    attempts = state.get("attempts", 0)
    if not settings.enable_query_rewrite or (not history and attempts == 0):
        return {"query": q, "attempts": attempts}
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    prompt = (
        "Rewrite the final question as a standalone search query, resolving pronouns using the "
        f"conversation. Output only the query.\n\nConversation:\n{convo}\n\nQuestion: {q}"
    )
    try:
        out = get_llm().invoke(prompt).content.strip().splitlines()[0]
        return {"query": out or q, "attempts": attempts}
    except Exception:
        return {"query": q, "attempts": attempts}


def retrieve(state: RAGState) -> RAGState:
    query = state["query"]
    docs = rerank(query, hybrid_search(query))
    return {"docs": docs}


def grade(state: RAGState) -> RAGState:
    docs = state.get("docs") or []
    # Primary signal: reranker score (free, deterministic).
    sufficient = bool(docs) and docs[0]["score"] >= settings.rerank_threshold
    if sufficient and settings.enable_llm_grading:
        ctx = "\n\n".join(d["text"][:600] for d in docs)
        prompt = (
            'Do these passages contain enough information to answer the question? '
            'Reply with JSON only: {"sufficient": true} or {"sufficient": false}.\n\n'
            f"Question: {state['question']}\n\nPassages:\n{ctx}"
        )
        try:
            m = re.search(r"\{.*?\}", get_llm().invoke(prompt).content, re.S)
            sufficient = bool(json.loads(m.group())["sufficient"]) if m else sufficient
        except Exception:
            pass  # malformed output from the model: fall back to the reranker signal
    return {"sufficient": sufficient, "attempts": state.get("attempts", 0) + 1}


def route_after_grade(state: RAGState) -> str:
    if state["sufficient"]:
        return "generate"
    retry_enabled = settings.enable_query_rewrite or settings.enable_llm_grading
    if retry_enabled and state["attempts"] < settings.max_attempts:
        return "rewrite_query"
    return "generate"


def build_prompt(state: RAGState) -> list[tuple[str, str]]:
    if state.get("intent") == SMALL_TALK:
        msgs = [("system", SMALL_TALK_PROMPT)]
        for m in (state.get("chat_history") or [])[-6:]:
            msgs.append(("human" if m["role"] == "user" else "ai", m["content"]))
        msgs.append(("human", state["question"]))
        return msgs
    docs = state.get("docs") or []
    ctx = "\n\n".join(
        f"[{i}] ({d['metadata']['title']}, p.{d['metadata']['page']})\n{d['text']}"
        for i, d in enumerate(docs, 1)
    )
    caveat = (
        "" if state.get("sufficient")
        else "\nNOTE: the retrieved context looks weak. Say so plainly if it doesn't answer the question."
    )
    msgs = [("system", SYSTEM_PROMPT + caveat)]
    for m in (state.get("chat_history") or [])[-6:]:
        msgs.append(("human" if m["role"] == "user" else "ai", m["content"]))
    msgs.append(("human", f"Context:\n{ctx}\n\nQuestion: {state['question']}"))
    return msgs


_ABSTAIN = re.compile(
    r"(could not|couldn.t|can.t|cannot|unable to|did not|didn.t) find|no (relevant )?information|not (mentioned|covered|in the)",
    re.I,
)


def citations_for(state: RAGState, answer: str) -> list[dict]:
    docs = state.get("docs") or []
    used = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    inferred = False
    if not used and docs and state.get("sufficient") and not _ABSTAIN.search(answer):
        # The model answered without citing: show the best passage rather than no source at all.
        # Only when retrieval was good, so small talk ("thanks!") never gets a made-up source.
        used, inferred = {1}, True
    return [
        {"n": i, "title": d["metadata"]["title"], "page": d["metadata"]["page"],
         "uri": d["metadata"]["uri"], "score": round(d["score"], 3), "snippet": d["text"][:300],
         "inferred": inferred}
        for i, d in enumerate(docs, 1) if i in used
    ]


def generate(state: RAGState) -> RAGState:
    if not state.get("docs") and state.get("intent") != SMALL_TALK:
        return {"answer": NO_DOCS_ANSWER, "citations": []}
    answer = get_llm().invoke(build_prompt(state)).content
    return {"answer": answer, "citations": citations_for(state, answer)}
