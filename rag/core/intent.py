"""Intent gate: decide whether a message is a real question or small talk. Free, no extra service.

1. Rules: obvious thanks/greetings/goodbyes are caught instantly with a regex (no API call).
2. Short, unclear messages go to inception/mercury-decide:free on OpenRouter (free structured-decision model,
   same /v1/systemone schema as Jev). If OPENROUTER_API_KEY is unset or the call fails, a small Groq model decides.
3. Long messages, or any failure, count as a real question so normal retrieval is never blocked.
"""
import json
import logging
import re
from functools import lru_cache

import httpx

from .config import settings

log = logging.getLogger(__name__)

QUESTION, SMALL_TALK = "question", "small_talk"

_PHRASE = re.compile(
    r"^(hi+|hello+|hey+|yo|good (morning|afternoon|evening|night)|thanks?( you)?( (so|very) much| a lot)?|thank u|thx|ty|"
    r"ok(ay)?|cool|great|nice|awesome|perfect|good|got it|understood|bye|goodbye|see you|lol|"
    r"(it |that |this )?(works?|worked) (well|great|fine|perfectly)( for me)?|(that|it) (helps|helped)|well done|good job|"
    r"nice one|all good|sounds good)$",
    re.I,
)


def _is_small_talk_rule(q: str) -> bool:
    parts = [p.strip() for p in re.split(r"[,.!;:]+|and", q) if p.strip()]
    return bool(parts) and all(_PHRASE.match(p) for p in parts)


_PROMPT = (
    "Classify the user's message. Reply with JSON only: {\"intent\": \"small_talk\"} or {\"intent\": \"question\"}.\n"
    "small_talk = greeting, thanks, praise/feedback on the previous reply, goodbye, chit-chat with nothing to look up.\n"
    "question = asks for information or help, including vague ones like 'what is this document about'.\n"
    "When unsure, answer question.\n\n"
    "Previous assistant reply: {last}\nUser message: {msg}"
)


@lru_cache
def _classifier():
    if settings.provider != "groq":
        from .models import get_llm

        return get_llm()
    from langchain_groq import ChatGroq

    return ChatGroq(model=settings.intent_model or settings.groq_model, api_key=settings.groq_api_key,
                    temperature=0, max_tokens=400)


def _mercury(q: str, last: str) -> str:
    """Ask mercury-decide a single Choice question. Request/response follow the System One schema."""
    body = {
        "model": settings.openrouter_intent_model,
        "state": {"previous_assistant_reply": last[:400], "user_message": q},
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "Is the user_message a real request for information/help, or just small talk?",
                "criteria": {
                    QUESTION: "Asks for facts, explanations or help, including vague ones like 'what is this document about'.",
                    SMALL_TALK: "Greeting, thanks, praise/feedback on the previous reply, goodbye, or other chit-chat.",
                },
            }
        },
    }
    r = httpx.post(
        settings.openrouter_systemone_url, json=body, timeout=10,
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
    )
    r.raise_for_status()
    ans = r.json()["answers"]["intent"]
    probs = ans.get("probabilities") or {}
    # Only skip retrieval when clearly small talk; any doubt means a real question.
    ok = ans.get("choice") == SMALL_TALK and probs.get(SMALL_TALK, 1.0) >= settings.intent_threshold
    return SMALL_TALK if ok else QUESTION


def classify_intent(question: str, chat_history: list[dict] | None = None) -> str:
    q = question.strip()
    if _is_small_talk_rule(q):
        return SMALL_TALK
    if len(q.split()) > settings.intent_max_words or "?" in q:
        return QUESTION  # clearly a real question; skip the API call
    last = next((m["content"] for m in reversed(chat_history or []) if m["role"] != "user"), "")
    if settings.openrouter_api_key:
        try:
            return _mercury(q, last)
        except Exception as e:
            log.warning("mercury-decide failed, falling back to Groq: %s", e)
    if not settings.llm_configured:
        return QUESTION
    try:
        out = _classifier().invoke(_PROMPT.replace("{last}", last[:400]).replace("{msg}", q)).content
        m = re.search(r"\{.*?\}", out, re.S)
        if m and json.loads(m.group()).get("intent") == SMALL_TALK:
            return SMALL_TALK
    except Exception as e:
        log.warning("intent gate failed, treating as question: %s", e)
    return QUESTION
