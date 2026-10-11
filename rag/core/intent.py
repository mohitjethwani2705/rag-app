"""Intent gate: decide whether a message is a real question or small talk.

Every message is classified by inception/mercury-decide:free on OpenRouter (System One schema).
No rules, no length shortcuts, no other model. If the call fails (or OPENROUTER_API_KEY is unset)
the message is treated as a real question so normal retrieval is never blocked.
"""
import logging

import httpx

from .config import settings

log = logging.getLogger(__name__)

QUESTION, SMALL_TALK = "question", "small_talk"


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
    if not settings.openrouter_api_key:
        log.warning("OPENROUTER_API_KEY not set; treating message as a question")
        return QUESTION
    last = next((m["content"] for m in reversed(chat_history or []) if m["role"] != "user"), "")
    try:
        return _mercury(question.strip(), last)
    except Exception as e:
        log.warning("mercury-decide failed, treating as question: %s", e)
        return QUESTION
