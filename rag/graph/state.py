from typing import TypedDict


class RAGState(TypedDict, total=False):
    question: str
    chat_history: list[dict]  # [{"role": "user"|"assistant", "content": str}]
    intent: str  # "question" | "small_talk"
    query: str # standalone search query (rewritten or original)
    docs: list[dict]
    sufficient: bool
    attempts: int
    answer: str
    citations: list[dict]
