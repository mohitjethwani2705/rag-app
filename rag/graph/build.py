from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from .nodes import generate, grade, intent, retrieve, rewrite_query, route_after_grade, route_after_intent
from .state import RAGState


def _build(include_generate: bool):
    g = StateGraph(RAGState)
    g.add_node("intent", intent)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_edge(START, "intent")
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("retrieve", "grade")
    skip = "generate" if include_generate else END  # small talk skips retrieval entirely
    g.add_conditional_edges("intent", route_after_intent, {"small_talk": skip, "rewrite_query": "rewrite_query"})
    if include_generate:
        g.add_node("generate", generate)
        g.add_conditional_edges(
            "grade", route_after_grade, {"generate": "generate", "rewrite_query": "rewrite_query"}
        )
        g.add_edge("generate", END)
    else:
        # Retrieval-only graph: the API streams generation token-by-token itself.
        g.add_conditional_edges(
            "grade", route_after_grade, {"generate": END, "rewrite_query": "rewrite_query"}
        )
    return g.compile()


@lru_cache
def get_graph():
    return _build(include_generate=True)


@lru_cache
def get_retrieval_graph():
    return _build(include_generate=False)
