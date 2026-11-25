from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .nodes import (
    classify_intent_node,
    retrieve_docs_node,
    generate_answer_node,
    fallback_node,
    clarify_node,
)

from .routing import (
    route_after_classification,
    route_after_retrieval,
)

from .state import UniBotState


def build_uni_bot_graph():

    builder = StateGraph(UniBotState)

    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("retrieve_docs", retrieve_docs_node)
    builder.add_node("generate_answer", generate_answer_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("clarify", clarify_node)

    builder.add_edge(START, "classify_intent")

    builder.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "retrieve_docs": "retrieve_docs",
            "generate_answer": "generate_answer",
            "fallback": "fallback",
            "clarify": "clarify",
        },
    )

    builder.add_conditional_edges(
        "retrieve_docs",
        route_after_retrieval,
        {
            "generate_answer": "generate_answer",
            "fallback": "fallback",
        },
    )

    builder.add_edge("generate_answer", END)
    builder.add_edge("fallback", END)
    builder.add_edge("clarify", END)

    checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


uni_bot_graph = build_uni_bot_graph()
