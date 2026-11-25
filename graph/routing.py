
from typing import Literal

from .state import UniBotState


def route_after_classification(state: UniBotState) -> Literal[
    "retrieve_docs",
    "generate_answer",
    "fallback",
    "clarify"
]:

    intent = state.get("intent", "unclear")

    if intent == "university":
        return "retrieve_docs"

    if intent == "academic_help":
        return "generate_answer"

    if intent == "out_of_scope":
        return "fallback"

    return "clarify"


def route_after_retrieval(
    state: UniBotState
) -> Literal["generate_answer", "fallback"]:

    documents = state.get("retrieved_docs", [])
    context = state.get("context", "")

    if documents and context.strip():
        return "generate_answer"

    return "fallback"
