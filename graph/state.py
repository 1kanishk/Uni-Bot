from typing import Literal

from typing_extensions import NotRequired, TypedDict
from langchain_core.documents import Document
from langgraph.graph import MessagesState



Intent = Literal[
    "university",
    "academic_help",
    "out_of_scope",
    "unclear",
]


class UniBotState(MessagesState):

    query: str
    retrieval_query: str
    intent: NotRequired[Intent]
    retrieved_docs: NotRequired[list[Document]]
    context: NotRequired[str]
    answer: NotRequired[str]
    intent_confidence: NotRequired[float]
    intent_probabilities: NotRequired[dict[str, float]]
