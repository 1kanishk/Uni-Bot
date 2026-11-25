
from uuid import uuid4

from langchain_core.messages import HumanMessage
from graph.workflow import uni_bot_graph


def ask(query, thread_id):

    result = uni_bot_graph.invoke(
        {
            "query": query,
            "messages": [
                HumanMessage(content=query)
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    print("\nQUESTION:", query)
    print("INTENT:", result.get("intent"))
    print("RETRIEVAL QUERY:", result.get("retrieval_query"))
    print("ANSWER:", result.get("answer"))

    return result


if __name__ == "__main__":

    university_thread = str(uuid4())

    print("\nUNIVERSITY CONVERSATION")
    print("=" * 60)

    ask(
        "What is the annual fee for BCA Core?",
        university_thread
    )

    ask(
        "What about Industry Specialization?",
        university_thread
    )

    academic_thread = str(uuid4())

    print("\nACADEMIC CONVERSATION")
    print("=" * 60)

    ask(
        "Explain merge sort.",
        academic_thread
    )

    ask(
        "Give me a Python example.",
        academic_thread
    )
