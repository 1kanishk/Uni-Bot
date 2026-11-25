
from uuid import uuid4

from graph.workflow import uni_bot_graph
from langchain_core.messages import HumanMessage


query = "Explain merge sort in detail."

config = {
    "configurable": {
        "thread_id": str(uuid4())
    }
}

streamed_parts = []
final_answer = None
stream_events = 0

for mode, chunk in uni_bot_graph.stream(
    {
        "query": query,
        "messages": [HumanMessage(content=query)]
    },
    config=config,
    stream_mode=["custom", "updates"]
):

    if mode == "custom":
        stream_events += 1

        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            if text:
                streamed_parts.append(text)

    elif mode == "updates":
        for node_name, node_update in chunk.items():
            if node_name == "generate_answer":
                final_answer = node_update.get("answer")


streamed = "".join(streamed_parts)

print("\nSTREAM EVENTS:", stream_events)
print("STREAM LENGTH:", len(streamed))
print("FINAL ANSWER LENGTH:", len(final_answer or ""))

print(
    "STREAM EQUALS FINAL:",
    streamed == (final_answer or "")
)

print(
    "STREAM IS EXACTLY DOUBLED:",
    bool(final_answer) and streamed == final_answer + final_answer
)

print("\nSTREAM PREVIEW:")
print(repr(streamed[:300]))

print("\nFINAL ANSWER PREVIEW:")
print(repr((final_answer or "")[:300]))
