from uuid import uuid4

from langchain_core.messages import HumanMessage
from graph.workflow import uni_bot_graph


query = "Explain merge sort."
thread_id = str(uuid4())

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

answer = result.get("answer", "")

print("ANSWER LENGTH:", len(answer))
print("ANSWER REPR:", repr(answer[:300]))

print("\nFINAL ANSWER:\n")
print(answer)