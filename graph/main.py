from uuid import uuid4

from graph.workflow import uni_bot_graph
from langchain_core.messages import HumanMessage


def ask_uni_bot(query: str, thread_id: str):

    query = query.strip()

    if not query:
        yield "Please enter a question."
        return

    streamed = False
    final_answer = None

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    for mode, chunk in uni_bot_graph.stream(
        {
            "query": query,
            "messages": [
                HumanMessage(content=query)
            ]
        },
        config=config,
        stream_mode=["custom", "updates"]
    ):

        if mode == "custom":

            text = chunk.get("text", "")

            if text:
                streamed = True
                yield text

        elif mode == "updates":

            for node_name, node_update in chunk.items():

                if node_name in {
                    "generate_answer",
                    "fallback",
                    "clarify"
                }:
                    if isinstance(node_update, dict):
                        final_answer = node_update.get(
                            "answer",
                            final_answer
                        )

    if not streamed and final_answer:
        yield final_answer


if __name__ == "__main__":

    thread_id = str(uuid4())

    print("Uni-Bot is ready. Type 'exit' to stop.\n")
    print("Conversation ID:", thread_id)

    while True:

        query = input("\nYou: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        print("\nUni-Bot: ", end="", flush=True)

        try:
            for token in ask_uni_bot(query, thread_id):
                print(token, end="", flush=True)

            print("\n")

        except Exception as error:
            print(f"\nError: {error}\n")