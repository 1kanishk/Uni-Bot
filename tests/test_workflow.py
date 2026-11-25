
from graph.workflow import uni_bot_graph


questions = [
    "What is the annual fee for BCA Core?",
    "Explain binary search in Python.",
    "Plan a five-day trip to Goa.",
    "What about that?"
]


for question in questions:

    print("\n" + "=" * 60)
    print("QUESTION:", question)

    result = uni_bot_graph.invoke({
        "query": question
    })

    print("INTENT:", result.get("intent"))
    print("ANSWER:", result.get("answer"))
