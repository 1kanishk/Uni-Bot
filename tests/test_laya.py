from graph.nodes import classify_intent_node

questions = [
    "What is the annual fee for BCA Core?",
    "Explain binary search in Python.",
    "Plan a five-day trip to Goa.",
    "What about that?"
]

for question in questions:

    result = classify_intent_node({
        "query": question
    })

    print("\nQuestion:", question)
    print("Intent:", result["intent"])
    print("Confidence:", result["intent_confidence"])
    print("Probabilities:", result["intent_probabilities"])