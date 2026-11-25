
from graph.routing import route_after_classification


test_cases = [
    {
        "query": "What is the annual fee for BCA Core?",
        "intent": "university"
    },
    {
        "query": "Explain binary search in Python.",
        "intent": "academic_help"
    },
    {
        "query": "Plan a trip to Goa.",
        "intent": "out_of_scope"
    },
    {
        "query": "What about that?",
        "intent": "unclear"
    }
]


for state in test_cases:

    destination = route_after_classification(state)

    print("Question:", state["query"])
    print("Intent:", state["intent"])
    print("Next Node:", destination)
    print("-" * 40)
