
from graph.nodes import is_context_dependent_query


TEST_CASES = [
    (True, "When is it?"),
    (True, "When does it begin?"),
    (True, "Can I apply for it?"),
    (True, "Can I register for that?"),
    (True, "Am I eligible for that?"),
    (True, "How much does it cost?"),
    (True, "How much is the fee?"),
    (True, "What are the requirements?"),
    (True, "What are the criteria?"),
    (True, "What is the process?"),
    (True, "What's the procedure?"),
    (True, "How do I do that?"),
    (True, "Where can I find it?"),
    (True, "Does this have a deadline?"),
    (True, "Is that available?"),
    (True, "Tell me more about that."),
    (True, "What about that?"),
    (True, "Which one is better?"),
    (True, "What happens next?"),
    (True, "What is next?"),

    (False, "What is the BCA admission process?"),
    (False, "What are the requirements for BCA?"),
    (False, "How much does BCA cost?"),
    (False, "When does the BCA programme start?"),
    (False, "Can I apply for BCA?"),
    (False, "Explain Python decorators."),
    (False, "What is Python and how does it work?"),
    (False, "Which one is better, BCA Core or BCA Industry Specialization?"),
    (False, "What is the application deadline for BCA?"),
    (False, "How much is the hostel fee for BCA?"),
]


passed = 0

for expected, query in TEST_CASES:
    predicted = is_context_dependent_query(query)
    status = "PASS" if predicted == expected else "FAIL"

    print(
        f"{status} | expected={expected!s:<5} "
        f"predicted={predicted!s:<5} | {query}"
    )

    passed += predicted == expected

print(f"\nResult: {passed}/{len(TEST_CASES)}")
