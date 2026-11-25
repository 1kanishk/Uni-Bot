
import json
import re
import time

from graph.workflow import uni_bot_graph


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def run_test_case(test_case: dict) -> dict:

    query = test_case["query"]

    final_state = {
        "query": query
    }

    executed_nodes = []

    start_time = time.perf_counter()

    for event in uni_bot_graph.stream(
        {"query": query},
        stream_mode="updates"
    ):

        for node_name, update in event.items():

            if node_name.startswith("__"):
                continue

            executed_nodes.append(node_name)
            final_state.update(update)

    execution_time = time.perf_counter() - start_time

    expected_intent = test_case["expected_intent"]

    expected_path = test_case["expected_path"]

    expected_keywords = test_case.get(
        "expected_answer_contains", []
    )

    actual_intent = final_state.get("intent")

    actual_path = executed_nodes

    answer = final_state.get("answer", "")

    normalized_answer = normalize_text(answer)

    intent_correct = actual_intent == expected_intent

    path_correct = actual_path == expected_path

    answer_checks = all(
        normalize_text(keyword) in normalized_answer
        for keyword in expected_keywords
    )

    answer_present = bool(answer.strip())

    min_docs = test_case.get("min_retrieved_docs", 0)

    documents_retrieved = len(
        final_state.get("retrieved_docs", [])
    )

    retrieval_correct = documents_retrieved >= min_docs

    passed = all([
        intent_correct,
        path_correct,
        answer_checks,
        answer_present,
        retrieval_correct
    ])

    return {
        "id": test_case["id"],
        "query": query,
        "expected_intent": expected_intent,
        "actual_intent": actual_intent,
        "executed_nodes": executed_nodes,
        "intent_correct": intent_correct,
        "path_correct": path_correct,
        "answer_checks": answer_checks,
        "retrieval_correct": retrieval_correct,
        "answer_present": answer_present,
        "documents_retrieved": documents_retrieved,
        "execution_time": execution_time,
        "answer": answer,
        "passed": passed
    }


def run_evaluation():

    with open(
        "graph_eval/e2e_dataset.json",
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)

    results = []

    for test_case in dataset:

        print("\n" + "=" * 70)
        print(
            f"TEST {test_case['id']}: {test_case['query']}"
        )
        print("=" * 70)

        try:

            result = run_test_case(test_case)

            results.append(result)

            print("EXPECTED INTENT:", result["expected_intent"])
            print("ACTUAL INTENT:", result["actual_intent"])

            print("EXECUTED NODES:", result["executed_nodes"])

            print("INTENT CORRECT:", result["intent_correct"])
            print("PATH CORRECT:", result["path_correct"])

            print(
                "DOCUMENTS RETRIEVED:",
                result["documents_retrieved"]
            )

            print("ANSWER CHECKS:", result["answer_checks"])

            print("EXECUTION TIME:",
                  round(result["execution_time"], 2), "seconds")

            print("RESULT:", "PASS" if result["passed"] else "FAIL")

        except Exception as error:

            print("ERROR:", str(error))

            results.append({
                "id": test_case["id"],
                "query": test_case["query"],
                "passed": False,
                "error": str(error)
            })

    successful = sum(
        result["passed"]
        for result in results
    )

    total = len(dataset)

    print("\n" + "=" * 70)
    print("END-TO-END EVALUATION SUMMARY")
    print("=" * 70)

    print("Total Questions:", total)
    print("Passed:", successful)
    print("Failed:", total - successful)

    print(
        "End-to-End Pass Rate:",
        round(successful / total * 100, 2),
        "%"
    )

    print("\nDETAILED METRICS:")

    valid_results = [
        result for result in results
        if "error" not in result
    ]

    if valid_results:

        print(
            "Intent Accuracy:",
            round(
                sum(r["intent_correct"] for r in valid_results)
                / len(valid_results) * 100,
                2
            ),
            "%"
        )

        print(
            "Routing Accuracy:",
            round(
                sum(r["path_correct"] for r in valid_results)
                / len(valid_results) * 100,
                2
            ),
            "%"
        )

        print(
            "Answer Check Pass Rate:",
            round(
                sum(r["answer_checks"] for r in valid_results)
                / len(valid_results) * 100,
                2
            ),
            "%"
        )

    with open(
        "graph/e2e_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nResults saved to graph/e2e_results.json")


if __name__ == "__main__":
    run_evaluation()
