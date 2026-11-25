
import json
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage
from graph.workflow import uni_bot_graph


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "workflow_integration_results_1.json"


def run_turn(query, thread_id):

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

    return result


def normalize(text):
    return "".join(
        char.lower()
        for char in str(text)
        if char.isalnum()
    )


def main():

    checks = {}
    results = {}

    # TEST 1: Academic generation

    academic_thread = str(uuid4())

    academic = run_turn(
        "Explain linear search in simple terms.",
        academic_thread
    )

    checks["academic_generation"] = (
        academic.get("intent") == "academic_help"
        and bool(academic.get("answer", "").strip())
    )

    results["academic"] = {
        "intent": academic.get("intent"),
        "answer": academic.get("answer")
    }

    # TEST 2: University retrieval

    university_thread = str(uuid4())

    university = run_turn(
        "What percentage of internships are paid at JECRC University?",
        university_thread
    )

    checks["university_retrieval"] = (
        university.get("intent") == "university"
        and bool(university.get("retrieved_docs"))
        and bool(university.get("answer", "").strip())
    )

    checks["university_answer"] = (
        "61" in normalize(university.get("answer", ""))
    )

    results["university"] = {
        "intent": university.get("intent"),
        "answer": university.get("answer"),
        "retrieved_docs": len(
            university.get("retrieved_docs", [])
        )
    }

    # TEST 3: Contextual university follow-up

    followup = run_turn(
        "And what is the stipend range?",
        university_thread
    )

    normalized_followup = normalize(
        followup.get("answer", "")
    )

    checks["contextual_memory"] = (
        followup.get("intent") == "university"
        and bool(followup.get("retrieved_docs"))
        and "internships" in followup.get(
            "retrieval_query", ""
        ).lower()
    )

    checks["followup_answer"] = (
        "5000" in normalized_followup
        and "60000" in normalized_followup
    )

    results["followup"] = {
        "intent": followup.get("intent"),
        "retrieval_query": followup.get("retrieval_query"),
        "answer": followup.get("answer"),
        "retrieved_docs": len(
            followup.get("retrieved_docs", [])
        )
    }

    # TEST 4: Out-of-scope fallback

    out_of_scope = run_turn(
        "Give me an aloo paratha recipe.",
        str(uuid4())
    )

    checks["out_of_scope"] = (
        out_of_scope.get("intent") == "out_of_scope"
        and bool(out_of_scope.get("answer", "").strip())
    )

    results["out_of_scope"] = {
        "intent": out_of_scope.get("intent"),
        "answer": out_of_scope.get("answer")
    }

    # TEST 5: Unclear question

    unclear = run_turn(
        "What about that?",
        str(uuid4())
    )

    checks["clarification"] = (
        unclear.get("intent") == "unclear"
        and bool(unclear.get("answer", "").strip())
    )

    results["unclear"] = {
        "intent": unclear.get("intent"),
        "answer": unclear.get("answer")
    }

    # REPORT

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                "checks": checks,
                "passed": all(checks.values()),
                "results": results
            },
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("UNIBOT WORKFLOW INTEGRATION RESULTS")
    print("=" * 60)

    for name, passed in checks.items():
        print(
            f"{name:<30} {'PASS' if passed else 'FAIL'}"
        )

    passed_count = sum(checks.values())

    print("\nTOTAL:", f"{passed_count}/{len(checks)}")
    print(
        "STATUS:",
        "ALL TESTS PASSED" if all(checks.values())
        else "SOME TESTS FAILED"
    )

    print("\nResults saved:", RESULTS_PATH)


if __name__ == "__main__":
    main()
