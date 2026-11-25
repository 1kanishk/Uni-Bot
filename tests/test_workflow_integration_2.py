import json
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage

from graph.workflow import uni_bot_graph


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "workflow_integration_results_2.json"


def run_turn(query, thread_id):
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    graph_input = {
        "query": query,
        "messages": [HumanMessage(content=query)]
    }

    final_state = {}
    executed_nodes = []

    for event in uni_bot_graph.stream(
        graph_input,
        config=config,
        stream_mode="updates"
    ):
        for node_name, update in event.items():
            executed_nodes.append(node_name)

            if isinstance(update, dict):
                final_state.update(update)

    return {
        "query": query,
        "intent": final_state.get("intent"),
        "answer": final_state.get("answer", ""),
        "executed_nodes": executed_nodes,
        "retrieved_docs": final_state.get("retrieved_docs", []),
        "retrieval_query": final_state.get("retrieval_query", "")
    }


def normalize(value):
    return "".join(
        char.lower()
        for char in str(value)
        if char.isalnum()
    )


def contains_all_groups(answer, groups):
    normalized = normalize(answer)

    return all(
        any(normalize(term) in normalized for term in group)
        for group in groups
    )


def summarize(result):
    return {
        "query": result["query"],
        "intent": result["intent"],
        "executed_nodes": result["executed_nodes"],
        "answer": result["answer"],
        "retrieved_chunk_count": len(result["retrieved_docs"]),
        "retrieved_sources": [
            {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page")
            }
            for doc in result["retrieved_docs"]
        ],
        "retrieval_query": result["retrieval_query"]
    }


def main():
    checks = {}
    results = {}

    thread_id = str(uuid4())

    university = run_turn(
        "What percentage of internships are paid at JECRC University?",
        thread_id
    )

    checks["university_routing"] = (
        university["intent"] == "university"
        and university["executed_nodes"] == [
            "classify_intent",
            "retrieve_docs",
            "generate_answer"
        ]
        and len(university["retrieved_docs"]) > 0
    )

    checks["university_answer"] = (
        bool(university["answer"].strip())
        and "61" in normalize(university["answer"])
    )

    followup = run_turn(
        "And what about the monthly amount?",
        thread_id
    )

    checks["conversation_memory"] = (
        followup["intent"] == "university"
        and followup["executed_nodes"] == [
            "classify_intent",
            "retrieve_docs",
            "generate_answer"
        ]
        and "internship" in followup["retrieval_query"].lower()
        and len(followup["retrieved_docs"]) > 0
    )

    checks["followup_answer"] = contains_all_groups(
        followup["answer"],
        [
            ["5000", "5k"],
            ["60000", "60k"]
        ]
    )

    academic = run_turn(
        "Explain binary search and why the input must be sorted.",
        str(uuid4())
    )

    checks["academic_help"] = (
        academic["intent"] == "academic_help"
        and academic["executed_nodes"] == [
            "classify_intent",
            "generate_answer"
        ]
        and contains_all_groups(
            academic["answer"],
            [["binary search"], ["sorted"]]
        )
    )

    out_of_scope = run_turn(
        "Recommend a smartphone under 25000 rupees.",
        str(uuid4())
    )

    checks["out_of_scope"] = (
        out_of_scope["intent"] == "out_of_scope"
        and out_of_scope["executed_nodes"] == [
            "classify_intent",
            "fallback"
        ]
        and bool(out_of_scope["answer"].strip())
    )

    unclear = run_turn(
        "What are the requirements?",
        str(uuid4())
    )

    checks["clarification"] = (
        unclear["intent"] == "unclear"
        and unclear["executed_nodes"] == [
            "classify_intent",
            "clarify"
        ]
        and bool(unclear["answer"].strip())
    )

    isolated = run_turn(
        "And what about the monthly amount?",
        str(uuid4())
    )

    checks["thread_isolation"] = (
        isolated["intent"] == "unclear"
        and isolated["executed_nodes"] == [
            "classify_intent",
            "clarify"
        ]
    )

    results = {
        "university": summarize(university),
        "followup": summarize(followup),
        "academic_help": summarize(academic),
        "out_of_scope": summarize(out_of_scope),
        "unclear": summarize(unclear),
        "isolated_followup": summarize(isolated)
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "checks": checks,
                "passed": all(checks.values()),
                "results": results
            },
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 55)
    print("UNI-BOT END-TO-END WORKFLOW TEST")
    print("=" * 55)

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print("\n" + "=" * 55)
    print(f"TOTAL: {sum(checks.values())}/{len(checks)}")
    print("RESULT:", "PASS" if all(checks.values()) else "FAIL")
    print("=" * 55)

    print("\nUNIVERSITY ANSWER:\n", university["answer"])
    print("\nFOLLOW-UP ANSWER:\n", followup["answer"])
    print("\nACADEMIC ANSWER:\n", academic["answer"])

    print(f"\nDetailed results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()