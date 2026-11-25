from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

from graph.workflow import uni_bot_graph


DATASET_PATH = Path("graph_eval/e2e_dataset.json")
RESULTS_PATH = Path("graph_eval/e2e_results.json")
DEFAULT_K = 5


def normalize_text(value: Any) -> str:
    """Normalize text for robust phrase and filename comparisons."""
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def source_basename(source: Any) -> str:
    """Normalize Windows/Linux paths to a comparable filename."""
    return str(source or "").replace("\\", "/").rsplit("/", 1)[-1].casefold()


def get_document_identity(document: Any) -> tuple[str, str | None]:
    """Return (source filename, page) for a LangChain Document."""
    metadata = getattr(document, "metadata", {}) or {}
    source = (
        metadata.get("source")
        or metadata.get("file_name")
        or metadata.get("filename")
        or ""
    )
    page = metadata.get("page")
    if page is None:
        page = metadata.get("page_number")
    if page is None:
        page = metadata.get("page_label")

    page_value = None if page is None else str(page).strip()
    return source_basename(source), page_value


def unique_retrieved_pages(documents: list[Any]) -> list[tuple[str, str | None]]:
    """Deduplicate chunks by source/page while preserving retrieval rank."""
    identities: list[tuple[str, str | None]] = []
    seen: set[tuple[str, str | None]] = set()

    for document in documents:
        identity = get_document_identity(document)
        if identity[0] and identity not in seen:
            identities.append(identity)
            seen.add(identity)

    return identities


def expected_document_identities(
    expected_documents: list[dict[str, Any]],
) -> set[tuple[str, str | None]]:
    identities: set[tuple[str, str | None]] = set()
    for item in expected_documents:
        source = source_basename(item.get("source"))
        page = item.get("page")
        identities.add((source, None if page is None else str(page).strip()))
    return identities


def evaluate_retrieval(
    retrieved_documents: list[Any],
    expected_documents: list[dict[str, Any]],
    k: int = DEFAULT_K,
) -> dict[str, Any] | None:
    """Compute page-level Precision@K, Recall@K and reciprocal rank.

    Returns None if the dataset case has no manually labelled relevant pages.
    Multiple chunks from the same source page count as one retrieved page.
    """
    if not expected_documents:
        return None

    relevant = expected_document_identities(expected_documents)
    ranked_pages = unique_retrieved_pages(retrieved_documents)[:k]
    hits = [identity for identity in ranked_pages if identity in relevant]

    precision_at_k = len(hits) / k
    recall_at_k = len(hits) / len(relevant)

    reciprocal_rank = 0.0
    for rank, identity in enumerate(ranked_pages, start=1):
        if identity in relevant:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "reciprocal_rank": reciprocal_rank,
        "relevant_pages_found": len(hits),
        "relevant_pages_expected": len(relevant),
        "retrieved_pages": [
            {"source": source, "page": page}
            for source, page in ranked_pages
        ],
        "all_relevant_pages_found": relevant.issubset(set(ranked_pages)),
    }


def check_answer(
    answer: str,
    test_case: dict[str, Any],
    retrieved_documents: list[Any],
) -> dict[str, Any]:
    """Run deterministic content, forbidden-claim and citation checks.

    These checks are intentionally not presented as a semantic truth judge.
    They verify explicitly labelled facts and citations without another LLM.
    """
    normalized_answer = normalize_text(answer)

    required = test_case.get("required_answer_contains", [])
    forbidden = test_case.get("forbidden_answer_contains", [])

    missing_required = [
        phrase for phrase in required
        if normalize_text(phrase) not in normalized_answer
    ]
    found_forbidden = [
        phrase for phrase in forbidden
        if normalize_text(phrase) in normalized_answer
    ]

    answer_present = bool(answer and answer.strip())

    citations_required = bool(test_case.get("require_expected_citations", False))
    expected_docs = test_case.get("relevant_documents", [])
    retrieved_identities = set(unique_retrieved_pages(retrieved_documents))
    missing_citations: list[dict[str, Any]] = []

    if citations_required:
        for expected in expected_docs:
            source = expected.get("source", "")
            page = expected.get("page")
            expected_identity = (
                source_basename(source),
                None if page is None else str(page).strip(),
            )

            source_in_answer = normalize_text(source) in normalized_answer
            page_in_answer = (
                page is not None
                and normalize_text(page) in normalized_answer
            )
            page_was_retrieved = expected_identity in retrieved_identities

            if not (source_in_answer and page_in_answer and page_was_retrieved):
                missing_citations.append({
                    "source": source,
                    "page": page,
                    "source_in_answer": source_in_answer,
                    "page_in_answer": page_in_answer,
                    "page_was_retrieved": page_was_retrieved,
                })

    citation_check = not missing_citations
    required_check = not missing_required
    forbidden_check = not found_forbidden

    return {
        "answer_present": answer_present,
        "required_phrases_pass": required_check,
        "missing_required_phrases": missing_required,
        "forbidden_phrases_pass": forbidden_check,
        "found_forbidden_phrases": found_forbidden,
        "citation_check_applicable": citations_required,
        "citation_check_pass": citation_check,
        "missing_citations": missing_citations,
        "answer_checks_pass": all([
            answer_present,
            required_check,
            forbidden_check,
            citation_check,
        ]),
    }


def run_case(test_case: dict[str, Any]) -> dict[str, Any]:
    query = test_case["query"]
    final_state: dict[str, Any] = {"query": query}
    executed_nodes: list[str] = []
    node_times: dict[str, float] = {}
    previous_update_time = time.perf_counter()
    start_time = previous_update_time

    # This is the only graph invocation for this test case. All node results
    # below come from this execution; the retriever/generator are not called
    # separately a second time by the evaluation script.
    for event in uni_bot_graph.stream(
        {"query": query},
        stream_mode="updates",
    ):
        for node_name, update in event.items():
            if node_name.startswith("__"):
                continue

            now = time.perf_counter()
            node_times[node_name] = round(now - previous_update_time, 4)
            previous_update_time = now
            executed_nodes.append(node_name)
            final_state.update(update)

    total_seconds = time.perf_counter() - start_time

    retrieved_documents = final_state.get("retrieved_docs", []) or []
    answer = final_state.get("answer", "") or ""

    expected_intent = test_case["expected_intent"]
    expected_path = test_case["expected_path"]
    actual_intent = final_state.get("intent")

    intent_pass = actual_intent == expected_intent
    path_pass = executed_nodes == expected_path

    min_chunks = int(test_case.get("min_retrieved_chunks", 0))
    retrieval_count_pass = len(retrieved_documents) >= min_chunks

    retrieval_metrics = evaluate_retrieval(
        retrieved_documents,
        test_case.get("relevant_documents", []),
        k=int(test_case.get("retrieval_k", DEFAULT_K)),
    )

    retrieval_relevance_pass = (
        True if retrieval_metrics is None
        else retrieval_metrics["relevant_pages_found"] > 0
    )

    answer_metrics = check_answer(
        answer,
        test_case,
        retrieved_documents,
    )

    checks = {
        "intent": intent_pass,
        "path": path_pass,
        "minimum_retrieval_count": retrieval_count_pass,
        "labelled_retrieval_relevance": retrieval_relevance_pass,
        "answer_and_citations": answer_metrics["answer_checks_pass"],
    }

    return {
        "id": test_case["id"],
        "query": query,
        "expected_intent": expected_intent,
        "actual_intent": actual_intent,
        "expected_path": expected_path,
        "actual_path": executed_nodes,
        "intent_confidence": final_state.get("intent_confidence"),
        "node_times_seconds": node_times,
        "total_time_seconds": round(total_seconds, 3),
        "retrieved_chunk_count": len(retrieved_documents),
        "retrieved_pages": [
            {"source": source, "page": page}
            for source, page in unique_retrieved_pages(retrieved_documents)
        ],
        "retrieval_metrics": retrieval_metrics,
        "retrieval_count_pass": retrieval_count_pass,
        "retrieval_relevance_pass": retrieval_relevance_pass,
        "answer_metrics": answer_metrics,
        "answer": answer,
        "checks": checks,
        "passed": all(checks.values()),
    }


def print_case_result(result: dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print(f"{result['id']}: {result['query']}")
    print("=" * 78)
    print("Intent:", result["actual_intent"],
          "| Expected:", result["expected_intent"])
    print("Executed path:", " -> ".join(result["actual_path"]))
    print("Retrieved chunks:", result["retrieved_chunk_count"])

    print("Retrieved source/pages:")
    for item in result["retrieved_pages"]:
        print(f"  - {item['source']} | page={item['page']}")

    if result["retrieval_metrics"] is not None:
        metrics = result["retrieval_metrics"]
        print(
            "Retrieval: "
            f"P@K={metrics['precision_at_k']:.3f}, "
            f"R@K={metrics['recall_at_k']:.3f}, "
            f"RR={metrics['reciprocal_rank']:.3f}"
        )
    else:
        print("Retrieval relevance: NOT SCORED (no gold pages labelled)")

    print("Answer checks:", result["answer_metrics"])
    print("Node timings:", result["node_times_seconds"])
    print("Total seconds:", result["total_time_seconds"])
    print("Checks:", result["checks"])
    print("RESULT:", "PASS" if result["passed"] else "FAIL")
    print("\nANSWER:\n", result["answer"])


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    results: list[dict[str, Any]] = []

    for test_case in dataset:
        try:
            result = run_case(test_case)
            results.append(result)
            print_case_result(result)
        except Exception as error:
            failed = {
                "id": test_case.get("id"),
                "query": test_case.get("query"),
                "passed": False,
                "error": repr(error),
            }
            results.append(failed)
            print("\nTEST ERROR:", failed["id"], repr(error))

    successful = sum(bool(item.get("passed")) for item in results)
    total = len(results)
    completed = [item for item in results if "checks" in item]

    print("\n" + "=" * 78)
    print("END-TO-END EVALUATION SUMMARY")
    print("=" * 78)
    print(f"Total cases: {total}")
    print(f"Completed: {len(completed)}")
    print(f"Passed: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Pass rate: {successful / total * 100:.2f}%" if total else "Pass rate: N/A")

    if completed:
        for check_name in completed[0]["checks"]:
            passing = sum(bool(item["checks"][check_name]) for item in completed)
            print(f"{check_name}: {passing}/{len(completed)} ({passing / len(completed) * 100:.1f}%)")

        labelled = [
            item["retrieval_metrics"]
            for item in completed
            if item.get("retrieval_metrics") is not None
        ]
        if labelled:
            print("\nLabelled retrieval cases:", len(labelled))
            print("Mean Precision@K:", round(statistics.mean(x["precision_at_k"] for x in labelled), 3))
            print("Mean Recall@K:", round(statistics.mean(x["recall_at_k"] for x in labelled), 3))
            print("MRR:", round(statistics.mean(x["reciprocal_rank"] for x in labelled), 3))
        else:
            print("\nNo labelled relevant pages: retrieval precision/recall/MRR not calculated.")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
