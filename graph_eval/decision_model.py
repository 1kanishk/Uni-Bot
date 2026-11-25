from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.nodes import ALLOWED_INTENTS, classify_intent_node


DEFAULT_DATASET = PROJECT_ROOT / "graph_eval" / "e2e_dataset.json"
DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "graph_eval" / "results.json"
DEFAULT_CSV_OUTPUT = PROJECT_ROOT / "graph_eval" / "results.csv"


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Expected graph_eval/intent_dataset.json. "
            "Run this script from the Uni-Bot project structure."
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("The E2E dataset must be a JSON list.")

    valid_cases = []
    for index, case in enumerate(data, start=1):
        if not isinstance(case, dict):
            print(f"[SKIP] Dataset item {index} is not an object.")
            continue

        if not case.get("query") or not case.get("expected_intent"):
            print(f"[SKIP] Dataset item {index} is missing query/expected_intent.")
            continue

        expected = str(case["expected_intent"]).strip()
        if expected not in ALLOWED_INTENTS:
            raise ValueError(
                f"Invalid expected_intent '{expected}' in case "
                f"{case.get('id', index)}. Allowed: {sorted(ALLOWED_INTENTS)}"
            )

        valid_cases.append(case)

    if not valid_cases:
        raise ValueError("No valid test cases found in the dataset.")

    return valid_cases


def calculate_metrics(
    results: list[dict[str, Any]],
    labels: list[str],
) -> dict[str, Any]:
    completed = [r for r in results if r["error"] is None]
    correct = [r for r in completed if r["passed"]]

    confusion = {
        expected: {predicted: 0 for predicted in labels}
        for expected in labels
    }

    for result in completed:
        expected = result["expected_intent"]
        predicted = result["predicted_intent"]
        if expected in confusion and predicted in confusion[expected]:
            confusion[expected][predicted] += 1

    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        per_class[label] = {
            "support": sum(confusion[label].values()),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    total = len(results)
    successful_calls = len(completed)
    correct_count = len(correct)

    return {
        "total_cases": total,
        "successful_calls": successful_calls,
        "failed_calls": total - successful_calls,
        "correct": correct_count,
        "incorrect": successful_calls - correct_count,
        "accuracy": round(correct_count / successful_calls, 4) if successful_calls else 0.0,
        "dataset_pass_rate": round(correct_count / total, 4) if total else 0.0,
        "macro_f1": round(
            sum(item["f1"] for item in per_class.values()) / len(labels), 4
        ),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def run_test(
    dataset: list[dict[str, Any]],
    pause_seconds: float = 0.0,
    selected_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    if selected_ids:
        dataset = [
            case for case in dataset
            if str(case.get("id", "")) in selected_ids
        ]

    if not dataset:
        raise ValueError("No test cases selected. Check --ids.")

    total = len(dataset)

    for index, case in enumerate(dataset, start=1):
        case_id = str(case.get("id", f"CASE-{index:03d}"))
        query = str(case["query"]).strip()
        expected = str(case["expected_intent"]).strip()

        started = time.perf_counter()

        try:
            output = classify_intent_node({"query": query})
            elapsed = time.perf_counter() - started

            predicted = str(output.get("intent", "missing")).strip()
            confidence = output.get("intent_confidence")
            probabilities = output.get("intent_probabilities")

            passed = predicted == expected

            result = {
                "id": case_id,
                "query": query,
                "expected_intent": expected,
                "predicted_intent": predicted,
                "intent_confidence": confidence,
                "intent_probabilities": probabilities,
                "latency_seconds": round(elapsed, 4),
                "passed": passed,
                "error": None,
            }

            status = "PASS" if passed else "FAIL"
            confidence_text = (
                f"{float(confidence):.4f}"
                if confidence is not None
                else "n/a"
            )
            print(
                f"[{index:02d}/{total:02d}] {status} {case_id} | "
                f"expected={expected:<14} predicted={predicted:<14} "
                f"confidence={confidence_text} | {elapsed:.2f}s"
            )

        except Exception as exc:
            elapsed = time.perf_counter() - started
            result = {
                "id": case_id,
                "query": query,
                "expected_intent": expected,
                "predicted_intent": None,
                "intent_confidence": None,
                "intent_probabilities": None,
                "latency_seconds": round(elapsed, 4),
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

            print(
                f"[{index:02d}/{total:02d}] ERROR {case_id} | "
                f"{type(exc).__name__}: {exc}"
            )

        results.append(result)

        if pause_seconds > 0 and index < total:
            time.sleep(pause_seconds)

    return results


def save_results(
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
    json_path: Path,
    csv_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_timestamp": datetime.now().astimezone().isoformat(),
        "test_type": "laya_only_classification",
        "classifier": "graph.nodes.classify_intent_node",
        "groq_inference_requests": 0,
        "metrics": metrics,
        "results": results,
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    fields = [
        "id",
        "query",
        "expected_intent",
        "predicted_intent",
        "intent_confidence",
        "latency_seconds",
        "passed",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in fields})


def print_summary(
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    print("\n" + "=" * 78)
    print("LAYA CLASSIFICATION SUMMARY")
    print("=" * 78)
    print(f"Total cases       : {metrics['total_cases']}")
    print(f"Successful calls  : {metrics['successful_calls']}")
    print(f"Failed calls      : {metrics['failed_calls']}")
    print(f"Correct           : {metrics['correct']}")
    print(f"Incorrect         : {metrics['incorrect']}")
    print(f"Accuracy          : {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1          : {metrics['macro_f1'] * 100:.2f}%")

    print("\nPER-CLASS METRICS")
    print("-" * 78)
    print(f"{'Intent':<18} {'Support':>8} {'Precision':>12} {'Recall':>10} {'F1':>10}")
    for label, item in metrics["per_class"].items():
        print(
            f"{label:<18} {item['support']:>8} "
            f"{item['precision'] * 100:>11.2f}% "
            f"{item['recall'] * 100:>9.2f}% "
            f"{item['f1'] * 100:>9.2f}%"
        )

    failures = [r for r in results if not r["passed"]]
    print("\nMISCLASSIFICATIONS / ERRORS")
    print("-" * 78)

    if not failures:
        print("No failures.")
    else:
        for result in failures:
            if result["error"]:
                print(f"{result['id']} | ERROR | {result['error']}")
            else:
                confidence = result.get("intent_confidence")
                confidence_text = (
                    f"{float(confidence):.4f}"
                    if confidence is not None
                    else "n/a"
                )
                print(
                    f"{result['id']} | {result['query']}\n"
                    f"  expected={result['expected_intent']} | "
                    f"predicted={result['predicted_intent']} | "
                    f"confidence={confidence_text}"
                )

    print("\nCONFUSION MATRIX")
    print("-" * 78)
    labels = list(metrics["per_class"].keys())
    print(f"{'Expected \\ Predicted':<20}" + "".join(f"{x:>16}" for x in labels))
    for expected in labels:
        row = metrics["confusion_matrix"][expected]
        print(f"{expected:<20}" + "".join(f"{row[predicted]:>16}" for predicted in labels))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run only Laya classification on the Uni-Bot E2E dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to evaluation/e2e_dataset.json",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help="Path for the detailed JSON results.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
        help="Path for the CSV results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally run only the first N dataset cases.",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Optional comma-separated case IDs, e.g. UNI-023,ACA-014,UNC-005.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="Optional delay between local Laya requests in seconds.",
    )

    args = parser.parse_args()

    if args.pause < 0:
        parser.error("--pause cannot be negative.")

    try:
        dataset = load_dataset(args.dataset)

        if args.limit is not None:
            if args.limit < 1:
                parser.error("--limit must be at least 1.")
            dataset = dataset[:args.limit]

        selected_ids = (
            {item.strip() for item in args.ids.split(",") if item.strip()}
            if args.ids
            else None
        )

        print(f"Dataset: {args.dataset}")
        results = run_test(
            dataset=dataset,
            pause_seconds=args.pause,
            selected_ids=selected_ids,
        )

        labels = ["university", "academic_help", "out_of_scope", "unclear"]
        metrics = calculate_metrics(results, labels)

        save_results(
            results=results,
            metrics=metrics,
            json_path=args.json_output,
            csv_path=args.csv_output,
        )
        print_summary(results, metrics)

        print("\nSaved files:")
        print(f"JSON: {args.json_output}")
        print(f"CSV : {args.csv_output}")

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nTest failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
