
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

from graph.nodes import classify_intent_node, ALLOWED_INTENTS


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "intent_dataset.json"
CSV_PATH = BASE_DIR / "intent_test_results.csv"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_metrics(results):
    valid_results = [
        row for row in results
        if row["predicted_intent"] != "ERROR"
    ]

    if not valid_results:
        print("\nNo successful classifications.")
        return

    correct = sum(row["correct"] for row in valid_results)
    total = len(valid_results)

    print("\n" + "=" * 65)
    print("OVERALL RESULTS")
    print("=" * 65)

    print(f"Total questions       : {len(results)}")
    print(f"Successful tests      : {total}")
    print(f"Failed requests       : {len(results) - total}")
    print(f"Correct predictions   : {correct}")
    print(f"Incorrect predictions : {total - correct}")
    print(f"Accuracy              : {correct / total * 100:.2f}%")

    classes = sorted({
        row["expected_intent"] for row in valid_results
    } | {
        row["predicted_intent"] for row in valid_results
    })

    print("\n" + "=" * 65)
    print("PER-INTENT METRICS")
    print("=" * 65)

    for intent in classes:
        tp = sum(
            row["expected_intent"] == intent
            and row["predicted_intent"] == intent
            for row in valid_results
        )

        fp = sum(
            row["expected_intent"] != intent
            and row["predicted_intent"] == intent
            for row in valid_results
        )

        fn = sum(
            row["expected_intent"] == intent
            and row["predicted_intent"] != intent
            for row in valid_results
        )

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        print(f"\n{intent}")
        print(f"  Precision : {precision:.2%}")
        print(f"  Recall    : {recall:.2%}")
        print(f"  F1-score  : {f1:.2%}")


def print_confusion_matrix(results):
    valid_results = [
        row for row in results
        if row["predicted_intent"] != "ERROR"
    ]

    classes = sorted({
        row["expected_intent"] for row in valid_results
    } | {
        row["predicted_intent"] for row in valid_results
    })

    matrix = defaultdict(Counter)

    for row in valid_results:
        matrix[row["expected_intent"]][row["predicted_intent"]] += 1

    print("\n" + "=" * 65)
    print("CONFUSION MATRIX")
    print("=" * 65)

    print(f"{'EXPECTED / PREDICTED':<20}", end="")
    for intent in classes:
        print(f"{intent[:12]:>14}", end="")
    print()

    for expected in classes:
        print(f"{expected:<20}", end="")

        for predicted in classes:
            print(f"{matrix[expected][predicted]:>14}", end="")

        print()


def save_results(results):
    if not results:
        return

    fields = [
        "id",
        "query",
        "expected_intent",
        "predicted_intent",
        "confidence",
        "correct",
        "error"
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDetailed results saved to: {CSV_PATH}")


def main():
    dataset = load_dataset()

    print("=" * 65)
    print("UNIBOT INTENT CLASSIFICATION EVALUATION")
    print("=" * 65)

    expected_categories = {
        item["expected_intent"] for item in dataset
    }

    unsupported = expected_categories - ALLOWED_INTENTS

    if unsupported:
        print("\nWARNING: Dataset contains unsupported intents:")
        print(unsupported)
        print("These cannot be predicted by the current classifier.")
        print("Update ALLOWED_INTENTS and the Laya instructions first.")

    results = []

    for index, item in enumerate(dataset, start=1):
        query = item["query"]
        expected = item["expected_intent"]

        print(f"\n[{index}/{len(dataset)}] {query}", flush=True)

        try:
            prediction = classify_intent_node({
                "query": query,
                "messages": []
            })

            predicted = prediction["intent"]
            confidence = prediction.get("intent_confidence", 0)

            correct = predicted == expected

            status = "PASS" if correct else "FAIL"

            print(
                f"Expected: {expected} | "
                f"Predicted: {predicted} | "
                f"Confidence: {confidence:.3f} | "
                f"{status}",
                flush=True
            )

            results.append({
                "id": item["id"],
                "query": query,
                "expected_intent": expected,
                "predicted_intent": predicted,
                "confidence": confidence,
                "correct": correct,
                "error": ""
            })

        except Exception as error:
            print(f"ERROR: {error}", flush=True)

            results.append({
                "id": item["id"],
                "query": query,
                "expected_intent": expected,
                "predicted_intent": "ERROR",
                "confidence": "",
                "correct": False,
                "error": str(error)
            })

    calculate_metrics(results)
    print_confusion_matrix(results)
    save_results(results)

    print("\n" + "=" * 65)
    print("MISCLASSIFIED QUESTIONS")
    print("=" * 65)

    for row in results:
        if not row["correct"]:
            print(f"\nID: {row['id']}")
            print(f"Query: {row['query']}")
            print(f"Expected: {row['expected_intent']}")
            print(f"Predicted: {row['predicted_intent']}")
            if row["error"]:
                print(f"Error: {row['error']}")


if __name__ == "__main__":
    main()
