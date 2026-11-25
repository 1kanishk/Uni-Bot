
import csv
import time
from pathlib import Path
from statistics import mean

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from graph.nodes import classify_intent_node


INTENTS = [
    "university",
    "academic_help",
    "out_of_scope",
    "unclear"
]


TEST_DATA = [
    ("What is the annual fee for BCA Core?", "university"),
    ("What are the admission eligibility criteria?", "university"),
    ("What is the minimum attendance requirement?", "university"),
    ("When is the examination form deadline?", "university"),
    ("Show me the BCA third semester syllabus.", "university"),
    ("Which departments offer artificial intelligence courses?", "university"),
    ("What scholarships are available for students?", "university"),
    ("What is the annual hostel fee?", "university"),
    ("Does the placement cell provide internship assistance?", "university"),
    ("Where can I find my semester timetable?", "university"),
    ("What is the procedure for obtaining a migration certificate?", "university"),
    ("How much is the revaluation fee?", "university"),
    ("When does the new academic session begin?", "university"),
    ("What are the university library timings?", "university"),
    ("What are the course withdrawal rules?", "university"),

    ("Explain binary search with an example.", "academic_help"),
    ("What is recursion in Python?", "academic_help"),
    ("Explain Python decorators.", "academic_help"),
    ("What is abstraction in object-oriented programming?", "academic_help"),
    ("Explain database normalization and 3NF.", "academic_help"),
    ("What is the difference between SQL joins?", "academic_help"),
    ("Explain gradient descent in machine learning.", "academic_help"),
    ("How does backpropagation work in neural networks?", "academic_help"),
    ("What is the difference between a stack and a queue?", "academic_help"),
    ("Explain React useEffect with an example.", "academic_help"),
    ("What are Python list comprehensions?", "academic_help"),
    ("Explain the TCP/IP protocol.", "academic_help"),
    ("What is Bayes theorem?", "academic_help"),
    ("Explain time complexity in DSA.", "academic_help"),
    ("What is a convolutional neural network?", "academic_help"),

    ("Plan a five-day trip to Goa.", "out_of_scope"),
    ("Recommend some good shoes for running.", "out_of_scope"),
    ("Who won yesterday's cricket match?", "out_of_scope"),
    ("Give me a recipe for chocolate cake.", "out_of_scope"),
    ("Write a romantic poem.", "out_of_scope"),
    ("What is the weather in Paris?", "out_of_scope"),
    ("Recommend some good movies.", "out_of_scope"),
    ("Suggest a laptop under 60000 rupees.", "out_of_scope"),
    ("Find good restaurants in Jaipur.", "out_of_scope"),
    ("Explain the latest stock market movements.", "out_of_scope"),

    ("What about that?", "unclear"),
    ("Can you explain it?", "unclear"),
    ("What is the process?", "unclear"),
    ("How much does it cost?", "unclear"),
    ("When is it?", "unclear"),
    ("Which one is better?", "unclear"),
    ("Tell me more.", "unclear"),
    ("Can I apply?", "unclear"),
    ("What happens next?", "unclear"),
    ("Is it compulsory?", "unclear")
]


results = []
latencies = []
confidences = []

for index, (query, expected) in enumerate(TEST_DATA, start=1):

    start_time = time.perf_counter()

    prediction = classify_intent_node({
        "query": query
    })

    latency = (time.perf_counter() - start_time) * 1000

    predicted = prediction["intent"]
    confidence = prediction["intent_confidence"]

    correct = predicted == expected

    results.append({
        "query": query,
        "expected": expected,
        "predicted": predicted,
        "confidence": confidence,
        "correct": correct,
        "latency_ms": round(latency, 2)
    })

    latencies.append(latency)
    confidences.append(confidence)

    print(
        f"{index:02d}/50 | "
        f"Expected: {expected:<15} | "
        f"Predicted: {predicted:<15} | "
        f"Confidence: {confidence:.4f} | "
        f"{'CORRECT' if correct else 'WRONG'}"
    )


y_true = [result["expected"] for result in results]
y_pred = [result["predicted"] for result in results]

print("\n========== EVALUATION RESULTS ==========")

print(f"Total questions: {len(TEST_DATA)}")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
print(f"Average confidence: {mean(confidences):.4f}")
print(f"Average latency: {mean(latencies):.2f} ms")

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        labels=INTENTS,
        zero_division=0,
        digits=4
    )
)

print("\nConfusion Matrix:\n")

print(confusion_matrix(
    y_true,
    y_pred,
    labels=INTENTS
))

output_path = Path("evaluation/intent_results.csv")

with output_path.open("w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to: {output_path}")
