import json

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric
)
from deepeval.evaluate import AsyncConfig
from evaluation.judge_llm import LocalQwenJudge


def load_results():

    with open(
        "evaluation/Golden_Dataset_v3.json",
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def main():

    results = load_results()

    test_cases = []

    for item in results:

        test_case = LLMTestCase(
            input=item["question"],
            actual_output=item["actual_answer"],
            expected_output=item["expected_answer"],
            retrieval_context=item["retrieval_context"]
        )

        test_cases.append(test_case)

    print(
        f"Loaded {len(test_cases)} evaluation cases."
    )

    judge = LocalQwenJudge()

    metrics = [
        ContextualRecallMetric(
            threshold=0.5,
            model=judge,
            include_reason=True,
            async_mode=False
        ),
        ContextualPrecisionMetric(
            threshold=0.5,
            model=judge,
            include_reason=True,
            async_mode=False
        )
    ]

    print("\nStarting DeepEval...\n")

    evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(
        run_async=False
    )

)


if __name__ == "__main__":
    main()