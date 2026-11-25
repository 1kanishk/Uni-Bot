import json

from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.evaluate import AsyncConfig
from deepeval.metrics.g_eval import Rubric
from evaluation.judge_llm import LocalQwenJudge
from deepeval.test_case import LLMTestCase, SingleTurnParams


DATASET = "evaluation/Generator_Data_v4.json"


def load_dataset():

    with open(DATASET, encoding="utf-8") as f:
        return json.load(f)


def main():

    data = load_dataset()

    test_cases = []

    for item in data:

        test_cases.append(
            LLMTestCase(
                input=item["question"],
                actual_output=item["actual_answer"],
                expected_output=item["expected_answer"]
            )
        )

    judge = LocalQwenJudge()

    correctness = GEval(
        name="Answer Correctness",

        evaluation_steps=[
            "Understand exactly what factual answer the question is asking for.",
            "Compare the factual meaning of the actual output with the expected output.",
            "Judge semantic equivalence, not exact wording.",
            "Treat equivalent number formats as the same fact, such as 34000+ and 34,000+, and 2230+ and 2,230+.",
            "Treat equivalent expressions such as 'more than 34,000' and '34,000+' as equivalent when they convey the same meaning.",
            "Do not penalize differences in capitalization, punctuation, grammar, sentence structure, or wording.",
            "Do not penalize concise answers when the factual answer is correct.",
            "Do not penalize additional information when that information is correct.",
            "Penalize only genuine factual errors, contradictions, wrong values, or wrong entities.",
            "Do not require the actual output to reproduce the wording of the expected output."
        ],

        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome=(
                    "The actual answer is factually correct. "
                    "It conveys the same answer as the expected output. "
                    "Minor wording or formatting differences must not reduce the score."
                )
            ),
            Rubric(
                score_range=(7, 8),
                expected_outcome=(
                    "The answer is essentially correct but contains a minor factual "
                    "imprecision that does not change the main answer."
                )
            ),
            Rubric(
                score_range=(5, 6),
                expected_outcome=(
                    "The answer is partially correct but contains a meaningful "
                    "factual problem or ambiguity."
                )
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome=(
                    "The answer contains a clear factual error, contradiction, "
                    "wrong value, wrong entity, or does not answer the question."
                )
            )
        ],

        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT
        ],

        threshold=0.5,
        model=judge,
        strict_mode=False,
        async_mode=False
    )

    completeness = GEval(
        name="Answer Completeness",

        evaluation_steps=[
            "Understand exactly what information the question requires.",
            "Use the expected output as the factual reference for determining the required answer.",
            "Identify only the essential information needed to answer the question.",
            "Check whether the actual output contains all essential information required by the question.",
            "Do not require the actual output to reproduce unnecessary wording from the expected output.",
            "Do not require phrases such as 'the brochure states' unless the question specifically asks for that.",
            "Do not require source descriptions, explanations, or background details that are not necessary to answer the question.",
            "Do not penalize concise answers when they contain all essential requested information.",
            "Do not penalize correct additional information.",
            "Do not judge whether the information is factually correct here; evaluate coverage of the required information only."
        ],

        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome=(
                    "The actual answer contains essentially all information "
                    "necessary to answer the question."
                )
            ),
            Rubric(
                score_range=(7, 8),
                expected_outcome=(
                    "The actual answer contains the main required information "
                    "but omits a minor relevant detail."
                )
            ),
            Rubric(
                score_range=(5, 6),
                expected_outcome=(
                    "The actual answer contains some required information but "
                    "misses an important part of the answer."
                )
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome=(
                    "The actual answer omits most of the essential information "
                    "needed to answer the question."
                )
            )
        ],

        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT
        ],

        threshold=0.5,
        model=judge,
        strict_mode=False,
        async_mode=False
    )

    print(f"Loaded {len(test_cases)} evaluation cases.")
    print("\nStarting G-EVAL...")

    evaluate(
        test_cases=test_cases,
        metrics=[
            correctness,
            completeness
        ],
        async_config=AsyncConfig(run_async=False)
    )


if __name__ == "__main__":
    main()