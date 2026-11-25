
import json
import time

from src.vector_store import (
    load_vector_store,
    load_documents_from_vectorstore
)
from src.retriever import AdvancedRetriever
from src.llm import get_llm
from src.prompt import get_prompt


def load_dataset():
    with open(
        "evaluation/Golden_1.json",
        "r",
        encoding="utf-8"
    ) as f:
          data = json.load(f)

    return data["questions"]


def main():
    questions = load_dataset()

    vectorstore = load_vector_store()
    documents = load_documents_from_vectorstore()

    retriever = AdvancedRetriever(documents, vectorstore)

    llm = get_llm()
    prompt = get_prompt()

    output = []

    print(f"Loaded {len(questions)} evaluation cases.")
    print("\nGenerating fresh answers...\n")

    for index, item in enumerate(questions, start=1):
        question = item["question"]

        print(f"[{index}/{len(questions)}] {question}")

        start_time = time.perf_counter()

        try:
            retrieved_docs = retriever.retrieve(
                question,
                top_k=5
            )

            context = "\n\n".join(
                doc.page_content
                for doc in retrieved_docs
            )

            messages = prompt.format_messages(
                context=context,
                question=question
            )

            response = llm.invoke(messages)
            answer = response.content

            latency = time.perf_counter() - start_time

            result = {
                "id": item["id"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "is_answerable": item["is_answerable"],
                "question": question,
                "expected_answer": item["ground_truth_answer"],
                "relevant_pages": item.get("relevant_pages", []),
                "actual_answer": answer,
                "retrieval_context": [
                    doc.page_content
                    for doc in retrieved_docs
                ],
                "sources": [
                    {
                        "source": doc.metadata.get("source"),
                        "page": doc.metadata.get("page")
                    }
                    for doc in retrieved_docs
                ],
                "latency": latency
            }

        except Exception as e:
            latency = time.perf_counter() - start_time

            result = {
                "id": item["id"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "is_answerable": item["is_answerable"],
                "question": question,
                "expected_answer": item["ground_truth_answer"],
                "relevant_pages": item.get("relevant_pages", []),
                "actual_answer": "",
                "retrieval_context": [],
                "sources": [],
                "latency": latency,
                "error": str(e)
            }

            print(f"ERROR: {e}")

        output.append(result)

    with open(
        "evaluation/generator_1.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\nGenerator dataset created successfully.")
    print(f"Total cases: {len(output)}")
    print(f"Failed cases: {sum('error' in item for item in output)}")


if __name__ == "__main__":
    main()
