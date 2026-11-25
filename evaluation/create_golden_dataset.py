import json

from src.vector_store import (
    load_vector_store,
    load_documents_from_vectorstore
)
from src.retriever import AdvancedRetriever


def main():

    with open(
        "evaluation/dataset.json",
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)
        questions = data["questions"]

    vectorstore = load_vector_store()
    documents = load_documents_from_vectorstore()

    retriever = AdvancedRetriever(
        documents,
        vectorstore
    )

    for item in questions:

        docs = retriever.retrieve(
            item["question"],
            top_k=5
        )

        item["retrieval_context"] = [
            doc.page_content
            for doc in docs
        ]

    with open(
        "evaluation/Golden_1.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("Golden_1.json created successfully.")


if __name__ == "__main__":
    main()