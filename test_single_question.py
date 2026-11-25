import sys
import time

from src.vector_store import load_vector_store, load_documents_from_vectorstore
from src.retriever import AdvancedRetriever
from src.llm import get_llm
from src.prompt import get_prompt


def main():
    question = " ".join(sys.argv[1:]).strip() or "What is the duration of the Integrated Law program and what are its eligibility requirements?"

    vectorstore = load_vector_store()
    documents = load_documents_from_vectorstore()

    if vectorstore is None or not documents:
        raise RuntimeError("Vector store or documents not found. Rebuild the vector store first.")

    retriever = AdvancedRetriever(documents, vectorstore)
    llm = get_llm()
    prompt = get_prompt()

    start = time.perf_counter()

    retrieved_docs = retriever.retrieve(question, top_k=5)

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    print("\nQUESTION:", question)
    print("\n========== RETRIEVED CONTEXTS ==========")

    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n--- CONTEXT {i} ---")
        print(doc.page_content)
        print("Source:", doc.metadata.get("source"))
        print("Page:", doc.metadata.get("page"))

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(messages)

    print("\n========== GENERATED ANSWER ==========")
    print(response.content)
    print(f"\nRetrieval + generation time: {time.perf_counter() - start:.2f} seconds")


if __name__ == "__main__":
    main()