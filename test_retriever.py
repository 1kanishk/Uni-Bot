from src.retriever import AdvancedRetriever
from src.vector_store import (
    load_vector_store,
    load_documents_from_vectorstore,
)


vectorstore = load_vector_store()
documents = load_documents_from_vectorstore()

retriever = AdvancedRetriever(
    documents=documents,
    vectorstore=vectorstore,
)

results = retriever.retrieve(
    "What is the annual fee for BCA Core?"
)

for document in results:
    print(document.page_content)
    print(document.metadata)
    print("-" * 50)