from src.vector_store import (
    load_vector_store,
    load_documents_from_vectorstore
)
from src.retriever import AdvancedRetriever

vector_store = load_vector_store()
documents = load_documents_from_vectorstore()
retriever = AdvancedRetriever(documents=documents, vectorstore=vector_store)

question = "When was JECRC University established?"

retrieved_docs = retriever.retrieve(question, top_k=5)

print(f"\nQuestion: {question}")
print(f"Retrieved documents: {len(retrieved_docs)}")

for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n{'=' * 80}")
    print(f"CONTEXT {i}")
    print(f"{'=' * 80}")
    print(doc.page_content)
    print("Metadata:", doc.metadata)