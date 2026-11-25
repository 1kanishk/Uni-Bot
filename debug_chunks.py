from src.loader import load_documents, split_documents
from src.config import DATA_DIR

docs = load_documents(DATA_DIR)
chunks = split_documents(docs)

for i, doc in enumerate(chunks):
    text = doc.page_content.lower()

    if "stipend" in text or ("5k" in text and "60k" in text):
        print(f"\nChunk: {i}")
        print(f"Metadata: {doc.metadata}")
        print(doc.page_content)
        