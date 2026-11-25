from src.config import DATA_DIR
from src.loader import load_documents, split_documents
from src.vector_store import create_vector_store

documents = load_documents(DATA_DIR)
documents = split_documents(documents)

create_vector_store(documents)

print(f"Rebuilt vector store with {len(documents)} documents.")