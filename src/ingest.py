from pathlib import Path

from src.config import DATA_DIR
from src.loader import load_documents, split_documents
from src.vector_store import create_vector_store


def main():
    data_dir = Path(DATA_DIR)

    if not data_dir.is_dir():
        raise SystemExit(f"Document directory not found: {data_dir}")

    source_files = sorted(
        file for file in data_dir.iterdir()
        if file.is_file() and file.suffix.lower() in {".pdf", ".docx"}
    )

    if not source_files:
        raise SystemExit("No PDF or DOCX documents found. Existing index was not changed.")

    print(f"\nFound {len(source_files)} source files.")

    documents = load_documents(str(data_dir))

    if not documents:
        raise SystemExit(
            "No content could be extracted. Existing index was not changed."
        )

    print(f"Extracted {len(documents)} document sections.")

    chunks = split_documents(documents)

    if not chunks:
        raise SystemExit(
            "No chunks were generated. Existing index was not changed."
        )

    print(f"Generated {len(chunks)} chunks.")
    print("\nBuilding and validating the new vector store...")

    create_vector_store(chunks)

    print("\nKnowledge base rebuilt successfully.")
    print(f"Indexed source files: {len(source_files)}")
    print(f"Extracted sections: {len(documents)}")
    print(f"Stored chunks: {len(chunks)}")
    print(f"Vector store: {Path('vectorstore').resolve()}")

    print("\nRestart FastAPI to load the updated knowledge base.")


if __name__ == "__main__":
    main()