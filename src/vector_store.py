import pickle
import shutil
from pathlib import Path
from uuid import uuid4

from langchain_community.vectorstores import FAISS

from src.embeddings import get_embeddings
from src.config import VECTOR_DB_DIR


def create_vector_store(documents):
    documents = list(documents)

    if not documents:
        raise ValueError("Cannot build vector store: no document chunks provided.")

    target = Path(VECTOR_DB_DIR)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not target.is_dir():
        raise NotADirectoryError(f"Vector store path is not a directory: {target}")

    identifier = uuid4().hex
    staging = target.parent / f".{target.name}.staging-{identifier}"
    backup = target.parent / f".{target.name}.backup-{identifier}"

    try:
        embeddings = get_embeddings()

        vectorstore = FAISS.from_documents(
            documents,
            embeddings
        )

        staging.mkdir()
        vectorstore.save_local(str(staging))

        with open(staging / "chunks.pkl", "wb") as f:
            pickle.dump(documents, f, protocol=pickle.HIGHEST_PROTOCOL)

        required_files = [
            "index.faiss",
            "index.pkl",
            "chunks.pkl"
        ]

        missing_files = [
            name for name in required_files
            if not (staging / name).is_file()
        ]

        if missing_files:
            raise RuntimeError(
                f"Vector store build incomplete. Missing: {missing_files}"
            )

        FAISS.load_local(
            str(staging),
            embeddings,
            allow_dangerous_deserialization=True
        )

        with open(staging / "chunks.pkl", "rb") as f:
            saved_documents = pickle.load(f)

        if len(saved_documents) != len(documents):
            raise RuntimeError("Stored document count does not match input count.")

        if target.exists():
            target.rename(backup)

        try:
            staging.rename(target)
        except Exception:
            if backup.exists() and not target.exists():
                backup.rename(target)
            raise

        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)

        return vectorstore

    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

        if backup.exists() and target.exists():
            shutil.rmtree(backup, ignore_errors=True)


def load_vector_store():
    target = Path(VECTOR_DB_DIR)

    required_files = [
        "index.faiss",
        "index.pkl",
        "chunks.pkl"
    ]

    if not target.is_dir():
        return None

    if not all((target / name).is_file() for name in required_files):
        return None

    embeddings = get_embeddings()

    return FAISS.load_local(
        str(target),
        embeddings,
        allow_dangerous_deserialization=True
    )


def load_documents_from_vectorstore():
    path = Path(VECTOR_DB_DIR) / "chunks.pkl"

    if not path.is_file():
        return None

    with open(path, "rb") as f:
        return pickle.load(f)