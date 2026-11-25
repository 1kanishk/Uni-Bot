from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = str(PROJECT_ROOT / "data" / "documents")
VECTOR_DB_DIR = str(PROJECT_ROOT / "vectorstore")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200

TOP_K = 5