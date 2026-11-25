import os
import re
import pdfplumber
import pytesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
import unicodedata

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def clean(value):
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def normalize_text(text):
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ")

    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[%₹])(?=[A-Za-z])", " ", text)

    text = re.sub(r"(?<=\+)(?=[A-Za-z])", " ", text)
    text = re.sub(
        r"(?<=LPA)(?=[A-Za-z])",
        " ",
        text,
        flags=re.IGNORECASE
    )

    return " ".join(text.split()).strip()


def normalize_header(value, index):
    value = clean(value)

    if not value:
        return f"Column {index + 1}"

    key = re.sub(r"[^a-z0-9]", "", value.lower())

    if "eligibility" in key:
        return "Eligibility"
    if "specialization" in key or "branch" in key:
        return "Specialization"
    if "tuition" in key:
        return "Tuition Fees"
    if "other" in key and "fee" in key:
        return "Other Fees"
    if "annual" in key and "fee" in key:
        return "Annual Fees"
    if "subsequent" in key or "subseq" in key:
        return "Subsequent Installment"
    if "first" in key or "istinst" in key:
        return "First Installment"
    if "program" in key or "course" in key:
        return "Program"

    return value


def structure_statistics(text):
    if not text:
        return text

    if "Total Placements Offers" not in text and "Average Package" not in text:
        return text

    patterns = [
        ("Total Placement Offers", r"(\d[\d,]*\+?)\s*Total Placements Offers"),
        ("Number of Companies", r"(\d[\d,]*\+?)\s*Number of Companies"),
        (
            "Average Package - Management",
            r"(\d+(?:\.\d+)?)\s*LPA\s*Average Package\s*\(Management\)",
        ),
        (
            "Average Package - Technical",
            r"(\d+(?:\.\d+)?)\s*LPA\s*Average Package\s*\(Technical\)",
        ),
        (
            "Average Package - ME/CE/ECE",
            r"(\d+(?:\.\d+)?)\s*LPA\s*Average Package\s*\(ME/CE/ECE\)",
        ),
        (
            "Average Package - Skill Based Programs",
            r"(\d+(?:\.\d+)?)\s*LPA\s*Average Package\s*\(Skill Based Programs\)",
        ),
        (
            "Highest Package - Management",
            r"(\d+(?:\.\d+)?)\s*LPA\s*Highest Package\s*\(Management\)",
        ),
        (
            "Highest Package - Technical",
            r"(\d+(?:\.\d+)?)\s*LPA\s*Highest Package\s*\(Technical\)",
        ),
        (
            "Highest Package - ME/CE/ECE",
            r"(\d+(?:\.\d+)?)\s*LPA\s*Highest Package\s*\(ME/CE/ECE\)",
        ),
        (
            "Highest Package - Skill Based Programs",
            r"(\d+(?:\.\d+)?)\s*LPA\s*Highest Package\s*\(Skill Based Programs\)",
        ),
    ]

    structured = []

    for label, pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            value = match.group(1)

            if "Package" in label:
                value = f"{value} LPA"

            structured.append(f"- {label}: {value}")

    if not structured:
        return text

    return text + "\n\nStructured Statistics:\n" + "\n".join(structured)


def table_to_documents(table, filename, page_number, table_index):
    rows = []

    for row in table:
        row = [clean(cell) for cell in row]
        if any(row):
            rows.append(row)

    if len(rows) < 2:
        return []

    width = max(len(row) for row in rows)

    if width < 2:
        return []

    header_index = None

    for i, row in enumerate(rows[:4]):
        nonempty = [cell for cell in row if cell]

        if (
            len(nonempty) >= 2
            and max(map(len, nonempty)) <= 120
            and sum(map(len, nonempty)) <= 500
        ):
            header_index = i
            break

    if header_index is None:
        return []

    raw_headers = (rows[header_index] + [""] * width)[:width]

    headers = [normalize_header(value, i) for i, value in enumerate(raw_headers)]

    documents = []
    previous_program = ""

    for row_index, row in enumerate(rows[header_index + 1 :], start=header_index + 1):
        row = (row + [""] * width)[:width]
        nonempty = [cell for cell in row if cell]

        if len(nonempty) < 2:
            continue

        if max(map(len, nonempty)) > 700 or sum(map(len, nonempty)) > 1400:
            continue

        fields = []

        for i, value in enumerate(row):
            header = headers[i]

            if header == "Program":
                if value:
                    previous_program = value
                else:
                    value = previous_program

            if value:
                fields.append(f"{header}: {value}")

        if len(fields) < 2:
            continue

        documents.append(
            Document(
                page_content=" | ".join(fields),
                metadata={
                    "source": filename,
                    "page": page_number,
                    "content_type": "table_row",
                    "table_index": table_index,
                    "row_index": row_index,
                },
            )
        )

    return documents


def compact_text(text):
    return re.sub(r"[^a-z0-9₹]", "", text.lower())


def get_missing_ocr_lines(native_text, ocr_text):
    native_compact = compact_text(native_text)
    native_tokens = set(
        re.findall(r"₹|[a-z]+|\d+(?:[.,]\d+)*%?", native_text.lower())
    )

    missing = []
    seen = set()

    for raw_line in ocr_text.splitlines():
        line = normalize_text(raw_line)
        compact = compact_text(line)

        if len(compact) < 3 or compact in seen:
            continue

        if compact in native_compact:
            continue

        tokens = set(
            re.findall(r"₹|[a-z]+|\d+(?:[.,]\d+)*%?", line.lower())
        )

        has_number = any(re.search(r"\d", token) for token in tokens)

        if (
            len(tokens) >= 5
            and not has_number
            and len(tokens & native_tokens) / len(tokens) >= 0.9
        ):
            continue

        missing.append(line)
        seen.add(compact)

    return "\n".join(missing)


def load_pdf(path, filename):
    documents = []
    normal_docs = PyPDFLoader(path).load()

    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            native_text = ""

            if page_number <= len(normal_docs):
                native_text = normalize_text(
                    normal_docs[page_number - 1].page_content
                )

            page_text = structure_statistics(native_text)

            if page_text:
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": filename,
                            "page": page_number,
                            "content_type": "text"
                        }
                    )
                )

            try:
                image = page.to_image(resolution=300).original
                ocr_text = pytesseract.image_to_string(
                    image,
                    config="--psm 3"
                )

                missing_text = get_missing_ocr_lines(
                    native_text,
                    ocr_text
                )

                if missing_text:
                    documents.append(
                        Document(
                            page_content=missing_text,
                            metadata={
                                "source": filename,
                                "page": page_number,
                                "content_type": "ocr"
                            }
                        )
                    )

            except Exception as e:
                print(f"OCR failed on page {page_number}: {e}")

            for table_index, table in enumerate(page.extract_tables() or []):
                documents.extend(
                    table_to_documents(
                        table,
                        filename,
                        page_number,
                        table_index
                    )
                )

    return documents


def load_documents(data_dir):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Document directory not found: {data_dir}")

    documents = []

    files = sorted(
        os.scandir(data_dir),
        key=lambda entry: entry.name.casefold()
    )

    for file in files:
        if not file.is_file():
            continue

        extension = os.path.splitext(file.name)[1].lower()

        if extension == ".pdf":
            documents.extend(load_pdf(file.path, file.name))

        elif extension == ".docx":
            docs = Docx2txtLoader(file.path).load()

            for doc in docs:
                doc.page_content = normalize_text(doc.page_content)
                doc.metadata["source"] = file.name
                doc.metadata["content_type"] = "text"

            documents.extend(docs)

    return documents

def split_documents(documents):

    text_documents = [
        doc for doc in documents if doc.metadata.get("content_type") != "table_row"
    ]

    table_documents = [
        doc for doc in documents if doc.metadata.get("content_type") == "table_row"
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    split_text = splitter.split_documents(text_documents)

    return split_text + table_documents


if __name__ == "__main__":

    from src.loader import load_documents
    from src.config import DATA_DIR

    docs = load_documents(DATA_DIR)

    for doc in docs:
        if doc.metadata.get("page") == 5:
            if doc.metadata.get("content_type") in ("ocr", "ocr_fact"):
                print(doc.metadata["content_type"], doc.page_content)
