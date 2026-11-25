import os
import pdfplumber
import pytesseract
from PIL import ImageOps
from src.config import DATA_DIR

for filename in os.listdir(DATA_DIR):
    if not filename.lower().endswith(".pdf"):
        continue

    path = os.path.join(DATA_DIR, filename)

    if "brochure" not in filename.lower() and "information" not in filename.lower():
        continue

    with pdfplumber.open(path) as pdf:
        page = pdf.pages[4]

        print(f"\nPDF: {filename}")
        print("\n--- NATIVE TEXT ---")
        print(page.extract_text() or "[No native text found]")

        image = page.to_image(resolution=400).original
        image = ImageOps.autocontrast(image.convert("L"))

        print("\n--- OCR TEXT (PSM 11) ---")
        print(
            pytesseract.image_to_string(
                image,
                config="--oem 3 --psm 11"
            )
        )