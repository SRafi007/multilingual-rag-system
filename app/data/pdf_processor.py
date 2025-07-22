# app/data/pdf_ocr_processor.py

from pathlib import Path
from pdf2image import convert_from_path
import pytesseract
import re
import os

# ✅ Set the Tesseract binary path manually
pytesseract.pytesseract.tesseract_cmd = (
    r"D:\softwares\code_related\Tesseract-OCR\tesseract.exe"
)

# ✅ Set the path to tessdata (for Bangla)
os.environ["TESSDATA_PREFIX"] = r"D:\softwares\code_related\Tesseract-OCR\tessdata"


def extract_bangla_text_from_pdf(pdf_path: str, output_txt_path: str = None):
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"🔍 Converting PDF pages to images...")
    images = convert_from_path(pdf_path, dpi=300)

    print(f"🔠 Running OCR on {len(images)} pages...")
    extracted_text = ""
    for i, img in enumerate(images, 1):
        print(f"📝 OCR processing page {i}...")
        try:
            text = pytesseract.image_to_string(img, lang="ben")
        except pytesseract.TesseractNotFoundError:
            print("❌ Tesseract not found. Please check the path.")
            return
        except FileNotFoundError as e:
            print(f"❌ OCR failed: {e}")
            return

        cleaned = clean_text(text)
        extracted_text += f"\n\n--- Page {i} ---\n{cleaned}"

    if output_txt_path:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        print(f"\n✅ Extracted text saved to: {output_txt_path}")

    return extracted_text


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\u0980-\u09FFa-zA-Z0-9\s.,;:!?()\-\[\]“”\"\'\n]", "", text)
    return text.strip()


"""
# Run the processor
if __name__ == "__main__":
    extract_bangla_text_from_pdf(
        "data/raw/HSC26-Bangla1st-Paper.pdf", "data/processed/bangla_extracted_text.txt"
    )
"""
