# app/data/pdf_ocr_processor.py

from pdf2image import convert_from_path
import pytesseract
from pathlib import Path
from typing import List


class BengaliOCRProcessor:
    def __init__(self, pdf_path: Path, lang: str = "ben"):
        self.pdf_path = Path(pdf_path)
        self.lang = lang

    def extract_text(self) -> str:
        images = convert_from_path(self.pdf_path)
        all_text = []
        for idx, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang=self.lang)
            all_text.append(text)
            print(f"[OCR] Page {idx + 1} extracted")
        return "\n".join(all_text)
