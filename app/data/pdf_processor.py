# app/data/pdf_processor.py
import pdfplumber
from pathlib import Path
from typing import List, Union
import re


class PDFProcessor:
    def __init__(self, pdf_path: Union[str, Path]):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

    def extract_text(self, join_pages: bool = True) -> Union[str, List[str]]:
        """
        Extracts text from all pages in the PDF.
        Args:
            join_pages (bool): If True, return a single string; otherwise, return list of page-wise strings.
        Returns:
            Union[str, List[str]]: Extracted text.
        """
        extracted_pages = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    cleaned = self._clean_text(text)
                    extracted_pages.append(cleaned)
                else:
                    print(f"[WARNING] Page {i+1} has no extractable text.")

        return "\n".join(extracted_pages) if join_pages else extracted_pages

    def _clean_text(self, text: str) -> str:
        """
        Perform basic cleaning for Bengali Unicode and noise.
        Args:
            text (str): Raw extracted text.
        Returns:
            str: Cleaned text.
        """
        text = text.replace("\xa0", " ")  # Non-breaking spaces
        text = re.sub(r"\s+", " ", text)  # Extra whitespace
        text = re.sub(
            r'[^\u0980-\u09FF\s.,;:!?()\-\[\]“”"\'\n]', "", text
        )  # Keep Bengali + basic punctuation
        return text.strip()


# Example usage (you can delete this in production):
if __name__ == "__main__":
    processor = PDFProcessor("data/raw/HSC26-Bangla1st-Paper.pdf")
    full_text = processor.extract_text()
    print(full_text[:1000])  # Preview the first 1000 characters
