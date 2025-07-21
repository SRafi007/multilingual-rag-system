from app.data.pdf_ocr_processor import BengaliOCRProcessor
from pathlib import Path


def main():
    pdf_path = Path("data/raw/HSC26-Bangla1st-Paper.pdf")
    output_path = Path("data/processed/ocr_extracted_text.txt")

    processor = BengaliOCRProcessor(pdf_path)
    extracted_text = processor.extract_text()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print(f"[✅] OCR text saved to {output_path}")
    print(f"[📄] Preview:\n{extracted_text[:1000]}")


if __name__ == "__main__":
    main()
