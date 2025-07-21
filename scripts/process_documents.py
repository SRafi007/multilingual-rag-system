# scripts/process_documents.py
from app.data.pdf_processor import PDFProcessor
from pathlib import Path


def main():
    pdf_path = "data/raw/HSC26-Bangla1st-Paper.pdf"
    output_path = "data/processed/extracted_text.txt"

    processor = PDFProcessor(pdf_path)
    extracted_text = processor.extract_text(join_pages=True)

    # Save the cleaned extracted text
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print(f"[✅] Extracted text saved to: {output_path}")
    print(f"[📄] Preview:\n{extracted_text[:1000]}...")


if __name__ == "__main__":
    main()
