# scripts/pdf_process_documents.py

import sys
from pathlib import Path
from app.data.pdf_processor import extract_bangla_text_from_pdf

# Add the project root to sys.path if needed so imports work
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def main():
    input_pdf = "data/raw/HSC26-Bangla1st-Paper.pdf"
    output_txt = "data/processed/bangla_extracted_text.txt"

    print(f"Starting OCR extraction from: {input_pdf}")
    extracted_text = extract_bangla_text_from_pdf(input_pdf, output_txt)
    if extracted_text:
        print("OCR extraction completed successfully.")


if __name__ == "__main__":
    main()
