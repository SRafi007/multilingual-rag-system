"""
Example usage of the OCR Gemini Extractor pipeline
"""

import logging
from pathlib import Path

from app.data_processor.pdf_processor import OCRGeminiPipeline
from app.schema.extractor_schema import ContentType
from app.settings import (
    GEMINI_API_KEY,
    TESSERACT_PATH,
    TESSDATA_PATH,
    PDF_PATH,
    OUTPUT_DIR,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating the pipeline usage"""

    try:
        logger.info("Initializing OCR Gemini Pipeline...")

        pipeline = OCRGeminiPipeline(
            gemini_api_key=GEMINI_API_KEY,
            dpi=300,
            tesseract_path=TESSERACT_PATH,
            tessdata_path=TESSDATA_PATH,
        )

        logger.info("Starting PDF processing...")
        results = pipeline.process_pdf(
            pdf_path=str(PDF_PATH),
            page_range=(1, 49),
        )

        print_processing_summary(results)
        demonstrate_analysis_methods(pipeline)

        output_path = Path(OUTPUT_DIR) / "ocr_gemini_results.json"
        pipeline.export_results(str(output_path))

        embedding_data = pipeline.get_embedding_data()
        logger.info(f"Generated {len(embedding_data)} embedding items")
        logger.info("✅ Processing completed successfully!")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


def print_processing_summary(results: dict):
    """Print a summary of processing results"""

    stats = results["statistics"]

    print("\n" + "=" * 50)
    print("PROCESSING SUMMARY")
    print("=" * 50)
    print(f"📄 Total pages processed: {results['total_pages']}")
    print(f"🎯 Average confidence: {stats.avg_confidence:.1f}%")
    print(f"✅ High confidence pages: {stats.high_confidence_pages}")
    print(f"❓ MCQ questions found: {stats.mcq_questions}")
    print(f"📚 Vocabulary entries: {stats.vocabulary_entries}")
    print(f"📖 Narrative sections: {stats.narrative_sections}")
    print("=" * 50)


def demonstrate_analysis_methods(pipeline: OCRGeminiPipeline):
    """Demonstrate various analysis methods available in the pipeline"""

    print("\n" + "=" * 50)
    print("CONTENT ANALYSIS")
    print("=" * 50)

    # Get pages by content type
    mcq_pages = pipeline.get_pages_by_content_type(ContentType.MCQ)
    vocab_pages = pipeline.get_pages_by_content_type(ContentType.VOCABULARY)
    narrative_pages = pipeline.get_pages_by_content_type(ContentType.NARRATIVE)

    print(f"📝 MCQ pages: {len(mcq_pages)}")
    print(f"📖 Vocabulary pages: {len(vocab_pages)}")
    print(f"📰 Narrative pages: {len(narrative_pages)}")

    # Show high confidence pages
    high_conf_pages = pipeline.get_high_confidence_pages(threshold=80.0)
    print(f"⭐ High confidence pages (>80%): {len(high_conf_pages)}")

    # Demonstrate text search
    search_results = pipeline.search_text_in_pages("প্রশ্ন", case_sensitive=False)
    print(f"🔍 Pages containing 'প্রশ্ন': {len(search_results)}")

    # Show sample content from first page
    if pipeline.processed_pages:
        first_page = pipeline.processed_pages[0]
        print(f"\n📄 Sample from page {first_page.page_number}:")
        print(f"   Type: {first_page.content_type.value}")
        print(f"   Language: {first_page.language}")
        print(f"   Confidence: {first_page.confidence_score:.1f}%")
        print(f"   Text preview: {first_page.cleaned_text[:100]}...")

        if first_page.title:
            print(f"   Title: {first_page.title}")


'''

def save_embedding_data(embedding_data: list, output_dir: str):
    """Save embedding data to a separate JSON file"""

    import json
    from pathlib import Path

    output_path = Path(output_dir) / "embedding_data.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embedding_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Embedding data saved to {output_path}")


def process_specific_pages_example():
    """Example of processing specific pages individually"""

    GEMINI_API_KEY = "your_api_key_here"

    # Initialize pipeline
    pipeline = OCRGeminiPipeline(gemini_api_key=GEMINI_API_KEY)

    # Convert PDF to images first
    pdf_path = "your_pdf_file.pdf"
    images = pipeline.pdf_to_images(pdf_path)

    # Process specific pages
    for image, page_num in images[:3]:  # Process first 3 pages
        extracted_content = pipeline.process_single_page(image, page_num)

        print(f"Page {page_num}:")
        print(f"  Content Type: {extracted_content.content_type.value}")
        print(f"  Confidence: {extracted_content.confidence_score:.1f}%")
        print(f"  Text Length: {len(extracted_content.cleaned_text)} chars")
        print("-" * 40)


def batch_processing_example():
    """Example of processing multiple PDFs in batch"""

    GEMINI_API_KEY = "your_api_key_here"
    pdf_files = ["file1.pdf", "file2.pdf", "file3.pdf"]

    pipeline = OCRGeminiPipeline(gemini_api_key=GEMINI_API_KEY)

    all_results = {}

    for pdf_file in pdf_files:
        logger.info(f"Processing {pdf_file}...")

        try:
            results = pipeline.process_pdf(pdf_file)
            all_results[pdf_file] = results

            # Export results for each file
            output_name = f"results_{Path(pdf_file).stem}.json"
            pipeline.export_results(output_name)

        except Exception as e:
            logger.error(f"Failed to process {pdf_file}: {e}")

    logger.info(f"Batch processing completed. Processed {len(all_results)} files.")
'''

if __name__ == "__main__":
    main()
