"""
Main pipeline for orchestrating OCR and Gemini Vision processing
"""

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import json
import re
import io
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from app.schema.extractor_schema import (
    ContentType,
    ExtractedContent,
    ProcessingStatistics,
)
from .ocr_processor import OCRProcessor
from .gemini_vision import GeminiVisionAnalyzer

# Configure logging
logger = logging.getLogger(__name__)


class OCRGeminiPipeline:
    """Main pipeline for processing PDF documents with OCR + Gemini Vision"""

    def __init__(
        self,
        gemini_api_key: str,
        dpi: int = 300,
        tesseract_path: str = None,
        tessdata_path: str = None,
    ):
        """
        Initialize the processing pipeline

        Args:
            gemini_api_key: Google Gemini API key
            dpi: DPI for PDF to image conversion
            tesseract_path: Path to Tesseract executable
            tessdata_path: Path to tessdata directory
        """
        self.dpi = dpi
        self.processed_pages = []

        # Initialize processors
        self.ocr_processor = OCRProcessor(tesseract_path, tessdata_path)
        self.gemini_analyzer = GeminiVisionAnalyzer(gemini_api_key)

        logger.info("OCR Gemini Pipeline initialized successfully")

    def pdf_to_images(self, pdf_path: str) -> List[Tuple[np.ndarray, int]]:
        """
        Convert PDF pages to high-quality images

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of (image, page_number) tuples
        """
        images = []

        try:
            pdf_document = fitz.open(pdf_path)

            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]

                # Convert to image with high DPI
                mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)  # 72 is default DPI
                pix = page.get_pixmap(matrix=mat)

                # Convert to numpy array
                img_data = pix.tobytes("ppm")
                img = Image.open(io.BytesIO(img_data))
                img_array = np.array(img)

                images.append((img_array, page_num + 1))
                logger.info(f"Converted page {page_num + 1}/{pdf_document.page_count}")

            pdf_document.close()

        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise

        return images

    def process_single_page(
        self, image: np.ndarray, page_number: int
    ) -> ExtractedContent:
        """
        Process a single page image using combined OCR + Gemini approach

        Args:
            image: Page image as numpy array
            page_number: Page number

        Returns:
            ExtractedContent object
        """
        logger.info(f"Processing page {page_number}")

        # Extract text using OCR
        ocr_text, ocr_confidence, bounding_boxes = (
            self.ocr_processor.extract_text_with_confidence(image)
        )

        # Analyze with Gemini Vision
        gemini_text, gemini_analysis = self.gemini_analyzer.analyze_image(image)

        # Combine results
        combined_results = self._combine_extraction_results(
            ocr_text, ocr_confidence, bounding_boxes, gemini_text, gemini_analysis
        )

        # Create ExtractedContent object
        extracted_content = self._create_extracted_content(
            combined_results, page_number
        )

        return extracted_content

    def _combine_extraction_results(
        self,
        ocr_text: str,
        ocr_confidence: float,
        bounding_boxes: list,
        gemini_text: str,
        gemini_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Combine OCR and Gemini results for optimal accuracy

        Args:
            ocr_text: Text from OCR
            ocr_confidence: OCR confidence score
            bounding_boxes: OCR bounding boxes
            gemini_text: Text from Gemini
            gemini_analysis: Gemini analysis data

        Returns:
            Combined results dictionary
        """
        # Determine which text source is more reliable
        gemini_quality = gemini_analysis.get("quality_score", 0)

        if gemini_quality > ocr_confidence:
            primary_text = gemini_text
            backup_text = ocr_text
            primary_source = "gemini"
        else:
            primary_text = ocr_text
            backup_text = gemini_text
            primary_source = "ocr"

        # Clean and merge texts
        cleaned_text = self._clean_and_merge_text(primary_text, backup_text)

        return {
            "primary_text": primary_text,
            "backup_text": backup_text,
            "cleaned_text": cleaned_text,
            "primary_source": primary_source,
            "ocr_confidence": ocr_confidence,
            "gemini_analysis": gemini_analysis,
            "bounding_boxes": [bbox.__dict__ for bbox in bounding_boxes],
        }

    def _clean_and_merge_text(self, primary: str, backup: str) -> str:
        """
        Clean and merge text from different sources

        Args:
            primary: Primary text source
            backup: Backup text source

        Returns:
            Cleaned and merged text
        """
        if not primary and backup:
            return backup
        if not backup:
            return primary

        # Basic cleaning
        cleaned = re.sub(r"\s+", " ", primary.strip())

        # Fix common OCR errors for Bengali
        replacements = {
            "া়": "া",  # Fix extra diacritics
            "ৗ": "ৌ",  # Common confusion
            "ৈ": "ৈ",  # Preserve correct forms
            "।।": "।",  # Double periods
            "??": "?",  # Double question marks
        }

        for wrong, correct in replacements.items():
            cleaned = cleaned.replace(wrong, correct)

        return cleaned

    def _create_extracted_content(
        self, combined_results: Dict[str, Any], page_number: int
    ) -> ExtractedContent:
        """
        Create ExtractedContent object from combined results

        Args:
            combined_results: Combined extraction results
            page_number: Page number

        Returns:
            ExtractedContent object
        """
        gemini_analysis = combined_results["gemini_analysis"]

        # Determine content type with expanded enum support
        content_type_str = gemini_analysis.get("content_type", "mixed")
        try:
            content_type = ContentType(content_type_str.lower())
        except ValueError:
            content_type = ContentType.MIXED

        # Extract other properties
        language = gemini_analysis.get("language", "mixed")
        structured_data = gemini_analysis.get("structured_content", {})

        # Calculate overall confidence
        ocr_conf = combined_results["ocr_confidence"]
        gemini_quality = gemini_analysis.get("quality_score", 50)
        confidence_score = (ocr_conf + gemini_quality) / 2

        # Create embedding text
        embedding_text = self._create_embedding_text(
            combined_results["cleaned_text"], content_type, structured_data
        )

        # Extract title
        title = self._extract_title_from_text(combined_results["cleaned_text"])

        return ExtractedContent(
            content_type=content_type,
            title=title,
            raw_text=combined_results["primary_text"],
            cleaned_text=combined_results["cleaned_text"],
            structured_data=structured_data,
            language=language,
            confidence_score=confidence_score,
            page_number=page_number,
            bounding_boxes=combined_results["bounding_boxes"],
            embedding_text=embedding_text,
        )

    def _create_embedding_text(
        self, text: str, content_type: ContentType, structured_data: Dict
    ) -> str:
        """
        Create optimized text for embedding with support for all content types

        Args:
            text: Source text
            content_type: Type of content
            structured_data: Structured data from analysis

        Returns:
            Optimized embedding text
        """
        if content_type == ContentType.MCQ and "mcqs" in structured_data:
            # Create searchable MCQ text
            mcq_texts = []
            for mcq in structured_data["mcqs"]:
                question = mcq.get("question", "")
                options = " ".join(
                    [f"{k}: {v}" for k, v in mcq.get("options", {}).items()]
                )
                mcq_texts.append(f"প্রশ্ন: {question} বিকল্প: {options}")
            return " ".join(mcq_texts)[:2000]

        elif content_type == ContentType.VOCABULARY and "vocabulary" in structured_data:
            # Create searchable vocabulary text
            vocab_texts = []
            for vocab in structured_data["vocabulary"]:
                word = vocab.get("word", "")
                meaning = vocab.get("meaning", "")
                vocab_texts.append(f"{word}: {meaning}")
            return " ".join(vocab_texts)[:2000]

        elif (
            content_type == ContentType.LEARNING_OUTCOME
            and "learning_outcomes" in structured_data
        ):
            # Create searchable learning outcome text
            outcome_texts = []
            for outcome in structured_data["learning_outcomes"]:
                outcome_text = outcome.get("outcome", "")
                context = outcome.get("context", "")
                outcome_texts.append(f"শিক্ষণীয় উদ্দেশ্য: {outcome_text} {context}")
            return " ".join(outcome_texts)[:2000]

        elif (
            content_type == ContentType.SHORT_ANSWER and "questions" in structured_data
        ):
            # Create searchable short answer text
            qa_texts = []
            for qa in structured_data["questions"]:
                question = qa.get("question", "")
                answer = qa.get("answer", "")
                qa_texts.append(f"প্রশ্ন: {question} উত্তর: {answer}")
            return " ".join(qa_texts)[:2000]

        elif content_type == ContentType.GRAMMAR and "grammar_rules" in structured_data:
            # Create searchable grammar text
            grammar_texts = []
            for rule in structured_data["grammar_rules"]:
                rule_text = rule.get("rule", "")
                example = rule.get("example", "")
                grammar_texts.append(f"ব্যাকরণ: {rule_text} উদাহরণ: {example}")
            return " ".join(grammar_texts)[:2000]

        elif (
            content_type == ContentType.MATCHING and "matching_pairs" in structured_data
        ):
            # Create searchable matching text
            match_texts = []
            for pair in structured_data["matching_pairs"]:
                left = pair.get("left", "")
                right = pair.get("right", "")
                match_texts.append(f"{left} - {right}")
            return " ".join(match_texts)[:2000]

        elif (
            content_type == ContentType.FILL_IN_THE_BLANK
            and "blanks" in structured_data
        ):
            # Create searchable fill-in-the-blank text
            blank_texts = []
            for blank in structured_data["blanks"]:
                sentence = blank.get("sentence", "")
                answer = blank.get("answer", "")
                blank_texts.append(f"শূন্যস্থান: {sentence} উত্তর: {answer}")
            return " ".join(blank_texts)[:2000]

        elif (
            content_type == ContentType.COMPREHENSION and "passages" in structured_data
        ):
            # Create searchable comprehension text
            comp_texts = []
            for passage in structured_data["passages"]:
                passage_text = passage.get("text", "")
                questions = passage.get("questions", [])
                question_text = " ".join([q.get("question", "") for q in questions])
                comp_texts.append(f"অনুচ্ছেদ: {passage_text} প্রশ্ন: {question_text}")
            return " ".join(comp_texts)[:2000]

        elif (
            content_type in [ContentType.LITERARY_PROSE, ContentType.NARRATIVE]
            and "narrative" in structured_data
        ):
            # Create searchable narrative text
            narrative = structured_data["narrative"]
            title = narrative.get("title", "")
            author = narrative.get("author", "")
            paragraphs = " ".join(narrative.get("paragraphs", []))
            return f"শিরোনাম: {title} লেখক: {author} বিষয়বস্তু: {paragraphs}"[:2000]

        elif content_type == ContentType.POETRY and "poems" in structured_data:
            # Create searchable poetry text
            poem_texts = []
            for poem in structured_data["poems"]:
                title = poem.get("title", "")
                author = poem.get("author", "")
                lines = " ".join(poem.get("lines", []))
                poem_texts.append(f"কবিতা: {title} কবি: {author} পঙক্তি: {lines}")
            return " ".join(poem_texts)[:2000]

        elif content_type == ContentType.DIALOGUE and "dialogues" in structured_data:
            # Create searchable dialogue text
            dialogue_texts = []
            for dialogue in structured_data["dialogues"]:
                speakers = dialogue.get("speakers", [])
                exchanges = dialogue.get("exchanges", [])
                speaker_text = " ".join(speakers)
                exchange_text = " ".join([ex.get("text", "") for ex in exchanges])
                dialogue_texts.append(f"কথোপকথন: {speaker_text} {exchange_text}")
            return " ".join(dialogue_texts)[:2000]

        elif content_type == ContentType.SUMMARY and "summary" in structured_data:
            # Create searchable summary text
            summary = structured_data["summary"]
            main_points = summary.get("main_points", [])
            conclusion = summary.get("conclusion", "")
            return f"মূল বিষয়: {' '.join(main_points)} উপসংহার: {conclusion}"[:2000]

        elif content_type == ContentType.TABLE and "table_data" in structured_data:
            # Create searchable table text
            table = structured_data["table_data"]
            headers = " ".join(table.get("headers", []))
            rows = []
            for row in table.get("rows", []):
                rows.append(" ".join(str(cell) for cell in row))
            return f"সারণী শিরোনাম: {headers} তথ্য: {' '.join(rows)}"[:2000]

        # Clean text for embedding (fallback for other content types)
        clean_text = re.sub(r"\s+", " ", text.strip())
        return clean_text[:2000]

    def _extract_title_from_text(self, text: str) -> Optional[str]:
        """
        Extract title from text

        Args:
            text: Source text

        Returns:
            Extracted title or None
        """
        lines = text.split("\n")[:3]  # Check first 3 lines

        for line in lines:
            line = line.strip()
            if line and len(line) < 100:
                # Title patterns
                if (
                    line.endswith(":")
                    or re.match(r"^[০-৯১-৯]+\.", line)
                    or len(line.split()) <= 6
                ):
                    return line

        return None

    def process_pdf(
        self, pdf_path: str, page_range: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Process entire PDF with OCR + Gemini

        Args:
            pdf_path: Path to PDF file
            page_range: Optional (start_page, end_page) tuple

        Returns:
            Processing results dictionary
        """
        logger.info(f"Processing PDF: {pdf_path}")

        # Convert PDF to images
        images = self.pdf_to_images(pdf_path)

        # Apply page range filter if specified
        if page_range:
            start, end = page_range
            images = [
                (img, page_num) for img, page_num in images if start <= page_num <= end
            ]

        # Initialize results structure
        results = {
            "pdf_path": pdf_path,
            "total_pages": len(images),
            "processed_pages": [],
            "statistics": ProcessingStatistics(),
        }

        # Process each page
        for image, page_num in images:
            try:
                extracted_content = self.process_single_page(image, page_num)
                results["processed_pages"].append(extracted_content)

                # Update statistics
                self._update_statistics(results["statistics"], extracted_content)

                logger.info(
                    f"Page {page_num} processed - "
                    f"Type: {extracted_content.content_type.value}, "
                    f"Confidence: {extracted_content.confidence_score:.1f}%"
                )

            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                continue

        # Finalize statistics
        self._finalize_statistics(
            results["statistics"], len(results["processed_pages"])
        )

        # Store processed pages for later use
        self.processed_pages = results["processed_pages"]

        return results

    def _update_statistics(
        self, stats: ProcessingStatistics, content: ExtractedContent
    ):
        """
        Update processing statistics with support for all content types

        Args:
            stats: Statistics object to update
            content: Extracted content to analyze
        """
        stats.total_pages += 1

        if content.confidence_score > 80:
            stats.high_confidence_pages += 1

        # Count content types based on structured data
        structured_data = content.structured_data or {}

        if content.content_type == ContentType.MCQ:
            mcq_count = len(structured_data.get("mcqs", []))
            stats.mcq_questions += mcq_count
        elif content.content_type == ContentType.VOCABULARY:
            vocab_count = len(structured_data.get("vocabulary", []))
            stats.vocabulary_entries += vocab_count
        elif content.content_type in [
            ContentType.NARRATIVE,
            ContentType.LITERARY_PROSE,
        ]:
            stats.narrative_sections += 1
        elif content.content_type == ContentType.LEARNING_OUTCOME:
            outcome_count = len(structured_data.get("learning_outcomes", []))
            stats.learning_outcomes += outcome_count

    def _finalize_statistics(self, stats: ProcessingStatistics, total_pages: int):
        """
        Finalize statistics calculations

        Args:
            stats: Statistics object to finalize
            total_pages: Total number of processed pages
        """
        if total_pages > 0:
            # Calculate average confidence from processed pages
            total_confidence = sum(
                page.confidence_score for page in self.processed_pages
            )
            stats.avg_confidence = total_confidence / total_pages

    def get_embedding_data(self) -> List[Dict[str, Any]]:
        """
        Get data optimized for embedding

        Returns:
            List of embedding-ready data items
        """
        embedding_data = []

        for page in self.processed_pages:
            if page.confidence_score > 50:  # Only include decent quality pages
                embedding_item = {
                    "id": f"page_{page.page_number}_{page.content_type.value}",
                    "text": page.embedding_text,
                    "metadata": {
                        "page_number": page.page_number,
                        "content_type": page.content_type.value,
                        "language": page.language,
                        "confidence_score": page.confidence_score,
                        "title": page.title,
                        "has_structured_data": bool(page.structured_data),
                    },
                }
                embedding_data.append(embedding_item)

        return embedding_data

    def export_results(self, output_path: str) -> None:
        """
        Export processing results to JSON file

        Args:
            output_path: Path for output JSON file
        """
        export_data = {
            "pages": [
                {
                    "page_number": page.page_number,
                    "content_type": page.content_type.value,
                    "title": page.title,
                    "raw_text": page.raw_text,
                    "cleaned_text": page.cleaned_text,
                    "structured_data": page.structured_data,
                    "language": page.language,
                    "confidence_score": page.confidence_score,
                    "embedding_text": page.embedding_text,
                }
                for page in self.processed_pages
            ]
        }

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Results exported to {output_path}")

    def get_pages_by_content_type(
        self, content_type: ContentType
    ) -> List[ExtractedContent]:
        """
        Get pages filtered by content type

        Args:
            content_type: Content type to filter by

        Returns:
            List of matching ExtractedContent objects
        """
        return [
            page for page in self.processed_pages if page.content_type == content_type
        ]

    def get_high_confidence_pages(
        self, threshold: float = 80.0
    ) -> List[ExtractedContent]:
        """
        Get pages with confidence above threshold

        Args:
            threshold: Minimum confidence threshold

        Returns:
            List of high-confidence ExtractedContent objects
        """
        return [
            page for page in self.processed_pages if page.confidence_score >= threshold
        ]

    def search_text_in_pages(
        self, query: str, case_sensitive: bool = False
    ) -> List[ExtractedContent]:
        """
        Search for text across all processed pages

        Args:
            query: Search query
            case_sensitive: Whether search should be case sensitive

        Returns:
            List of pages containing the query
        """
        if not case_sensitive:
            query = query.lower()

        matching_pages = []
        for page in self.processed_pages:
            search_text = (
                page.cleaned_text if case_sensitive else page.cleaned_text.lower()
            )
            if query in search_text:
                matching_pages.append(page)

        return matching_pages

    def get_content_type_statistics(self) -> Dict[str, int]:
        """
        Get statistics for each content type

        Returns:
            Dictionary with content type counts
        """
        content_stats = {}
        for content_type in ContentType:
            count = len(self.get_pages_by_content_type(content_type))
            if count > 0:
                content_stats[content_type.value] = count
        return content_stats

    def get_pages_by_language(self, language: str) -> List[ExtractedContent]:
        """
        Get pages filtered by language

        Args:
            language: Language to filter by

        Returns:
            List of matching ExtractedContent objects
        """
        return [
            page
            for page in self.processed_pages
            if page.language.lower() == language.lower()
        ]

    def get_structured_content_by_type(
        self, content_type: ContentType
    ) -> List[Dict[str, Any]]:
        """
        Get structured content for a specific content type

        Args:
            content_type: Content type to extract structured data for

        Returns:
            List of structured data dictionaries
        """
        structured_content = []
        for page in self.get_pages_by_content_type(content_type):
            if page.structured_data:
                structured_content.append(
                    {
                        "page_number": page.page_number,
                        "title": page.title,
                        "confidence_score": page.confidence_score,
                        "structured_data": page.structured_data,
                    }
                )
        return structured_content
