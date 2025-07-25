"""
Gemini Vision API integration for advanced text analysis and content structuring
"""

import google.generativeai as genai
import numpy as np
from PIL import Image, ImageEnhance
import json
import logging
from typing import Dict, Any, Tuple
import re
from app.schema.extractor_schema import ContentType

# Configure logging
logger = logging.getLogger(__name__)


def _normalize_quotes(text: str) -> str:
    # Fix curly quotes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")

    # Remove illegal commas like: "আমি দুঃখিত, , কিন্তু"
    text = re.sub(r",\s*,+", ",", text)

    # Remove broken words with extra newlines or multiple spaces (Gemini hallucination)
    text = re.sub(r"([^\s])\s{2,}([^\s])", r"\1 \2", text)  # Fix over-spaced words
    text = re.sub(r'"\s*\n\s*', '"', text)  # Fix line breaks inside string literals

    # Ensure no trailing commas before closing braces/brackets
    text = re.sub(r",\s*([}\]])", r"\1", text)

    return text


class GeminiVisionAnalyzer:
    """Handles text extraction and analysis using Google Gemini Vision API"""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Initialize Gemini Vision analyzer

        Args:
            api_key: Google Gemini API key
            model_name: Gemini model to use
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        self.api_key = api_key

        logger.info(f"Initialized Gemini Vision with model: {model_name}")

    def _create_analysis_prompt(self) -> str:
        """Create comprehensive prompt for Gemini Vision analysis"""

        prompt = """
        You are an expert in Bengali (Bangla) language and educational content analysis.
        
        Analyze this image and:
        1. Extract ALL text accurately (preserve Bengali script exactly)
        2. Identify the content type from these options: mcq, short_answer, learning_outcome, vocabulary, narrative, literary_prose, poetry, dialogue, grammar, matching, fill_in_the_blank, table, summary, comprehension, instruction, mixed
        3. Detect language (Bengali, English, or mixed)
        4. Structure the content appropriately
        5. For MCQs: identify question numbers, questions, and options (ক, খ, গ, ঘ)
        6. For vocabulary: identify word-meaning pairs
        7. For learning outcomes: identify specific learning objectives
        8. For narratives/literary prose: identify paragraphs
        9. For short answers: identify questions and expected answer formats
        10. For grammar: identify rules, examples, and exercises
        11. For matching: identify items to be matched
        12. For fill-in-the-blank: identify sentences with blanks and possible answers
        13. For comprehension: identify passages and related questions
        14. Rate the text quality and readability (1-100)
        15. For MCQs: if the correct answer is given write it.
        16. For MCQs: answer is not given look for this type of pattern SL Ans ১ খ ২ গ ৩ খ and match the question number with this.
        17. For MCqs: assign the correct question number, check it again.
        
        Return response in this JSON format:
        {
            "extracted_text": "complete text from image",
            "content_type": "mcq|short_answer|learning_outcome|vocabulary|narrative|literary_prose|poetry|dialogue|grammar|matching|fill_in_the_blank|table|summary|comprehension|instruction|mixed",
            "language": "bangla|english|mixed",
            "quality_score": 85,
            "structured_content": {
                "mcqs": [{ "question_number": n, "question": "...", "options": {"ক": "...", "খ": "...", "গ": "...", "ঘ": "..."},"answer":"null/ক"}],
                "vocabulary": [{"word": "...", "meaning": "...", "language": "bangla"}],
                "learning_outcomes": [{"outcome": "...","language": "bangla"}],
                "narratives": [{"title": "...", "paragraphs": ["...", "..."]}],
                "short_answers": [{"question": "...", "expected_format": "..."}],
                "grammar_rules": [{"rule": "...", "examples": ["...", "..."], "exercises": ["...", "..."]}],
                "matching_items": [{"left_items": ["...", "..."], "right_items": ["...", "..."], "instructions": "..."}],
                "fill_blanks": [{"sentence": "...", "blanks": ["...", "..."], "options": ["...", "...", "..."]}],
                "comprehension": [{"passage": "...", "questions": [{"question": "...", "type": "mcq|short"}]}],
                "tables": [{"headers": ["...", "..."], "rows": [["...", "..."]]}],
                "summaries": [{"title": "...", "content": "...", "key_points": ["...", "..."]}],
                "instructions": [{"step": 1, "instruction": "...", "details": "..."}]
            },
            "layout_info": {
                "has_columns": true/false,
                "has_tables": true/false,
                "text_regions": [...],
                "difficulty_level": "easy|medium|hard",
                "page_structure": "single_column|multi_column|mixed",
                "has_images": true/false,
                "has_diagrams": true/false
            }
        }
        
        IMPORTANT: 
        - Preserve Bengali text exactly as shown. Do not transliterate or translate.
        - Choose the most appropriate single content type, use 'mixed' only when multiple distinct types are present
        - For educational content, prioritize identifying learning outcomes when present
        - Pay attention to question numbering and formatting patterns
        """

        return prompt

    def _enhance_image_for_analysis(self, image: np.ndarray) -> Image.Image:
        """
        Enhance image quality before sending to Gemini

        Args:
            image: Input image as numpy array

        Returns:
            Enhanced PIL Image
        """
        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Enhance contrast for better text recognition
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.2)

        # Enhance sharpness for clearer text
        sharpness_enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = sharpness_enhancer.enhance(1.1)

        # Ensure image is not too large (Gemini has size limits)
        max_size = 2048
        if max(pil_image.size) > max_size:
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        return pil_image

    def analyze_image(self, image: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        """
        Extract and analyze text using Gemini Vision API

        Args:
            image: Input image as numpy array

        Returns:
            Tuple of (extracted_text, analysis_data)
        """
        try:
            # Enhance image quality
            enhanced_image = self._enhance_image_for_analysis(image)

            # Create analysis prompt
            prompt = self._create_analysis_prompt()

            for attempt in range(2):  # Retry up to 2 times
                try:
                    # Generate content using Gemini Vision
                    response = self.model.generate_content([prompt, enhanced_image])

                    # Parse response
                    extracted_text, analysis_data = self._parse_gemini_response(
                        response.text
                    )

                    logger.info(
                        f"Gemini analysis completed - Attempt {attempt + 1} - "
                        f"Content type: {analysis_data.get('content_type', 'N/A')}, "
                        f"Quality score: {analysis_data.get('quality_score', 'N/A')}"
                    )

                    return extracted_text, analysis_data

                except json.JSONDecodeError as parse_error:
                    logger.warning(
                        f"JSON parsing failed on attempt {attempt + 1}: {parse_error}"
                    )
                    if attempt == 1:
                        raise  # Rethrow after final attempt

        except Exception as e:
            logger.error(f"Gemini vision analysis failed: {e}")
            return "", {"error": str(e), "quality_score": 0, "content_type": "mixed"}

    def _parse_gemini_response(self, response_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse JSON response from Gemini

        Args:
            response_text: Raw response text from Gemini

        Returns:
            Tuple of (extracted_text, parsed_analysis_data)
        """
        try:
            # Clean up response if it has markdown formatting
            clean_response = response_text.strip()
            if clean_response.startswith("```json"):
                clean_response = (
                    clean_response.replace("```json", "").replace("```", "").strip()
                )

            # Normalize curly quotes
            clean_response = _normalize_quotes(clean_response)

            # Parse JSON
            try:
                analysis_data = json.loads(clean_response)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse Gemini JSON response at pos {e.pos}: {e}"
                )
                logger.error(
                    f"Snippet around error: {clean_response[e.pos-50:e.pos+50]}"
                )
                raise
            extracted_text = analysis_data.get("extracted_text", "")

            # Validate content type against new enum values
            content_type_str = analysis_data.get("content_type", "mixed").lower()
            valid_content_types = [ct.value for ct in ContentType]

            if content_type_str not in valid_content_types:
                logger.warning(
                    f"Invalid content type '{content_type_str}', defaulting to 'mixed'. "
                    f"Valid types: {valid_content_types}"
                )
                analysis_data["content_type"] = "mixed"

            # Ensure structured_content exists
            if "structured_content" not in analysis_data:
                analysis_data["structured_content"] = {}

            # Ensure layout_info exists with all expected fields
            if "layout_info" not in analysis_data:
                analysis_data["layout_info"] = {}

            # Set default layout_info values if missing
            layout_defaults = {
                "has_columns": False,
                "has_tables": False,
                "text_regions": [],
                "difficulty_level": "medium",
                "page_structure": "single_column",
                "has_images": False,
                "has_diagrams": False,
            }

            for key, default_value in layout_defaults.items():
                if key not in analysis_data["layout_info"]:
                    analysis_data["layout_info"][key] = default_value

            return extracted_text, analysis_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {e}")
            # logger.debug(f"Raw response: {response_text[:500]}...")
            logger.error(
                f"Full Gemini response text:\n{response_text}"
            )  # just for testing

            # Fallback: extract text manually
            fallback_text = self._extract_text_fallback(response_text)
            fallback_data = {
                "extracted_text": fallback_text,
                "content_type": "mixed",
                "language": "mixed",
                "quality_score": 30,
                "error": "JSON parsing failed",
                "structured_content": {},
                "layout_info": {
                    "has_columns": False,
                    "has_tables": False,
                    "text_regions": [],
                    "difficulty_level": "medium",
                    "page_structure": "single_column",
                    "has_images": False,
                    "has_diagrams": False,
                },
            }

            return fallback_text, fallback_data

    def _extract_text_fallback(self, response_text: str) -> str:
        """
        Fallback method to extract text when JSON parsing fails

        Args:
            response_text: Raw response text

        Returns:
            Extracted text
        """
        # Try to find text within quotes or after certain keywords
        lines = response_text.split("\n")
        text_lines = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith("{") and not line.startswith("}"):
                # Remove common JSON artifacts
                line = line.replace('"extracted_text":', "").replace('"', "").strip()
                if line and not any(
                    keyword in line.lower()
                    for keyword in [
                        "content_type",
                        "language",
                        "quality_score",
                        "structured_content",
                    ]
                ):
                    text_lines.append(line)

        return " ".join(text_lines[:10])  # Take first 10 relevant lines

    def batch_analyze_images(self, images: list) -> list:
        """
        Analyze multiple images in batch

        Args:
            images: List of numpy arrays

        Returns:
            List of analysis results
        """
        results = []

        for i, image in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)} with Gemini")
            text, analysis = self.analyze_image(image)
            results.append((text, analysis))

        return results

    def get_supported_content_types(self) -> list:
        """Get list of supported content types"""
        return [content_type.value for content_type in ContentType]

    def validate_structured_content(
        self, analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and clean structured content based on content type

        Args:
            analysis_data: Raw analysis data from Gemini

        Returns:
            Validated and cleaned analysis data
        """
        content_type = analysis_data.get("content_type", "mixed")
        structured_content = analysis_data.get("structured_content", {})

        # Content type specific validation
        if content_type == "mcq":
            if "mcqs" in structured_content:
                # Validate MCQ structure
                valid_mcqs = []
                for mcq in structured_content["mcqs"]:
                    if isinstance(mcq, dict) and "question" in mcq and "options" in mcq:
                        valid_mcqs.append(mcq)
                structured_content["mcqs"] = valid_mcqs

        elif content_type == "vocabulary":
            if "vocabulary" in structured_content:
                # Validate vocabulary entries
                valid_vocab = []
                for entry in structured_content["vocabulary"]:
                    if (
                        isinstance(entry, dict)
                        and "word" in entry
                        and "meaning" in entry
                    ):
                        valid_vocab.append(entry)
                structured_content["vocabulary"] = valid_vocab

        elif content_type == "learning_outcome":
            if "learning_outcomes" in structured_content:
                # Validate learning outcomes
                valid_outcomes = []
                for outcome in structured_content["learning_outcomes"]:
                    if isinstance(outcome, dict) and "outcome" in outcome:
                        valid_outcomes.append(outcome)
                structured_content["learning_outcomes"] = valid_outcomes

        elif content_type in ["narrative", "literary_prose"]:
            if "narratives" in structured_content:
                # Validate narrative structure
                valid_narratives = []
                for narrative in structured_content["narratives"]:
                    if isinstance(narrative, dict) and "paragraphs" in narrative:
                        valid_narratives.append(narrative)
                structured_content["narratives"] = valid_narratives

        analysis_data["structured_content"] = structured_content
        return analysis_data
