"""
Gemini Vision API integration for advanced text analysis and content structuring
"""

import google.generativeai as genai
import numpy as np
from PIL import Image, ImageEnhance
import json
import logging
from typing import Dict, Any, Tuple

from app.schema.extractor_schema import ContentType

# Configure logging
logger = logging.getLogger(__name__)


class GeminiVisionAnalyzer:
    """Handles text extraction and analysis using Google Gemini Vision API"""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
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
        2. Identify the content type (MCQ, vocabulary, narrative, poetry, etc.)
        3. Detect language (Bengali, English, or mixed)
        4. Structure the content appropriately
        5. For MCQs: identify question numbers, questions, and options (ক, খ, গ, ঘ)
        6. For vocabulary: identify word-meaning pairs
        7. Rate the text quality and readability (1-100)
        
        Return response in this JSON format:
        {
            "extracted_text": "complete text from image",
            "content_type": "mcq|narrative|vocabulary|poetry|dialogue|table|mixed",
            "language": "bangla|english|mixed",
            "quality_score": 85,
            "structured_content": {
                "mcqs": [...] or "vocabulary": [...] or "paragraphs": [...]
            },
            "layout_info": {
                "has_columns": true/false,
                "has_tables": true/false,
                "text_regions": [...],
                "difficulty_level": "easy|medium|hard"
            }
        }
        
        IMPORTANT: Preserve Bengali text exactly as shown. Do not transliterate or translate.
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

            # Generate content using Gemini Vision
            response = self.model.generate_content([prompt, enhanced_image])

            # Parse response
            extracted_text, analysis_data = self._parse_gemini_response(response.text)

            logger.info(
                f"Gemini analysis completed - Quality score: {analysis_data.get('quality_score', 'N/A')}"
            )

            return extracted_text, analysis_data

        except Exception as e:
            logger.error(f"Gemini vision analysis failed: {e}")
            return "", {"error": str(e), "quality_score": 0}

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

            # Parse JSON
            analysis_data = json.loads(clean_response)
            extracted_text = analysis_data.get("extracted_text", "")

            # Validate content type
            content_type_str = analysis_data.get("content_type", "mixed").lower()
            try:
                ContentType(content_type_str)
            except ValueError:
                logger.warning(
                    f"Invalid content type '{content_type_str}', defaulting to 'mixed'"
                )
                analysis_data["content_type"] = "mixed"

            return extracted_text, analysis_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {e}")
            logger.debug(f"Raw response: {response_text[:500]}...")

            # Fallback: extract text manually
            fallback_text = self._extract_text_fallback(response_text)
            fallback_data = {
                "extracted_text": fallback_text,
                "content_type": "mixed",
                "language": "mixed",
                "quality_score": 30,
                "error": "JSON parsing failed",
                "structured_content": {},
                "layout_info": {},
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
                if line and not line.startswith("content_type"):
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
