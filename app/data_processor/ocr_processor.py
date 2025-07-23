"""
OCR processing module using Tesseract for text extraction
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
import os
import logging
from typing import List, Dict, Tuple

from app.schema.extractor_schema import BoundingBox

# Configure logging
logger = logging.getLogger(__name__)


class OCRProcessor:
    """Handles OCR processing using Tesseract with Bengali and English support"""

    def __init__(self, tesseract_path: str = None, tessdata_path: str = None):
        """
        Initialize OCR processor

        Args:
            tesseract_path: Path to Tesseract executable
            tessdata_path: Path to tessdata directory
        """
        self._setup_tesseract_paths(tesseract_path, tessdata_path)

        # Configure Tesseract for Bengali + English
        self.tesseract_config = {
            "lang": "ben+eng",  # Bengali + English
            "config": "--oem 3 --psm 6 -c preserve_interword_spaces=1",
        }

        # Verify installation
        self._verify_tesseract()

    def _setup_tesseract_paths(self, tesseract_path: str, tessdata_path: str):
        """Setup Tesseract paths"""
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            # Default Windows path
            pytesseract.pytesseract.tesseract_cmd = (
                r"D:\softwares\code_related\Tesseract-OCR\tesseract.exe"
            )

        if tessdata_path:
            os.environ["TESSDATA_PREFIX"] = tessdata_path
        else:
            # Default tessdata path
            os.environ["TESSDATA_PREFIX"] = (
                r"D:\softwares\code_related\Tesseract-OCR\tessdata"
            )

    def _verify_tesseract(self):
        """Verify Tesseract installation and Bengali language support"""
        try:
            # Check if Tesseract is installed
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")

            # Check available languages
            langs = pytesseract.get_languages()
            logger.info(f"Available languages: {langs}")

            if "ben" not in langs:
                logger.warning(
                    "Bengali language pack not found. Make sure Bengali tessdata is available at: "
                    f"{os.environ.get('TESSDATA_PREFIX', 'default tessdata path')}"
                )
            else:
                logger.info("Bengali OCR support available")

        except Exception as e:
            logger.error(f"Tesseract verification failed: {e}")
            logger.error(
                "Please check if Tesseract is properly installed and paths are correct"
            )
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced image preprocessing for better OCR results

        Args:
            image: Input image as numpy array

        Returns:
            Preprocessed image
        """
        # Convert to PIL Image for advanced processing
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.5)

        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(2.0)

        # Convert back to numpy for OpenCV processing
        img = np.array(pil_image)

        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Resize image for better OCR (if too small)
        height, width = cleaned.shape
        if height < 600:
            scale_factor = 600 / height
            new_width = int(width * scale_factor)
            cleaned = cv2.resize(
                cleaned, (new_width, 600), interpolation=cv2.INTER_CUBIC
            )

        return cleaned

    def extract_text_with_confidence(
        self, image: np.ndarray
    ) -> Tuple[str, float, List[BoundingBox]]:
        """
        Extract text using OCR with confidence scores and bounding boxes

        Args:
            image: Input image

        Returns:
            Tuple of (extracted_text, confidence_score, bounding_boxes)
        """
        # Preprocess image
        processed_img = self.preprocess_image(image)

        # Get detailed OCR data
        ocr_data = pytesseract.image_to_data(
            processed_img,
            lang=self.tesseract_config["lang"],
            config=self.tesseract_config["config"],
            output_type=pytesseract.Output.DICT,
        )

        # Extract text with confidence
        extracted_text = ""
        total_confidence = 0
        word_count = 0
        bounding_boxes = []

        for i in range(len(ocr_data["text"])):
            word = ocr_data["text"][i].strip()
            confidence = int(ocr_data["conf"][i])

            if word and confidence > 30:  # Filter low confidence words
                extracted_text += word + " "
                total_confidence += confidence
                word_count += 1

                # Create bounding box object
                bbox = BoundingBox(
                    word=word,
                    confidence=confidence,
                    x=ocr_data["left"][i],
                    y=ocr_data["top"][i],
                    width=ocr_data["width"][i],
                    height=ocr_data["height"][i],
                )
                bounding_boxes.append(bbox)

        avg_confidence = total_confidence / max(word_count, 1)

        return extracted_text.strip(), avg_confidence, bounding_boxes

    def get_tesseract_languages(self) -> List[str]:
        """Get list of available Tesseract languages"""
        try:
            return pytesseract.get_languages()
        except Exception as e:
            logger.error(f"Failed to get Tesseract languages: {e}")
            return []

    def set_custom_config(self, lang: str = None, config: str = None):
        """
        Set custom Tesseract configuration

        Args:
            lang: Language configuration (e.g., 'ben+eng')
            config: Custom OCR configuration string
        """
        if lang:
            self.tesseract_config["lang"] = lang
        if config:
            self.tesseract_config["config"] = config

        logger.info(f"Updated Tesseract config: {self.tesseract_config}")
