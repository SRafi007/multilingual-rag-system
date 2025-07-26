# models/gemini_model.py

import google.generativeai as genai
from PIL import Image, ImageEnhance
import numpy as np
import logging
import time
from config.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)


class GeminiModel:
    def __init__(self, api_key: str = GEMINI_API_KEY, model_name: str = GEMINI_MODEL):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.delay = 9  # Gemini API rate limit delay
        logger.info(f"Gemini model initialized: {model_name}")

    def enhance_image(self, image: np.ndarray) -> Image.Image:
        """Enhance contrast and sharpness of image for better OCR."""
        img = Image.fromarray(image)
        img = ImageEnhance.Contrast(img).enhance(1.2)
        img = ImageEnhance.Sharpness(img).enhance(1.1)

        max_dim = 2048
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        return img

    def generate_text_from_image(self, prompt: str, image: np.ndarray) -> str:
        """Send image + prompt to Gemini and return plain text."""
        try:
            enhanced = self.enhance_image(image)
            logger.info("Calling Gemini API...")

            time.sleep(self.delay)  # Respect rate limit

            response = self.model.generate_content([prompt, enhanced])
            text = response.text.strip()

            # Remove possible markdown block
            if text.startswith("```"):
                text = text.split("```")[-1].strip()

            return text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return ""
