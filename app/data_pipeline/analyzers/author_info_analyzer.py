# pipeline/analyzers/author_info_analyzer.py

import numpy as np
from app.models.gemini_model import GeminiModel
import logging

logger = logging.getLogger(__name__)


class AuthorInfoAnalyzer:
    def __init__(self):
        self.model = GeminiModel()
        logger.info("Author Info Analyzer ready.")

    def _build_prompt(self) -> str:
        return (
            "You are a Bangla literature expert. From this image, extract information of the author of the story — "
            "At the beginning of the content, include 'লেখক পরিচিতি' or an introductory phrase like 'অপরিচিতা গল্পের লেখক রবীন্দ্রনাথ ঠাকুর' to maintain clear context."
            "While formatting the extracted text, consistently refer to the author's name to maintain clear context. "
            "Do not include any metadata, explanations, or structured formatting — just return the main Bangla text as plain output."
        )

    def analyze_author_info(self, image: np.ndarray) -> str:
        prompt = self._build_prompt()
        return self.model.generate_text_from_image(prompt, image)
