# pipeline/analyzers/narrative_analyzer.py

import numpy as np
from app.models.gemini_model import GeminiModel
import logging

logger = logging.getLogger(__name__)


class NarrativeAnalyzer:
    def __init__(self):
        self.model = GeminiModel()
        logger.info("Narrative Analyzer ready.")

    def _build_prompt(self) -> str:
        return (
            "Ignore any header or banner text such as 'MINUTE SCHOOL HSC 26 অনলাইন ব্যাচ বাংলা ইংরেজি আইসিটি অনলাইন ব্যাচ সম্পর্কিত যেকোনো জিজ্ঞাসায়, কল করো 16910'. "
            "You're a Bangla literature expert. Extract the full narrative passage from this image, preserving paragraph breaks. "
            "Only return the story text as plain output. Do not include JSON or metadata."
        )

    def analyze_narrative(self, image: np.ndarray) -> str:
        prompt = self._build_prompt()
        return self.model.generate_text_from_image(prompt, image)
