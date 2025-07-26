# pipeline/analyzers/lesson_intro_analyzer.py

import numpy as np
from app.models.gemini_model import GeminiModel
import logging

logger = logging.getLogger(__name__)


class LessonIntroAnalyzer:
    def __init__(self):
        self.model = GeminiModel()
        logger.info("Lesson Intro Analyzer ready.")

    def _build_prompt(self) -> str:
        return (
            "Ignore any header or banner text such as 'MINUTE SCHOOL HSC 26 অনলাইন ব্যাচ বাংলা ইংরেজি আইসিটি অনলাইন ব্যাচ সম্পর্কিত যেকোনো জিজ্ঞাসায়, কল করো 16910'. "
            "You are a Bangla literature teacher. Extract the introductory lesson about the story  from this image. "
            "Return the full text in plain format, preserving line breaks. "
            "Do not include any metadata or formatting tags — just clean readable Bangla text."
        )

    def analyze_intro(self, image: np.ndarray) -> str:
        prompt = self._build_prompt()
        return self.model.generate_text_from_image(prompt, image)
