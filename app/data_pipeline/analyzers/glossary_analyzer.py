# pipeline/analyzers/glossary_analyzer.py

import numpy as np
from app.models.gemini_model import GeminiModel
import logging

logger = logging.getLogger(__name__)


class GlossaryAnalyzer:
    def __init__(self):
        self.model = GeminiModel()
        logger.info("Glossary Analyzer ready.")

    def _build_prompt(self) -> str:
        return """
            -You are a Bangla education expert. From the given image, extract all multiple-choice (MCQ) questions accurately.
            -Convert each glossary entry into a single, grammatically complete Bangla sentence that fully captures the definition.
            Example transformation:
             from this : গজানন :- দেবী দুর্গার দুই পুত্র; অগ্রজ গণেশ ও অনুজ কার্তিকেয়। দুর্গার কোলে থাকা দেব-সেনাপতি কার্তিকেয়কে বোঝানো হয়েছে। ব্যঙ্গার্থে প্রয়োেগ।
             to this: গজানন বলতে দেবী দুর্গার দুই পুত্র গণেশ ও কার্তিকেয়কে বোঝায়, এবং এটি ব্যঙ্গার্থেও ব্যবহার করা হয়।
            -These become self-contained knowledge statements.
            -You are an expert in Bangla language. Extract all word-meaning pairs from the image. -
            -Only return the glossary as plain text. Do not include metadata or JSON.-
             """

    def analyze_glossary(self, image: np.ndarray) -> str:
        prompt = self._build_prompt()
        return self.model.generate_text_from_image(prompt, image)
