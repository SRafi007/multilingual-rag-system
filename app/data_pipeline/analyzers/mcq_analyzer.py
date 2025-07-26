# pipeline/analyzers/mcq_analyzer.py

import numpy as np
from app.models.gemini_model import GeminiModel
import logging

logger = logging.getLogger(__name__)


class MCQAnalyzer:
    def __init__(self):
        self.model = GeminiModel()
        logger.info("MCQ Analyzer ready.")

    def _build_prompt(self) -> str:
        return """
        You are a Bangla education expert. From the given image, extract all multiple-choice (MCQ) questions accurately.

Identify the correct answer for each MCQ (answer_map).

Use the correct option and question to generate a meaningful, natural Bangla sentence conveying the knowledge embedded in the question.

Example transformation:
From this question:
"অনুপমের বাবা কী করে জীবিকা নির্বাহ করতেন?"
(ক): ডাক্তারি
(খ): ওকালতি
(গ): মাস্টারি
(ঘ): ব্যবসা
উত্তর: খ
→ Output: অনুপমের বাবা জীবিকা নির্বাহের জন্য ওকালতি করতেন।

Only return plain Bangla text with one transformed sentence per question.

Do not include JSON, metadata, or any explanation.
            """

    def analyze_mcq(self, image: np.ndarray) -> str:
        prompt = self._build_prompt()
        return self.model.generate_text_from_image(prompt, image)
