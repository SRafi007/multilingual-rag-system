# app/data/text_preprocessor.py

import re
import unicodedata


class BengaliTextPreprocessor:
    def __init__(self):
        # You can add stopwords or special cleaning rules here later if needed
        pass

    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Bengali Unicode characters.
        Example: decomposed characters → composed form
        """
        return unicodedata.normalize("NFC", text)

    def remove_unwanted_chars(self, text: str) -> str:
        """
        Removes noise characters while keeping valid Bengali, English, numbers, and punctuation.
        """
        # Remove unwanted characters outside Bengali block and basic punctuation
        cleaned = re.sub(r"[^\u0980-\u09FFa-zA-Z0-9\s.,;:!?()\[\]“”\"\'\n]", "", text)
        # Remove repeated punctuation (e.g., "!!", "..")
        cleaned = re.sub(r"([.,;:!?])\1+", r"\1", cleaned)
        return cleaned

    def handle_punctuation(self, text: str) -> str:
        """
        Fix spacing around Bengali/English punctuation marks.
        """
        text = re.sub(r"\s*([.,;:!?।])\s*", r"\1 ", text)
        return text.strip()

    def validate_quality(self, text: str, min_len: int = 20) -> bool:
        """
        Check if the text chunk is of reasonable quality (non-gibberish).
        """
        bangla_chars = re.findall(r"[\u0980-\u09FF]", text)
        return len(text) >= min_len and len(bangla_chars) > 10

    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline.
        """
        text = self.normalize_unicode(text)
        text = self.remove_unwanted_chars(text)
        text = self.handle_punctuation(text)
        return text


if __name__ == "__main__":
    from pathlib import Path

    sample_path = Path("data/processed/bangla_extracted_text.txt")
    raw_text = sample_path.read_text(encoding="utf-8")

    preprocessor = BengaliTextPreprocessor()
    cleaned = preprocessor.preprocess(raw_text)

    Path("data/processed/preprocessed_text.txt").write_text(cleaned, encoding="utf-8")
    print("[✅] Preprocessed text saved.")
    print(cleaned[:1000])  # Preview
