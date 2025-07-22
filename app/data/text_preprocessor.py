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
        cleaned = re.sub(r"[^\u0980-\u09FFa-zA-Z0-9\s.,;:!?()\\[\]“”\"\'\n]", "", text)
        # Remove repeated punctuation (e.g., "!!", "..")
        cleaned = re.sub(r"([.,;:!?])\\1+", r"\\1", cleaned)
        return cleaned

    def handle_punctuation(self, text: str) -> str:
        """
        Fix spacing around Bengali/English punctuation marks.
        """
        text = re.sub(r"\\s*([.,;:!?।])\\s*", r"\\1 ", text)
        return text.strip()

    def clean_artifacts(self, text: str) -> str:
        """
        Removes specific PDF/OCR artifacts like page markers and batch IDs.
        """
        # Remove explicit page headers (e.g., "--- Page X ---" or "Page X")
        text = re.sub(r"---?\s*Page\s+\d+\s*---?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Page\s+\d+", "", text, flags=re.IGNORECASE)

        # Remove specific batch/ID strings and similar bracketed noise
        # This regex attempts to catch variations of "কল আললাইন ব্যাচ" followed by numbers
        text = re.sub(r"\[কল\s*আললাইন\s*ব্যাচ”?\s*\d+\s*\d*\s*\d*?]", "", text)
        text = re.sub(
            r"\[লন\s*৯\s*অনলাইন\s*ব্যাচ]", "", text
        )  # Another specific batch string

        # Remove standalone single digits or specific unwanted symbols if they frequently appear as noise
        # Be cautious: adjust this based on your actual data. Removing all single digits might remove meaningful ones.
        # These are examples based on previous analysis of your text:
        text = re.sub(r"\b[৪১২৬৩৯]\b", "", text)  # Removes standalone Bengali digits
        text = re.sub(r"[ছ্ট]", "", text)  # Removes 'ছ্ট'

        # Collapse multiple spaces into a single space
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline.
        """
        text = self.normalize_unicode(text)
        # Apply artifact cleaning early to remove large noise patterns
        text = self.clean_artifacts(text)
        text = self.remove_unwanted_chars(text)
        text = self.handle_punctuation(text)
        # Final pass to ensure consistent spacing after all operations
        text = re.sub(r"\s+", " ", text).strip()
        return text


if __name__ == "__main__":
    from pathlib import Path

    # Define paths
    input_file_path = Path("data/processed/bangla_extracted_text.txt")
    output_file_path = Path("data/processed/preprocessed_text.txt")

    # Ensure input file exists
    if not input_file_path.exists():
        print(f"Error: Input file not found at {input_file_path}")
    else:
        # Read the extracted text
        with open(input_file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Initialize preprocessor
        preprocessor = BengaliTextPreprocessor()

        # Preprocess the text
        preprocessed_text = preprocessor.preprocess(raw_text)

        # Write the preprocessed text to a new file
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(preprocessed_text)

        print(f"Preprocessing complete. Output saved to {output_file_path}")
