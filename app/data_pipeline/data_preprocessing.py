# preprocessing_normalization.py

import os
import re
import unicodedata
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

# === Setup for Bengali ===
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("bn")

INPUT_DIR = "app/data/output"
OUTPUT_DIR = "app/data/normalized"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Normalization ===


def load_text_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None


def save_text_file(filepath, text):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


def normalize_bengali_text(text):
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = normalizer.normalize(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = (
        text.replace("?", "? ")
        .replace("!", "! ")
        .replace(",", ", ")
        .replace(":", ": ")
        .replace(";", "; ")
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_bengali_sentence(sentence):
    if not sentence:
        return []
    return indic_tokenize.trivial_tokenize(sentence, lang="bn")


# === Process Files ===


def process_and_save_file(filename):
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, f"normalized_{filename}")

    print(f"🔄 Processing: {input_path}")
    content = load_text_file(input_path)
    if content:
        normalized = normalize_bengali_text(content)
        save_text_file(output_path, normalized)
        print(f"✅ Saved: {output_path}")


if __name__ == "__main__":
    # List of your five known files
    file_list = [
        "narrative.txt",
        "glossary.txt",
        "author.txt",
        "lesson_intro.txt",
        "mcq_ans.txt",
    ]

    for filename in file_list:
        process_and_save_file(filename)
