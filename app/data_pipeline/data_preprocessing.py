# preprocessing.py

import re
from pathlib import Path


def normalize_bangla_text(text: str) -> str:
    """
    Normalize Bangla text by standardizing punctuation, whitespace, and common OCR errors.
    """
    # Standardize Bangla punctuation
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("…", "...")

    # Normalize extra whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n +", "\n", text)
    text = text.strip()

    # Remove duplicate punctuation
    text = re.sub(r"[।]{2,}", "।", text)
    text = re.sub(r"[!]{2,}", "!", text)
    text = re.sub(r"[?]{2,}", "?", text)

    # Remove artifacts like broken Unicode (if OCR was noisy)
    text = re.sub(r"[^\u0980-\u09FF\s.,!?\"'()\-\n।]", "", text)

    return text


def clean_file(input_path: Path, output_path: Path):
    """
    Read raw text file, normalize, and save cleaned version.
    """
    print(f"🧹 Cleaning {input_path.name}...")

    raw_text = input_path.read_text(encoding="utf-8")
    cleaned_text = normalize_bangla_text(raw_text)
    output_path.write_text(cleaned_text, encoding="utf-8")

    print(f"✅ Cleaned file saved to {output_path}")


def run_preprocessing(data_dir="app/data/output", output_dir="app/data/processed"):
    """
    Process all relevant files in the data directory.
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_map = {
        "narrative.txt": "cleaned_story.txt",
        "glossary.txt": "cleaned_glossary.txt",
        "author.txt": "cleaned_author.txt",
        "lesson_intro.txt": "cleaned_intro.txt",
        "mcq_ans.txt": "cleaned_knowledge.txt",
    }

    for filename, out_filename in file_map.items():
        input_file = data_path / filename
        output_file = output_path / out_filename

        if input_file.exists():
            clean_file(input_file, output_file)
        else:
            print(f"⚠️ File not found: {filename}")


if __name__ == "__main__":
    run_preprocessing()
