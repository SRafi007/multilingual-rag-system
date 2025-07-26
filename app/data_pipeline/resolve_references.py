# resolve_references.py

import re
import unicodedata
from pathlib import Path


def resolve_pronouns(text: str) -> str:
    """
    Replace 1st-person Bangla pronouns with the character name 'অনুপম'.
    Handles punctuation and Unicode normalization.
    """
    # Normalize Unicode (handles issues with composed characters like ি + ম)
    text = unicodedata.normalize("NFC", text)

    # Mapping from pronoun to character name (ordered by length to avoid partial overlaps)
    mappings = {
        r"আমারই": "অনুপমেরই",
        r"আমাদের": "অনুপমদের",
        r"আমাকে": "অনুপমকে",
        r"আমার": "অনুপমের",
        r"আমি": "অনুপম",
        r"নিজে": "অনুপম নিজে",
    }

    for pronoun, replacement in mappings.items():
        # Replace whole word or followed by punctuation/suffix
        pattern = rf"\b{pronoun}(?=[\s।.,!?\"“”’\'():;\-\n]|$)"
        text = re.sub(pattern, replacement, text)

    return text


def process_story(input_path: Path, output_path: Path):
    """
    Load cleaned story file, resolve pronouns, save resolved version.
    """
    print(f"🔄 Resolving pronouns in: {input_path.name}")
    raw_text = input_path.read_text(encoding="utf-8")
    resolved_text = resolve_pronouns(raw_text)
    output_path.write_text(resolved_text, encoding="utf-8")
    print(f"✅ Resolved file saved to: {output_path}")


def run_resolution(cleaned_dir="app/data/processed", resolved_dir="app/data/resolved"):
    """
    Wrapper to resolve pronouns from cleaned story file.
    """
    input_file = Path(cleaned_dir) / "cleaned_story.txt"
    output_file = Path(resolved_dir) / "resolved_story.txt"
    Path(resolved_dir).mkdir(parents=True, exist_ok=True)

    if input_file.exists():
        process_story(input_file, output_file)
    else:
        print(f"❌ File not found: {input_file}")


if __name__ == "__main__":
    run_resolution()
