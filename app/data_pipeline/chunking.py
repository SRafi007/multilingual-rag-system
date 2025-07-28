import os
import re
import json
from deep_translator import GoogleTranslator

# --- Configuration ---
INPUT_FILES = {
    "story": "app/data/normalized/resolved_narrative.txt",
    "glossary": "app/data/normalized/normalized_glossary.txt",
    "author": "app/data/normalized/normalized_author.txt",
    "intro": "app/data/normalized/normalized_mcq_ans.txt",
    "knowledge": "app/data/normalized/normalized_lesson_intro.txt",
}
OUTPUT_FILE = "app/data/chunks/processed_chunks.json"
TRANSLATE = True  # Set to False if you don’t want English translations

import time

# Example protected terms
PROTECTED_TERMS = {
    "অপরিচিতা": "APORICHITA",  # temp placeholder
    # Add more terms as needed
}


def protect_terms(text):
    for bn_term, placeholder in PROTECTED_TERMS.items():
        text = text.replace(bn_term, placeholder)
    return text


def restore_terms(text):
    for bn_term, placeholder in PROTECTED_TERMS.items():
        text = text.replace(placeholder, bn_term)
    return text


def safe_translate(sentence, retries=3, delay=1):
    protected_sentence = protect_terms(sentence)

    for attempt in range(retries):
        try:
            translated = GoogleTranslator(source="bn", target="en").translate(
                protected_sentence
            )
            return restore_terms(translated)
        except Exception as e:
            print(f"[!] Translation error (attempt {attempt + 1}): {e}")
            time.sleep(delay)

    return None


# --- Utility Functions ---
def bn_sentence_tokenizer(text):
    return [s.strip() for s in re.split(r"(?<=[।!?])\s+", text) if s.strip()]


def process_file(file_path, section_name, chunk_prefix, translate=False):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    sentences = bn_sentence_tokenizer(raw_text)
    chunks = []

    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue

        chunk_id = f"{chunk_prefix}-{str(i+1).zfill(3)}"
        chunk = {
            "chunk_id": chunk_id,
            "text_bn": sentence,
            "section": section_name,
            "source": os.path.basename(file_path),
            "index": i + 1,
            "metadata": {
                "language": "bn",
                "section": section_name,
                "translated": translate,
            },
        }

        if translate:
            if len(sentence) > 5000:
                print(
                    f"[!] Skipped translation for {chunk_id}: Sentence too long ({len(sentence)} chars)"
                )
                chunk["text_en"] = None
            else:
                try:
                    translated = safe_translate(sentence)
                    chunk["text_en"] = translated

                    chunk["text_en"] = translated
                except Exception as e:
                    print(f"[!] Translation failed for {chunk_id}: {e}")
                    chunk["text_en"] = None

        chunks.append(chunk)

    return chunks


# --- Main Execution ---
def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    all_chunks = []

    for idx, (section, path) in enumerate(INPUT_FILES.items()):
        if not os.path.exists(path):
            print(f"[!] File not found: {path}")
            continue
        print(f"[+] Processing '{section}' from '{path}'...")
        prefix = section.upper()[:3]
        section_chunks = process_file(path, section, prefix, translate=TRANSLATE)
        all_chunks.extend(section_chunks)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(
        f"\n✅ Processing complete. Saved {len(all_chunks)} chunks to '{OUTPUT_FILE}'"
    )


if __name__ == "__main__":
    main()
