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
    "অপরিচিতা": "APORICHITA",
    "অনুপম": "ANUPAM",
    "কল্যাণী": "KALYANI",
    "মামা": "Anupam's maternal uncle",
    "শম্ভুনাথ": "SHAMBHUNATH",
    "হরিশ": "HARISH",
    "বিনুদাদা": "BINUDADA",
    "শম্ভুনাথ সেন": "SHAMBHUNATH SEN",
    "শম্ভুনাথবাবুর উকিল-বন্ধু": "SHAMBHUNATHBABUR UKIL-BONDHU",
    "অন্নপূর্ণা": "ANNAPURNA",
    "গজানন": "GAJANAN",
    "মনু": "MANU",
    "প্রজাপতি": "PRAJAPATI",
    "পঞ্চশর": "PANCHASHAR",
    "সতী": "SATI",
    "শিব": "SHIVA",
    "দময়ন্তী": "DAMAYANTI",
    "দীপু": "DIPU",
    "পারভেজ": "PARVEZ",
    "লতিফ মিয়া": "LATIF MIA",
    "হারুন মিয়া": "HARUN MIA",
    "লাবনী": "LABONI",
    "সবিতা": "SABITA",
    "পরেশ": "PARESH",
    "শাকিল সাহেব": "SHAKIL SAHEB",
    "শিরিন": "SHIRIN",
    "স্বাতী": "SWATI",
    "রায়ন": "RAYON",
    "মোতালেব সাহেব": "MOTALEB SAHEB",
    "অনন্যা": "ANANNA",
    "আদিব": "ADIB",
    "রাফিক": "RAFIQUE",
    "সবুজ": "SABUJ",
    "রথি": "RATHI",
    "হালিমা": "HALIMA",
    "মমতা": "MAMATA",
    "রামসুন্দর বাবু": "RAMSUNDAR BABU",
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


import os
import re
import json
import time
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

# --- Protected terms: names and mythology ---
CHARACTER_NAMES = [
    "অনুপম",
    "কল্যাণী",
    "মামা",
    "শম্ভুনাথ সেন",
    "হরিশ",
    "বিনুদাদা",
    "শম্ভুনাথবাবুর উকিল-বন্ধু",
    "অন্নপূর্ণা",
    "গজানন",
    "মনু",
    "প্রজাপতি",
    "পঞ্চশর",
    "সতী",
    "শিব",
    "দময়ন্তী",
    "দীপু",
    "পারভেজ",
    "লতিফ মিয়া",
    "হারুন মিয়া",
    "লাবনী",
    "সবিতা",
    "পরেশ",
    "শাকিল সাহেব",
    "শিরিন",
    "স্বাতী",
    "রায়ন",
    "মোতালেব সাহেব",
    "অনন্যা",
    "আদিব",
    "রাফিক",
    "সবুজ",
    "রথি",
    "হালিমা",
    "মমতা",
    "রামসুন্দর বাবু",
    "অপরিচিতা",
]


def normalize_bengali_numerals(text):
    mapping = {
        "০": "0",
        "১": "1",
        "২": "2",
        "৩": "3",
        "৪": "4",
        "৫": "5",
        "৬": "6",
        "৭": "7",
        "৮": "8",
        "৯": "9",
    }
    for bn, en in mapping.items():
        text = text.replace(bn, en)
    return text


# Map for placeholder protection
PROTECTED_TERMS = {name: f"__{i}__" for i, name in enumerate(CHARACTER_NAMES)}


# --- Protection/Restoration ---
def protect_terms(text):
    for name, placeholder in PROTECTED_TERMS.items():
        text = text.replace(name, placeholder)
    return text


def restore_terms(text):
    for name, placeholder in PROTECTED_TERMS.items():
        text = text.replace(placeholder, name)
    return text


def safe_translate(sentence, retries=3, delay=1):
    sentence = normalize_bengali_numerals(sentence)
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
                translated = safe_translate(sentence)
                chunk["text_en"] = translated

        chunks.append(chunk)

    return chunks


# --- Main Execution ---
def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    all_chunks = []

    for section, path in INPUT_FILES.items():
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
