# chunking.py

import json
from pathlib import Path
import unicodedata


def intelligent_chunk_bangla_text(text, chunk_size=300, overlap=50):
    """
    Break Bangla text into semantically coherent chunks using '।' as sentence delimiter.
    """
    text = unicodedata.normalize("NFC", text)
    sentences = [s.strip() for s in text.split("।") if s.strip()]
    chunks = []
    current_chunk = []

    for sentence in sentences:
        sentence += "।"  # Reattach punctuation
        joined = "".join(current_chunk + [sentence])

        if len(joined) <= chunk_size:
            current_chunk.append(sentence)
        else:
            chunks.append("".join(current_chunk).strip())
            overlap_sentences = current_chunk[-1:] if overlap else []
            current_chunk = overlap_sentences + [sentence]

    if current_chunk:
        chunks.append("".join(current_chunk).strip())

    return chunks


def chunk_file(input_path: Path, output_path: Path, source_type: str):
    print(f"🔪 Chunking: {input_path.name}")
    text = input_path.read_text(encoding="utf-8")
    chunks = intelligent_chunk_bangla_text(text)

    output_data = [
        {
            "text": chunk,
            "source": input_path.name,
            "source_type": source_type,
            "chunk_index": idx,
        }
        for idx, chunk in enumerate(chunks)
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ {len(chunks)} chunks written to: {output_path}")


def run_chunking(
    input_dir="app/data/resolved",
    cleaned_dir="app/data/processed",
    output_dir="app/data/chunks",
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Each file and its source_type
    files = {
        "resolved_story.txt": "story",
        "cleaned_glossary.txt": "glossary",
        "cleaned_author.txt": "author",
        "cleaned_intro.txt": "lesson_intro",
        "cleaned_knowledge.txt": "knowledge",
    }

    for filename, source_type in files.items():
        input_path = (
            Path(input_dir if "resolved" in filename else cleaned_dir) / filename
        )
        output_path = Path(output_dir) / f"{filename.replace('.txt', '')}.jsonl"

        if input_path.exists():
            chunk_file(input_path, output_path, source_type)
        else:
            print(f"⚠️ Skipped: {filename} (not found)")


if __name__ == "__main__":
    run_chunking()
