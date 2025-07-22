from app.data.chunking import Chunker
from pathlib import Path
import json


def main():
    input_path = Path("data/processed/preprocessed_text.txt")
    output_path = Path("data/processed/chunks/chunks.json")

    text = input_path.read_text(encoding="utf-8")
    chunker = Chunker()
    chunks = chunker.chunk_text(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[✅] {len(chunks)} chunks created and saved to {output_path}")
    print(f"[📄] Sample chunk:\n{chunks[0]['text'][:300]}...")


if __name__ == "__main__":
    main()
