# embed.py

import os
import json
from pathlib import Path
from tqdm import tqdm
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_chunks(chunks_dir):
    """
    Load all JSONL chunks from the chunks directory.
    Returns: list of dicts {text, source, source_type, chunk_index}
    """
    chunks = []
    for file in Path(chunks_dir).glob("*.jsonl"):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line.strip())
                if chunk.get("text"):
                    chunks.append(chunk)
    return chunks


def embed_texts(texts, model_name="sentence-transformers/LaBSE"):
    """
    Embed texts using multilingual LaBSE model.
    Returns: np.array of embeddings
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    )
    return embeddings


def save_faiss_index(embeddings, metadata, output_dir="output/faiss_index"):
    """
    Save FAISS index and metadata.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, str(Path(output_dir) / "index.faiss"))

    # Save metadata
    with open(Path(output_dir) / "metadata.jsonl", "w", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ FAISS index saved to {output_dir}/index.faiss")
    print(f"✅ Metadata saved to {output_dir}/metadata.jsonl")


def main():
    chunks_dir = "app/data/chunks"
    output_dir = "app/data/faiss_index"

    print("📄 Loading text chunks...")
    chunk_data = load_chunks(chunks_dir)
    texts = [item["text"] for item in chunk_data]

    print("🔢 Generating embeddings (LaBSE)...")
    embeddings = embed_texts(texts)

    print("💾 Saving index and metadata...")
    save_faiss_index(embeddings, chunk_data, output_dir)


if __name__ == "__main__":
    main()
