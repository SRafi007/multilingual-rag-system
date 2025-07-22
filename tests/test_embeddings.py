from app.embeddings.embedding_manager import EmbeddingManager
import json
from pathlib import Path


def main():
    chunk_path = Path("data/processed/chunks/chunks.json")
    chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
    texts = [chunk["text"] for chunk in chunks]

    embedder = EmbeddingManager()
    vectors = embedder.embed(texts)

    print(f"[✅] Generated {len(vectors)} embeddings.")
    print(f"[🔢] First vector shape: {vectors[0].shape}")
    print(f"[📊] First 10 dims: {vectors[0][:10]}")


if __name__ == "__main__":
    main()
