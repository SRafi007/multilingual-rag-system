from app.embeddings.embedding_manager import EmbeddingManager
from app.embeddings.vector_store import VectorStore
from app.data.chunking import Chunker
import json
from pathlib import Path


def main():
    chunks_file = Path("data/processed/chunks/chunks.json")
    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {"text": chunk["text"], "metadata": chunk["metadata"]} for chunk in chunks
    ]

    # Generate embeddings
    embedder = EmbeddingManager()
    vectors = embedder.embed(texts)

    # Build vector store
    store = VectorStore()
    store.build_index(vectors, metadatas)


if __name__ == "__main__":
    main()
