# retriever.py

import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(
        self,
        index_path="app/data/faiss_index/index.faiss",
        meta_path="app/data/faiss_index/metadata.jsonl",
    ):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.model = SentenceTransformer("sentence-transformers/LaBSE")

        print("📦 Loading FAISS index...")
        self.index = faiss.read_index(str(self.index_path))

        print("📚 Loading metadata...")
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        with open(self.meta_path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f]

    def search(self, query, top_k=5):
        print(f"🔍 Searching for: {query}")
        embedding = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        D, I = self.index.search(embedding, top_k)

        results = []
        for idx in I[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results
