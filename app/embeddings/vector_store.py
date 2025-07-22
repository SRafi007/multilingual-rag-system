# app/embeddings/vector_store.py
import faiss
import numpy as np
import os
import json
from pathlib import Path
from typing import List, Dict


class VectorStore:
    def __init__(
        self,
        index_path: str = "data/vector_db/faiss_index.index",
        metadata_path: str = "data/vector_db/metadata.json",
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index = None
        self.metadata = []

    def build_index(self, embeddings: List[np.ndarray], metadatas: List[Dict]):
        """
        Build and save a FAISS index from embeddings and store associated metadata.
        """
        dim = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        vectors = np.array(embeddings).astype("float32")

        index.add(vectors)
        self.index = index
        self.metadata = metadatas

        self._save_index()
        self._save_metadata()

        print(f"[✅] FAISS index built with {index.ntotal} vectors.")

    def _save_index(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

    def _save_metadata(self):
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self):
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Index or metadata file not found.")

        self.index = faiss.read_index(str(self.index_path))
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        print(f"[📦] Loaded FAISS index with {self.index.ntotal} vectors.")

    def query(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Perform a similarity search using the query vector.
        Returns top_k results with metadata.
        """
        if self.index is None:
            self.load()

        query_vector = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx]
                result["score"] = float(dist)
                results.append(result)

        return results

    def add_documents(self, embeddings: List[np.ndarray], metadatas: List[Dict]):
        """
        Add new embeddings and associated metadata to the existing index.
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() or build_index() first.")

        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.metadata.extend(metadatas)

        self._save_index()
        self._save_metadata()
        print(f"[➕] Added {len(vectors)} new vectors.")

    def delete_by_filter(self, key: str, value: str):
        """
        Delete vectors from the index where metadata[key] == value.
        NOTE: FAISS does not support true deletion — we rebuild the index.
        """
        print(f"[⚠️] Rebuilding index to remove metadata where {key} == '{value}'")

        # Filter out matching metadata
        new_metadata = []
        new_embeddings = []

        for i, meta in enumerate(self.metadata):
            if meta.get("metadata", {}).get(key) != value:
                new_metadata.append(meta)
                vec = self.index.reconstruct(i)
                new_embeddings.append(vec)

        # Rebuild index
        self.build_index(new_embeddings, new_metadata)

    def query_with_filter(
        self, query_vector: np.ndarray, top_k: int = 5, filters: Dict = {}
    ) -> List[Dict]:
        """
        Perform a similarity search with optional metadata filters.
        """
        all_results = self.query(
            query_vector, top_k=top_k + 20
        )  # oversample, filter later
        if not filters:
            return all_results[:top_k]

        def match(meta: Dict) -> bool:
            return all(meta.get("metadata", {}).get(k) == v for k, v in filters.items())

        filtered = [r for r in all_results if match(r)]
        return filtered[:top_k]
