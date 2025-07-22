# app/embeddings/embedding_manager.py

from sentence_transformers import SentenceTransformer
import hashlib
import numpy as np
from typing import List
import os
import pickle


class EmbeddingManager:
    def __init__(
        self,
        model_name: str = "sentence-transformers/LaBSE",
        cache_dir: str = "data/vector_db/cache",
    ):
        print(f"[🔁] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _text_hash(self, text: str) -> str:
        """
        Create a hash from text for caching.
        """
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def embed(self, texts: List[str], use_cache: bool = True) -> List[np.ndarray]:
        """
        Generate embeddings in batch, with optional caching.

        Args:
            texts (List[str]): List of text chunks.
            use_cache (bool): Whether to use caching.

        Returns:
            List[np.ndarray]: Embeddings for each chunk.
        """
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for idx, text in enumerate(texts):
            cache_path = os.path.join(self.cache_dir, self._text_hash(text) + ".pkl")
            if use_cache and os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    embeddings.append(pickle.load(f))
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(idx)

        if uncached_texts:
            new_embeddings = self.model.encode(
                uncached_texts,
                batch_size=8,
                show_progress_bar=True,
                convert_to_numpy=True,
            )

            for idx, embed_vec in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embed_vec
                if use_cache:
                    hash_key = self._text_hash(texts[idx])
                    with open(
                        os.path.join(self.cache_dir, hash_key + ".pkl"), "wb"
                    ) as f:
                        pickle.dump(embed_vec, f)

        return embeddings
