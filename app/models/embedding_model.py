"""Simple Ollama embedding model"""

import requests
import numpy as np
from typing import List, Union
from loguru import logger


class OllamaEmbedder:
    def __init__(
        self,
        model_name: str = "distiluse-base-multilingual-cased",
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.embedding_dimension = None
        self._get_dimension()

    def _get_dimension(self):
        """Get embedding dimension with test text"""
        embedding = self._embed_single("test")
        self.embedding_dimension = len(embedding)
        logger.info(f"Model: {self.model_name}, Dimension: {self.embedding_dimension}")

    def _embed_single(self, text: str) -> List[float]:
        """Generate single embedding"""
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model_name, "prompt": text},
            timeout=60,
        )
        return response.json()["embedding"]

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text(s) into embeddings"""
        if isinstance(texts, str):
            return np.array([self._embed_single(texts)])

        embeddings = [self._embed_single(text) for text in texts]
        return np.array(embeddings)

    def get_sentence_embedding_dimension(self) -> int:
        """Compatibility with sentence-transformers"""
        return self.embedding_dimension
