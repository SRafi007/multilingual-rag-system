"""Embedding service for generating multilingual embeddings using Sentence Transformers"""

import numpy as np
from typing import List, Dict, Any, Union
import json
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch


class MultilingualEmbedder:
    def __init__(self, model_name: str = "distiluse-base-multilingual-cased"):
        """Initialize the multilingual embedder with Sentence Transformers"""
        self.model_name = model_name
        self.model = None
        self.embedding_dimension = None
        self._load_model()

    def _load_model(self):
        """Load the Sentence Transformer embedding model"""
        try:
            logger.info(f"Loading Sentence Transformer model: {self.model_name}")

            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            self.model = SentenceTransformer(self.model_name, device=device)

            # Get embedding dimension by encoding a test sentence
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            self.embedding_dimension = test_embedding.shape[0]

            logger.info(
                f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into embedding"""
        try:
            embedding = self.model.encode(
                text, convert_to_numpy=True, normalize_embeddings=True
            )
            return embedding
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise

    def encode_batch(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = True
    ) -> np.ndarray:
        """Encode a batch of texts into embeddings"""
        try:
            logger.info(f"Encoding {len(texts)} texts with batch size {batch_size}")

            # Use sentence-transformers' built-in batch processing
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            logger.info(f"Successfully encoded {len(texts)} texts")
            return embeddings

        except Exception as e:
            logger.error(f"Error in batch encoding: {e}")
            raise

    def compute_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            # If embeddings are already normalized, we can just use dot product
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find most similar embeddings to query"""
        try:
            # Ensure query embedding is normalized
            if np.linalg.norm(query_embedding) != 1.0:
                query_embedding = query_embedding / np.linalg.norm(query_embedding)

            # Ensure candidate embeddings are normalized
            candidate_norms = np.linalg.norm(
                candidate_embeddings, axis=1, keepdims=True
            )
            normalized_candidates = candidate_embeddings / candidate_norms

            similarities = np.dot(normalized_candidates, query_embedding)
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                results.append(
                    {"index": int(idx), "similarity": float(similarities[idx])}
                )
            return results
        except Exception as e:
            logger.error(f"Error finding similar embeddings: {e}")
            return []

    def get_sentence_embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.embedding_dimension


class EmbeddingProcessor:
    def __init__(self, embedder: MultilingualEmbedder):
        """Initialize embedding processor"""
        self.embedder = embedder

    def process_json_data(
        self, json_data: List[Dict[str, Any]], batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """Process JSON data and add embeddings"""
        try:
            logger.info(f"Processing {len(json_data)} items for embedding")

            # Extract texts for embedding
            texts = []
            for item in json_data:
                content = item.get("content", "")
                # For MCQ, combine question with options for better context
                if item.get("content_type") == "mcq" and "question" in item:
                    question = item["question"]
                    options = item.get("options", {})
                    combined_text = f"{question} {' '.join(options.values())}"
                    texts.append(combined_text)
                else:
                    texts.append(content)

            # Generate embeddings in batches
            logger.info("Generating embeddings...")
            embeddings = self.embedder.encode_batch(texts, batch_size=batch_size)

            # Add embeddings to data
            processed_data = []
            for i, item in enumerate(json_data):
                processed_item = item.copy()
                processed_item["embedding"] = embeddings[i].tolist()
                processed_item["embedding_model"] = self.embedder.model_name
                processed_data.append(processed_item)

            logger.info("Successfully added embeddings to all items")
            return processed_data

        except Exception as e:
            logger.error(f"Error processing JSON data: {e}")
            raise

    def save_embeddings(self, processed_data: List[Dict[str, Any]], output_path: str):
        """Save processed data with embeddings"""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Embeddings saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            raise

    def load_embeddings(self, input_path: str) -> List[Dict[str, Any]]:
        """Load processed data with embeddings"""
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} items with embeddings from {input_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise

    def extract_embeddings_matrix(
        self, processed_data: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Extract embeddings as a numpy matrix"""
        try:
            embeddings = []
            for item in processed_data:
                if "embedding" in item:
                    embeddings.append(item["embedding"])
                else:
                    logger.warning(
                        f"No embedding found for item {item.get('id', 'unknown')}"
                    )

            if not embeddings:
                raise ValueError("No embeddings found in data")

            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error extracting embeddings matrix: {e}")
            raise

    def validate_embeddings(
        self, processed_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate embeddings in processed data"""
        try:
            validation_results = {
                "total_items": len(processed_data),
                "items_with_embeddings": 0,
                "embedding_dimension": None,
                "average_embedding_norm": 0.0,
                "embedding_model": None,
            }

            embedding_norms = []
            for item in processed_data:
                if "embedding" in item:
                    validation_results["items_with_embeddings"] += 1
                    embedding = np.array(item["embedding"])

                    if validation_results["embedding_dimension"] is None:
                        validation_results["embedding_dimension"] = len(embedding)

                    if validation_results["embedding_model"] is None:
                        validation_results["embedding_model"] = item.get(
                            "embedding_model", "unknown"
                        )

                    norm = np.linalg.norm(embedding)
                    embedding_norms.append(norm)

            if embedding_norms:
                validation_results["average_embedding_norm"] = float(
                    np.mean(embedding_norms)
                )

            validation_results["embedding_coverage"] = (
                validation_results["items_with_embeddings"]
                / validation_results["total_items"]
            ) * 100

            logger.info(f"Embedding validation: {validation_results}")
            return validation_results

        except Exception as e:
            logger.error(f"Error validating embeddings: {e}")
            raise


def main():
    """Main function to process embeddings"""
    from app.config.config import Config

    # Initialize embedder
    embedder = MultilingualEmbedder(Config.EMBEDDING_MODEL)
    processor = EmbeddingProcessor(embedder)

    # Input and output paths
    input_path = Config.JSON_DATA_PATH.replace(".json", "_enhanced.json")
    output_path = Config.JSON_DATA_PATH.replace(".json", "_with_embeddings.json")

    if not Path(input_path).exists():
        logger.error(f"Enhanced data file not found: {input_path}")
        logger.info("Please run data enhancement first")
        return

    try:
        # Load enhanced data
        with open(input_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        logger.info(f"Loaded {len(json_data)} enhanced items")

        # Process embeddings
        processed_data = processor.process_json_data(
            json_data, batch_size=Config.BATCH_SIZE
        )

        # Save processed data
        processor.save_embeddings(processed_data, output_path)

        # Validate embeddings
        validation_results = processor.validate_embeddings(processed_data)

        # Save validation results
        validation_path = output_path.replace(".json", "_validation.json")
        with open(validation_path, "w", encoding="utf-8") as f:
            json.dump(validation_results, f, ensure_ascii=False, indent=2)

        logger.info("Embedding processing completed successfully!")

    except Exception as e:
        logger.error(f"Error in embedding processing: {e}")
        raise


if __name__ == "__main__":
    main()
