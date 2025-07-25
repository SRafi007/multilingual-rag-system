"""Configuration settings for the Multilingual RAG System"""

import os
from pathlib import Path
from typing import Optional

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = DATA_DIR / "vectors"

# Ensure directories exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_DB_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class Config:
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TITLE: str = "Multilingual RAG API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "A multilingual RAG system for Bengali literature"

    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "llama2")
    OLLAMA_EMBEDDING_MODEL: str = os.getenv(
        "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
    )

    # Embedding Configuration
    EMBEDDING_MODEL: str = "distiluse-base-multilingual-cased"
    EMBEDDING_DIMENSION: int = 512
    BATCH_SIZE: int = 32

    # Vector Database Configuration
    VECTOR_DB_PATH: str = str(VECTOR_DB_DIR)
    COLLECTION_NAME: str = "bangla_book_rag"
    SIMILARITY_THRESHOLD: float = 0.5

    # Retrieval Configuration
    DEFAULT_TOP_K: int = 5
    MAX_CONTEXT_LENGTH: int = 2000
    MIN_SIMILARITY_SCORE: float = 0.3

    # Generation Configuration
    MAX_TOKENS: int = 500
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9

    # Memory Configuration
    MAX_SHORT_TERM_MEMORY: int = 10
    MAX_CONVERSATION_HISTORY: int = 3

    # Data Configuration
    JSON_DATA_PATH: str = str(PROCESSED_DATA_DIR / "structured_data.json")

    # Language Configuration
    SUPPORTED_LANGUAGES: list = ["bn", "en"]
    DEFAULT_LANGUAGE: str = "bn"

    # Evaluation Configuration
    TEST_CASES_PATH: str = str(PROCESSED_DATA_DIR / "test_cases.json")
    EVALUATION_OUTPUT_PATH: str = str(PROCESSED_DATA_DIR / "evaluation_results.json")

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = str(BASE_DIR / "logs" / "rag_system.log")


# Create logs directory
(BASE_DIR / "logs").mkdir(exist_ok=True)

# Test cases for evaluation
TEST_CASES = [
    {
        "query": "অনুপেমর ভাষায় সুপুরুষ কোকে বলা হয়েছে?",
        "expected_answer": "শুম্ভু নাথ",
        "language": "bn",
    },
    {
        "query": "কোকে অনুপেমর ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "expected_answer": "মামাকে",
        "language": "bn",
    },
    {
        "query": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
        "expected_answer": "১৫ বছর",
        "language": "bn",
    },
    {
        "query": "Who is referred to as a good person in Anupam's language?",
        "expected_answer": "Shumbhu Nath",
        "language": "en",
    },
    {
        "query": "What was Kalyani's actual age at the time of marriage?",
        "expected_answer": "15 years",
        "language": "en",
    },
]
