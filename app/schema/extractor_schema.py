"""
Data models and type definitions for OCR Gemini Extractor
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional


class ContentType(Enum):
    """Enumeration of different content types that can be extracted"""

    MCQ = "mcq"
    NARRATIVE = "narrative"
    VOCABULARY = "vocabulary"
    INSTRUCTION = "instruction"
    POETRY = "poetry"
    DIALOGUE = "dialogue"
    TABLE = "table"
    MIXED = "mixed"


@dataclass
class MCQQuestion:
    """Data model for Multiple Choice Questions"""

    question: str
    options: Dict[str, str]
    correct_answer: Optional[str] = None
    explanation: Optional[str] = None
    question_number: Optional[int] = None


@dataclass
class VocabularyEntry:
    """Data model for vocabulary entries"""

    word: str
    meaning: str
    explanation: Optional[str] = None
    language: str = "bangla"
    pronunciation: Optional[str] = None


@dataclass
class ExtractedContent:
    """Main data model for extracted content from a page"""

    content_type: ContentType
    title: Optional[str]
    raw_text: str
    cleaned_text: str
    structured_data: Optional[Dict[str, Any]]
    language: str
    confidence_score: float
    page_number: int
    bounding_boxes: Optional[List[Dict]] = None
    embedding_text: str = ""


@dataclass
class BoundingBox:
    """Data model for text bounding boxes"""

    word: str
    confidence: int
    x: int
    y: int
    width: int
    height: int


@dataclass
class ProcessingStatistics:
    """Statistics for processing results"""

    mcq_questions: int = 0
    vocabulary_entries: int = 0
    narrative_sections: int = 0
    high_confidence_pages: int = 0
    avg_confidence: float = 0.0
    total_pages: int = 0
