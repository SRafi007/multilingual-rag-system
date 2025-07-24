"""
Simple processor for generating embedding-ready text from extracted content
Uses Gemini 2.5-flash to understand content and format appropriately
"""

import google.generativeai as genai
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ProcessorConfig:
    """Configuration for the content processor"""

    gemini_api_key: str
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.1
    max_output_tokens: int = 2048


class ContentProcessor:
    """Processes extracted content for embedding generation"""

    def __init__(self, config: ProcessorConfig):
        self.config = config
        genai.configure(api_key=config.gemini_api_key)
        self.model = genai.GenerativeModel(config.model_name)

    def process_content(self, content_data: Dict[str, Any]) -> str:
        """
        Process content based on type and return embedding-ready text

        Args:
            content_data: Dictionary with content_type, structured_data, embedding_text

        Returns:
            str: Processed text ready for embedding
        """
        content_type = content_data.get("content_type", "")
        structured_data = content_data.get("structured_data", {})
        embedding_text = content_data.get("embedding_text", "")

        # Create prompt based on content type
        prompt = self._create_prompt(content_type, structured_data, embedding_text)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                ),
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error processing content: {e}")
            # Fallback to embedding_text if API fails
            return embedding_text

    def _create_prompt(
        self, content_type: str, structured_data: Dict, embedding_text: str
    ) -> str:
        """Create appropriate prompt based on content type"""

        base_instruction = """
        You are processing multilingual educational content (Bangla/English) for embedding generation.
        Your task is to create clean, embedding-ready text that preserves all important information.
        
        Guidelines:
        - For MCQs: Include question with correct answer (not all options)
        - For tables: Convert to readable text format
        - For stories/narratives: Keep complete text
        - For learning outcomes: Keep as structured points
        - For vocabulary: Include word and meaning
        - Maintain original language (Bangla/English mixed)
        - Don't manipulate original content, just format for embedding
        - Priority: Use embedding_text as base, enhance with structured_data
        - REMOVE headers like "HSC 26", "অনলাইন ব্যাচ", "10 MINUTE SCHOOL", course names, batch info
        - Focus only on educational content: questions, answers, learning outcomes, stories, vocabulary
        """

        if content_type == "mcq":
            return f"""{base_instruction}
            
            Content Type: Multiple Choice Questions
            Task: For each question, include the question and correct answer only (not all options).
            
            Embedding Text: {embedding_text}
            Structured Data: {json.dumps(structured_data, ensure_ascii=False, indent=2)}
            
            Output clean embedding-ready text:"""

        elif content_type == "mixed":
            return f"""{base_instruction}
            
            Content Type: Mixed Content (MCQs, Learning Outcomes, Tables, etc.)
            Task: Process each component appropriately - questions with answers, outcomes as points, tables as text.
            IMPORTANT: Remove course headers, batch names, school names. Keep only educational content.
            
            Embedding Text: {embedding_text}
            Structured Data: {json.dumps(structured_data, ensure_ascii=False, indent=2)}
            
            Output clean embedding-ready text without headers:"""

        elif content_type == "table":
            return f"""{base_instruction}
            
            Content Type: Table
            Task: Convert table data to readable text format.
            
            Embedding Text: {embedding_text}
            Structured Data: {json.dumps(structured_data, ensure_ascii=False, indent=2)}
            
            Output clean embedding-ready text:"""

        elif content_type == "learning_outcome":
            return f"""{base_instruction}
            
            Content Type: Learning Outcomes
            Task: Keep learning outcomes as clear, structured points.
            
            Embedding Text: {embedding_text}
            Structured Data: {json.dumps(structured_data, ensure_ascii=False, indent=2)}
            
            Output clean embedding-ready text:"""

        elif content_type in ["narrative", "literary_prose", "poetry"]:
            return f"""{base_instruction}
            
            Content Type: {content_type.title()}
            Task: Keep complete text content for literary/narrative material.
            
            Embedding Text: {embedding_text}
            Structured Data: {json.dumps(structured_data, ensure_ascii=False, indent=2)}
            
            Output clean embedding-ready text:"""

        elif content_type == "vocabulary":
            return f"""{base_instruction}
            
            Content Type: Vocabulary
            Task: Include word and meaning pairs clearly.
            
            Embedding Text: {embedding_text}
            Structured Data: {json.dumps(structured_data, ensure_ascii=False, indent=2)}
            
            Output clean embedding-ready text:"""

        else:
            # Generic processing for other content types
            return f"""{base_instruction}
            
            Content Type: {content_type}
            Task: Process content appropriately based on its nature.
            
            Embedding Text: {embedding_text}
            Structured Data: {json.dumps(structured_data, ensure_ascii=False, indent=2)}
            
            Output clean embedding-ready text:"""


def process_single_content(content_data: Dict[str, Any], gemini_api_key: str) -> str:
    """
    Convenience function to process a single content item

    Args:
        content_data: Content dictionary
        gemini_api_key: Your Gemini API key

    Returns:
        str: Processed embedding-ready text
    """
    config = ProcessorConfig(gemini_api_key=gemini_api_key)
    processor = ContentProcessor(config)
    return processor.process_content(content_data)


def process_multiple_contents(contents_list: list, gemini_api_key: str) -> list:
    """
    Process multiple content items

    Args:
        contents_list: List of content dictionaries
        gemini_api_key: Your Gemini API key

    Returns:
        list: List of processed texts
    """
    config = ProcessorConfig(gemini_api_key=gemini_api_key)
    processor = ContentProcessor(config)

    results = []
    for content in contents_list:
        processed_text = processor.process_content(content)
        results.append(processed_text)

    return results


def process_json_file(
    json_file_path: str, gemini_api_key: str, output_file_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process entire JSON file with pages structure

    Args:
        json_file_path: Path to input JSON file
        gemini_api_key: Your Gemini API key
        output_file_path: Optional path to save processed results

    Returns:
        dict: Processed results with original structure plus processed_text field
    """
    config = ProcessorConfig(gemini_api_key=gemini_api_key)
    processor = ContentProcessor(config)

    # Load JSON file
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Process each page
    processed_data = {"pages": []}

    for page in data.get("pages", []):
        print(
            f"Processing page {page.get('page_number', 'unknown')} - {page.get('content_type', 'unknown')}"
        )

        # Extract required fields
        content_data = {
            "content_type": page.get("content_type", ""),
            "structured_data": page.get("structured_data", {}),
            "embedding_text": page.get("embedding_text", ""),
        }

        # Process content
        processed_text = processor.process_content(content_data)

        # Create processed page data
        processed_page = {
            "page_number": page.get("page_number"),
            "content_type": page.get("content_type"),
            "title": page.get("title"),
            "language": page.get("language"),
            "confidence_score": page.get("confidence_score"),
            "original_embedding_text": page.get("embedding_text", ""),
            "processed_text": processed_text,
            "structured_data": page.get("structured_data", {}),  # Keep for reference
        }

        processed_data["pages"].append(processed_page)

    # Save to file if output path provided
    if output_file_path:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        print(f"Processed data saved to: {output_file_path}")

    return processed_data


def extract_processed_texts(processed_data: Dict[str, Any]) -> list:
    """
    Extract only the processed texts from processed data

    Args:
        processed_data: Result from process_json_file

    Returns:
        list: List of processed texts ready for embedding
    """
    return [page["processed_text"] for page in processed_data.get("pages", [])]


def process_and_save_embeddings(
    json_file_path: str, gemini_api_key: str, output_dir: str = "output"
) -> tuple:
    """
    Complete pipeline: process JSON file and prepare texts for embeddings

    Args:
        json_file_path: Path to input JSON file
        gemini_api_key: Your Gemini API key
        output_dir: Directory to save outputs

    Returns:
        tuple: (processed_data, embedding_texts)
    """
    import os

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process JSON file
    processed_file_path = os.path.join(output_dir, "processed_content.json")
    processed_data = process_json_file(
        json_file_path, gemini_api_key, processed_file_path
    )

    # Extract embedding texts
    embedding_texts = extract_processed_texts(processed_data)

    # Save embedding texts
    embedding_texts_file = os.path.join(output_dir, "embedding_texts.json")
    with open(embedding_texts_file, "w", encoding="utf-8") as f:
        json.dump(embedding_texts, f, ensure_ascii=False, indent=2)

    print(f"Total pages processed: {len(embedding_texts)}")
    print(f"Embedding texts saved to: {embedding_texts_file}")

    return processed_data, embedding_texts


# Example usage
if __name__ == "__main__":
    # Replace with your actual API key
    API_KEY = "AIzaSyBShjOlBmWFq4FfgxFg37b0Yk_fQbHKj5s"

    # Method 1: Process entire JSON file
    json_file_path = "data/processed/extracted_data.json"  # Your main JSON file

    try:
        # Complete pipeline
        processed_data, embedding_texts = process_and_save_embeddings(
            json_file_path=json_file_path,
            gemini_api_key=API_KEY,
            output_dir="data/processed",
        )

        # Show sample results
        print("\nSample processed texts:")
        for i, text in enumerate(embedding_texts[:3]):  # Show first 3
            print(f"\nPage {i+1}:")
            print(text[:200] + "..." if len(text) > 200 else text)

    except FileNotFoundError:
        print(f"File {json_file_path} not found. Using sample data instead.")

        # Method 2: Process sample content (fallback)
        sample_content = {
            "content_type": "mixed",
            "structured_data": {
                "learning_outcomes": [
                    {
                        "outcome": "নিম্নবিত্ত ব্যক্তির হঠাৎ বিত্তশালী হয়ে ওঠার ফলে সমাজে পরিচয় সংকট সম্পর্কে ধারণা লাভ করবে।",
                        "context": "সাহিত্যিক পাঠের মাধ্যমে সামাজিক ধারণা অর্জন",
                        "language": "bangla",
                    }
                ],
                "mcqs": [
                    {
                        "question_number": 1,
                        "question": "অনুপমের বাবা কী করে জীবিকা নির্বাহ করতেন?",
                        "options": {
                            "ক": "ডাক্তারি",
                            "খ": "ওকালতি",
                            "গ": "মাস্টারি",
                            "ঘ": "ব্যবসা",
                        },
                    }
                ],
            },
            "embedding_text": "HSC 26 অনলাইন ব্যাচ বাংলা শিখনফল নিম্নবিত্ত ব্যক্তির হঠাৎ বিত্তশালী হয়ে ওঠার ফলে সমাজে পরিচয় সংকট সম্পর্কে ধারণা লাভ করবে। প্রাক-মূল্যায়ন ১। অনুপমের বাবা কী করে জীবিকা নির্বাহ করতেন? ক) ডাক্তারি খ) ওকালতি গ) মাস্টারি ঘ) ব্যবসা",
        }

        # Process single content
        result = process_single_content(sample_content, API_KEY)
        print("Processed Text:")
        print(result)
