import json
import re
import google.generativeai as genai
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import time
import os
from dotenv import load_dotenv
from app.config.settings import (
    GEMINI_API_KEY,
    GEMINI_API_KEY_BACKUP,
    OUTPUT_DIR,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EducationalContentProcessor:
    def __init__(
        self, api_key: Optional[str] = None, backup_api_key: Optional[str] = None
    ):
        """
        Initialize Gemini processor with primary and optional backup API key.
        """
        self.api_key = GEMINI_API_KEY_BACKUP
        self.backup_api_key = GEMINI_API_KEY

        if not self.api_key:
            raise ValueError("Primary GEMINI_API_KEY is required")

        # Initialize primary model
        genai.configure(api_key=self.api_key)
        self.primary_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={
                "temperature": 0.1,
                "top_p": 0.8,
                "top_k": 20,
                "max_output_tokens": 8192,
            },
        )

        # Backup model will be initialized on-demand when needed
        self.backup_model = None

    def clean_json_response(self, response_text: str) -> str:
        """
        Clean and extract valid JSON from Gemini response

        Args:
            response_text (str): Raw response from Gemini

        Returns:
            str: Cleaned JSON string
        """
        # Remove markdown code blocks if present
        response_text = re.sub(r"```json\s*", "", response_text)
        response_text = re.sub(r"```\s*$", "", response_text)

        # Remove any leading/trailing whitespace
        response_text = response_text.strip()

        # Find JSON array pattern
        json_patterns = [
            r"\[[\s\S]*\]",  # Array pattern
            r"\{[\s\S]*\}",  # Object pattern
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                # Clean up common issues
                json_str = re.sub(
                    r",\s*}", "}", json_str
                )  # Remove trailing commas before }
                json_str = re.sub(
                    r",\s*]", "]", json_str
                )  # Remove trailing commas before ]
                return json_str

        return response_text

    def validate_and_fix_json(self, json_str: str) -> List[Dict[str, Any]]:
        """
        Validate and attempt to fix JSON string

        Args:
            json_str (str): JSON string to validate

        Returns:
            List[Dict]: Parsed JSON or empty list if failed
        """
        try:
            # First attempt: direct parsing
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                logger.warning(f"Unexpected JSON structure: {type(data)}")
                return []

        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parsing failed: {e}")

            # Second attempt: fix common issues
            try:
                # Fix common JSON issues
                fixed_json = json_str

                # Fix unescaped quotes in strings
                fixed_json = re.sub(r'(?<!\\)"(?![,}\]\s])', '\\"', fixed_json)

                # Fix trailing commas
                fixed_json = re.sub(r",(\s*[}\]])", r"\1", fixed_json)

                # Try parsing again
                data = json.loads(fixed_json)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data]
                else:
                    return []

            except json.JSONDecodeError as e2:
                logger.error(f"JSON fixing failed: {e2}")
                logger.error(f"Problematic JSON: {json_str[:500]}...")
                return []

    def remove_noise(self, text: str) -> str:
        """
        Remove noise patterns from text content

        Args:
            text (str): Input text with potential noise

        Returns:
            str: Cleaned text
        """
        noise_patterns = [
            r"10 MINUTE SCHOOL HSC \d+",
            r"MINUTE\s*SCHOOL",
            r"HSC \d+",
            r"অনলাইন ব্যাচ বাংলা\s*[\•·]\s*ইংরেজি\s*[\•·]\s*আইসিটি",
            r"বাংলা ১ম পত্র অনলাইন ব্যাচ",
            r"সম্পর্কিত যেকোনো জিজ্ঞাসায়,?\s*কল করো\s*📞?\s*16910",
            r"\(For any inquiries related to the online batch, call 📞 16910\)",
            r"📞\s*16910",
            r"কল করো\s*16910",
            r"Call 16910 for any inquiries",
            # Add more patterns as needed
        ]

        cleaned_text = text
        for pattern in noise_patterns:
            cleaned_text = re.sub(
                pattern, "", cleaned_text, flags=re.IGNORECASE | re.MULTILINE
            )

        # Remove excessive whitespace
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

        return cleaned_text

    def _try_generate(
        self, model, prompt: str, max_retries=3, base_delay=1
    ) -> Optional[str]:
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                if response and response.text.strip():
                    return response.text.strip()
                logger.warning(f"Empty response (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
            time.sleep(base_delay * (2**attempt))
        return None

    def create_prompt(self, structured_data: Dict[str, Any]) -> str:
        """
        Create a detailed prompt for Gemini API

        Args:
            structured_data (Dict): The structured data to process

        Returns:
            str: Formatted prompt
        """
        return f"""You are an expert educational content processor. Your task is to convert structured educational data into clean JSON chunks.

IMPORTANT INSTRUCTIONS:
1. Return ONLY a valid JSON array, no additional text, explanations, or markdown formatting
2. Each chunk must have "content_type" as the first key
3. For MCQs, use this exact format: {{"content_type": "mcq","question_number":"n", "question": "...", "options": {{}}, "correct_answer": "..."}}
4. For other content types, use: {{"content_type": "content_type_name", "content": "..."}}
5. Remove any noise text like promotional content, phone numbers, or course advertisements
6. Identify if the content is or not educational content

CONTENT TYPES TO RECOGNIZE:
- mcq: Multiple Choice Questions
- learning_outcome: Learning objectives/outcomes
- vocabulary: Word definitions
- narrative: Story passages
- grammar: Grammar rules/exercises
- instruction: Instructional content
- short_answer: Short answer questions
- comprehension: Reading comprehension
- summary: Summary content
- table: Tabular data
- mixed: Mixed content

INPUT DATA:
{json.dumps(structured_data, ensure_ascii=False, indent=2)}

OUTPUT (JSON array only):"""

    def process_structured_data(
        self, structured_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        prompt = self.create_prompt(structured_data)

        # Try primary model first
        response_text = self._try_generate(self.primary_model, prompt)

        # If primary fails and backup API key is available
        if not response_text and self.backup_api_key:
            logger.warning("Primary model failed. Switching to backup API key...")

            try:
                genai.configure(api_key=self.backup_api_key)
                if not self.backup_model:
                    self.backup_model = genai.GenerativeModel(
                        model_name="gemini-2.0-pro",  # Or same model if needed
                        generation_config={
                            "temperature": 0.1,
                            "top_p": 0.8,
                            "top_k": 20,
                            "max_output_tokens": 8192,
                        },
                    )
                response_text = self._try_generate(self.backup_model, prompt)
            except Exception as backup_err:
                logger.error(
                    f"Backup model failed to initialize or generate: {backup_err}"
                )

        if not response_text:
            logger.error("All model attempts failed.")
            return []

        try:
            cleaned_json = self.clean_json_response(response_text)
            chunks = self.validate_and_fix_json(cleaned_json)

            for chunk in chunks:
                if "content" in chunk:
                    chunk["content"] = self.remove_noise(chunk["content"])
                if "question" in chunk:
                    chunk["question"] = self.remove_noise(chunk["question"])

            logger.info(f"Successfully processed {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Failed to clean/process response: {e}")
            return []

    def fallback_processing(
        self, structured_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Fallback method to process structured data without API

        Args:
            structured_data (Dict): The structured data

        Returns:
            List[Dict]: Manually processed chunks
        """
        chunks = []

        try:
            # Process learning outcomes
            if "learning_outcomes" in structured_data:
                for outcome in structured_data["learning_outcomes"]:
                    chunks.append(
                        {
                            "content_type": "learning_outcome",
                            "content": self.remove_noise(outcome.get("outcome", "")),
                        }
                    )

            # Process MCQs
            if "mcqs" in structured_data:
                for mcq in structured_data["mcqs"]:
                    chunk = {
                        "content_type": "mcq",
                        "question_number": mcq.get(
                            "question_number", 1
                        ),  # Add this line
                        "question": self.remove_noise(mcq.get("question", "")),
                        "options": mcq.get("options", {}),
                        "correct_answer": mcq.get(
                            "answer", mcq.get("correct_answer", "")
                        ),
                    }
                    chunks.append(chunk)

            # Process instructions
            if "instructions" in structured_data:
                for instruction in structured_data["instructions"]:
                    content = f"{instruction.get('instruction', '')} {instruction.get('details', '')}".strip()
                    chunks.append(
                        {
                            "content_type": "instruction",
                            "content": self.remove_noise(content),
                        }
                    )

            # Process other content types
            content_mappings = {
                "vocabulary": "vocabulary",
                "narratives": "narrative",
                "short_answers": "short_answer",
                "grammar_rules": "grammar",
                "summaries": "summary",
                "comprehension": "comprehension",
            }

            for key, content_type in content_mappings.items():
                if key in structured_data:
                    for item in structured_data[key]:
                        if isinstance(item, dict):
                            # Extract main content field
                            content_fields = [
                                "content",
                                "text",
                                "passage",
                                "rule",
                                "summary",
                            ]
                            content = ""
                            for field in content_fields:
                                if field in item:
                                    content = item[field]
                                    break

                            if not content and "word" in item and "meaning" in item:
                                content = f"{item['word']}: {item['meaning']}"

                            if content:
                                chunks.append(
                                    {
                                        "content_type": content_type,
                                        "content": self.remove_noise(str(content)),
                                    }
                                )

            logger.info(f"Fallback processing created {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Fallback processing failed: {e}")
            return []

    def process_page(self, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a single page from your JSON data

        Args:
            page_data (Dict): Single page data from your JSON

        Returns:
            List[Dict]: List of processed chunks for the page
        """
        if "structured_data" not in page_data:
            logger.warning(
                f"No structured_data found in page {page_data.get('page_number', 'unknown')}"
            )
            return []

        structured_data = page_data["structured_data"]

        # Try API processing first
        chunks = self.process_structured_data(structured_data)

        # If API processing fails, use fallback
        if not chunks:
            logger.info(
                f"Using fallback processing for page {page_data.get('page_number')}"
            )
            chunks = self.fallback_processing(structured_data)

        # Add page metadata to each chunk
        for chunk in chunks:
            chunk["page_number"] = page_data.get("page_number")
            chunk["language"] = page_data.get("language")
        #  chunk["quality_score"] = page_data.get( "quality_score", page_data.get("confidence_score"))

        return chunks

    def process_json_file(
        self,
        input_file_path: str,
        output_file_path: str,
        delay_between_pages: float = 1.0,
    ) -> None:
        """
        Process entire JSON file and save processed chunks

        Args:
            input_file_path (str): Path to input JSON file
            output_file_path (str): Path to output JSON file
            delay_between_pages (float): Delay between processing pages (seconds)
        """
        try:
            # Load input JSON
            logger.info(f"Loading input file: {input_file_path}")
            with open(input_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            all_chunks = []
            pages = data.get("pages", [])

            logger.info(f"Processing {len(pages)} pages...")

            # Process each page
            for i, page in enumerate(pages, 1):
                logger.info(
                    f"Processing page {i}/{len(pages)} (Page number: {page.get('page_number', 'N/A')})"
                )

                try:
                    chunks = self.process_page(page)
                    all_chunks.extend(chunks)
                    logger.info(
                        f"Page {i} processed successfully: {len(chunks)} chunks"
                    )

                except Exception as page_error:
                    logger.error(f"Error processing page {i}: {page_error}")
                    continue

                # Add delay to respect API rate limits
                if i < len(pages):  # Don't delay after the last page
                    time.sleep(delay_between_pages)

            # Save output
            logger.info(f"Saving {len(all_chunks)} chunks to: {output_file_path}")

            # Ensure output directory exists
            output_path = Path(output_file_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)

            logger.info(f"Processing completed successfully!")
            logger.info(f"Total pages processed: {len(pages)}")
            logger.info(f"Total chunks created: {len(all_chunks)}")
            logger.info(f"Output saved to: {output_file_path}")

        except FileNotFoundError:
            logger.error(f"Input file not found: {input_file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in input file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise


def main():
    """
    Main function to run the content processor
    """
    try:
        # Initialize processor (API key from environment variable GEMINI_API_KEY)
        processor = EducationalContentProcessor()

        # Define file paths - adjust these to your actual paths
        input_file = "data/processed/extracted_data.json"  # Your input JSON file
        output_file = "data/processed/chunk_structured_data.json"  # Output file

        # Check if input file exists
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            logger.info("Please update the input_file path in the main() function")
            return

        # Process the file
        processor.process_json_file(
            input_file_path=input_file,
            output_file_path=output_file,
            delay_between_pages=6,  # 1.5 seconds delay between pages
        )

    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
