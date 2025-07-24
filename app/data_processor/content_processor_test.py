import json
import re
import os
from typing import Dict, List, Any
import google.generativeai as genai


class EducationalContentProcessor:
    def __init__(self, api_key: str):
        """Initialize the processor with Gemini API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")

        # Define content types for file organization
        self.content_types = [
            "mcq",
            "short_answer",
            "learning_outcome",
            "vocabulary",
            "narrative",
            "literary_prose",
            "poetry",
            "dialogue",
            "grammar",
            "matching",
            "fill_in_the_blank",
            "table",
            "summary",
            "comprehension",
            "instruction",
        ]

        # Storage for processed content
        self.processed_content = {
            content_type: [] for content_type in self.content_types
        }
        self.answered_mcqs = []
        self.unanswered_mcqs = []
        self.answer_keys = []

    def clean_noise(self, text: str) -> str:
        """Remove noise patterns from text"""
        noise_patterns = [
            r"10 MINUTE SCHOOL HSC \d+ অনলাইন ব্যাচ.*?📞 16910.*?\)",
            r"For any inquiries related to the online batch, call.*?\)",
            r"HSC \d+ অনলাইন ব্যাচ বাংলা • ইংরেজি • আইসিটি",
            r"\d+ MINUTE SCHOOL",
        ]

        for pattern in noise_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

        return text.strip()

    def extract_answer_key(self, text: str) -> Dict[str, Any]:
        """Extract answer key pattern like 'SL Ans SL Ans...'"""
        pattern = r"SL\s+Ans(?:\s+SL\s+Ans)*\s*((?:\d+\s*[ক-ঘ]\s*)+)"
        match = re.search(pattern, text)

        if match:
            answers_text = match.group(1)
            answers = {}
            # Extract individual question-answer pairs
            qa_pattern = r"(\d+)\s*([ক-ঘ])"
            for qa_match in re.finditer(qa_pattern, answers_text):
                q_num = int(qa_match.group(1))
                answer = qa_match.group(2)
                answers[q_num] = answer

            return {
                "type": "answer_key",
                "answers": answers,
                "raw_text": match.group(0),
            }
        return None

    def get_gemini_prompt(self) -> str:
        """Get the processing prompt for Gemini"""
        return """Objective:
Analyze the provided text to identify its constituent content types, chunk the content accordingly, and extract specific information based on the content type.

Input:
A block of Bengali text containing educational materials.

Task:

Content Type Identification:
Examine the input text and classify each distinct section based on the provided list of content types.

The possible content types are:
mcq (Multiple Choice Question)
short_answer (Short Answer Question)
learning_outcome (Learning Outcome)
vocabulary (Vocabulary Definition)
narrative (Narrative Passage)
literary_prose (Literary Prose)
poetry (Poetry)
dialogue (Dialogue)
grammar (Grammar Exercise)
matching (Matching Exercise)
fill_in_the_blank (Fill-in-the-Blank Exercise)
table (Tabular Data)
summary (Summary)
comprehension (Comprehension Passage)
instruction (Instructional Text)
mixed (Mixed Content)

Content Chunking:
Segment the input text into logical "chunks," where each chunk corresponds to a single identified content type.

Data Extraction (Conditional):
For mcq chunks only:
- Extract the full question text.
- Extract all provided options.
- Identify and extract the correct answer from the answer key (if available) and associate it with the question.

For all other content types:
- Simply provide the full text of the chunk.

Output Format:
The output should be a structured JSON array.
Each chunk should be represented as a separate element in the array.
Each element should contain at least two keys: content_type and content.
For mcq chunks, additional keys for question, options, and correct_answer should be included.

Return ONLY the JSON array, no additional text or explanation."""

    def extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from Gemini response that might be wrapped in markdown"""
        # Remove markdown code blocks if present
        json_pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(json_pattern, response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response_text.strip()

    def process_with_gemini(self, text: str) -> List[Dict[str, Any]]:
        """Process text using Gemini API"""
        try:
            # Clean the text first
            cleaned_text = self.clean_noise(text)

            # Check for answer keys first
            answer_key = self.extract_answer_key(cleaned_text)
            if answer_key:
                self.answer_keys.append(answer_key)
                # Remove answer key from text for processing
                cleaned_text = re.sub(
                    r"SL\s+Ans.*?(?=\n\n|\Z)", "", cleaned_text, flags=re.DOTALL
                )

            prompt = f"{self.get_gemini_prompt()}\n\nText to analyze:\n{cleaned_text}"

            response = self.model.generate_content(prompt)

            # Extract and parse JSON response
            try:
                json_text = self.extract_json_from_response(response.text)
                chunks = json.loads(json_text)
                return chunks if isinstance(chunks, list) else []
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response from page. Error: {e}")
                print(f"Response snippet: {response.text[:300]}...")
                return []

        except Exception as e:
            print(f"Error processing with Gemini: {e}")
            return []

    def process_page(self, page_data: Dict[str, Any]) -> None:
        """Process a single page from the JSON data"""
        page_number = page_data.get("page_number", 0)
        content_type = page_data.get("content_type", "")
        embedding_text = page_data.get("embedding_text", "")

        if not embedding_text:
            return

        # Process with Gemini
        chunks = self.process_with_gemini(embedding_text)

        for chunk in chunks:
            chunk["page_number"] = page_number
            chunk_type = chunk.get("content_type", "").lower()

            # Handle MCQs separately for answered/unanswered
            if chunk_type == "mcq":
                if chunk.get("correct_answer"):
                    self.answered_mcqs.append(chunk)
                else:
                    self.unanswered_mcqs.append(chunk)
            elif chunk_type in self.processed_content:
                self.processed_content[chunk_type].append(chunk)

    def assign_answers_to_mcqs(self) -> None:
        """Assign answers from answer keys to unanswered MCQs"""
        for mcq in self.unanswered_mcqs:
            mcq_page = mcq.get("page_number", 0)

            # Find the closest answer key by page number
            closest_key = None
            min_distance = float("inf")

            for answer_key in self.answer_keys:
                key_page = answer_key.get("page_number", mcq_page)
                distance = abs(key_page - mcq_page)
                if distance < min_distance:
                    min_distance = distance
                    closest_key = answer_key

            if closest_key and "answers" in closest_key:
                # Extract question number from MCQ
                question_text = mcq.get("question", "")
                q_num_match = re.search(r"(\d+)\.", question_text)
                if q_num_match:
                    q_num = int(q_num_match.group(1))
                    if q_num in closest_key["answers"]:
                        mcq["correct_answer"] = closest_key["answers"][q_num]
                        mcq["answer_source"] = f"Answer key from page proximity"

        # Move answered MCQs to answered list
        still_unanswered = []
        for mcq in self.unanswered_mcqs:
            if mcq.get("correct_answer"):
                self.answered_mcqs.append(mcq)
            else:
                still_unanswered.append(mcq)

        self.unanswered_mcqs = still_unanswered

    def save_processed_content(self, output_dir: str = "processed_content") -> None:
        """Save processed content to separate files by type"""
        os.makedirs(output_dir, exist_ok=True)

        # Save MCQs
        if self.answered_mcqs:
            with open(f"{output_dir}/mcq_answered.json", "w", encoding="utf-8") as f:
                json.dump(self.answered_mcqs, f, ensure_ascii=False, indent=2)

        if self.unanswered_mcqs:
            with open(f"{output_dir}/mcq_unanswered.json", "w", encoding="utf-8") as f:
                json.dump(self.unanswered_mcqs, f, ensure_ascii=False, indent=2)

        # Save other content types
        for content_type, content_list in self.processed_content.items():
            if content_list:
                with open(
                    f"{output_dir}/{content_type}.json", "w", encoding="utf-8"
                ) as f:
                    json.dump(content_list, f, ensure_ascii=False, indent=2)

        # Save answer keys
        if self.answer_keys:
            with open(f"{output_dir}/answer_keys.json", "w", encoding="utf-8") as f:
                json.dump(self.answer_keys, f, ensure_ascii=False, indent=2)

        print(f"Processed content saved to {output_dir}/")

    def process_json_file(
        self, input_file: str, output_dir: str = "processed_content"
    ) -> None:
        """Process the main JSON file"""
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            pages = data.get("pages", [])
            total_pages = len(pages)

            print(f"Processing {total_pages} pages...")

            for i, page in enumerate(pages, 1):
                print(f"Processing page {i}/{total_pages}...")
                self.process_page(page)

            # Assign answers to unanswered MCQs
            print("Assigning answers to unanswered MCQs...")
            self.assign_answers_to_mcqs()

            # Save processed content
            self.save_processed_content(output_dir)

            # Print summary
            print("\n=== Processing Summary ===")
            print(f"Answered MCQs: {len(self.answered_mcqs)}")
            print(f"Unanswered MCQs: {len(self.unanswered_mcqs)}")
            for content_type, content_list in self.processed_content.items():
                if content_list:
                    print(f"{content_type.title()}: {len(content_list)}")

        except Exception as e:
            print(f"Error processing file: {e}")


# Usage example
if __name__ == "__main__":
    # Initialize processor with your Gemini API key
    API_KEY = (
        "AIzaSyADj1E0thtPcIW4d9yz7XdM4Bl9pm8PVis"  # Replace with your actual API key
    )
    processor = EducationalContentProcessor(API_KEY)

    # Process your JSON file
    input_file = (
        "data/processed/extracted_data.json"  # Replace with your JSON file path
    )
    output_directory = "processed_content"

    processor.process_json_file(input_file, output_directory)
