import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import google.generativeai as genai


@dataclass
class ProcessedContent:
    page_number: int
    content_type: str
    language: str
    processed_text: str
    original_content: Any
    quality_score: float
    confidence_score: float


@dataclass
class ContentChunk:
    type: str
    content: str
    page: int
    metadata: Optional[Dict] = None


class KnowledgeBaseProcessor:
    def __init__(self, gemini_api_key: str):
        """Initialize the processor with Gemini API key"""
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

        # Storage for different content types
        self.content_storage = {
            "mcqs_with_answers": [],
            "mcqs_without_answers": [],
            "answer_keys": [],
            "literary_prose": [],
            "poetry": [],
            "narrative": [],
            "dialogue": [],
            "vocabulary": [],
            "learning_outcomes": [],
            "short_answers": [],
            "grammar_rules": [],
            "matching_items": [],
            "fill_blanks": [],
            "comprehension": [],
            "tables": [],
            "summaries": [],
            "instructions": [],
            "mixed_content": [],
        }

        # Storage for embedding chunks
        self.embedding_chunks = []

        # Enhanced noise patterns to remove
        self.noise_patterns = [
            r"10 MINUTE SCHOOL.*?(?:16910|আইসিটি).*?(?:\)|।)",
            r"HSC \d+ অনলাইন ব্যাচ.*?আইসিটি",
            r"অনলাইন ব্যাচ সম্পর্কিত.*?16910.*?\)",
            r"For any inquiries.*?16910.*?\)",
            r"কল করো 📞 16910",
            r"call 📞 16910",
            r"বাংলা • ইংরেজি • আইসিটি",
            r"সম্পর্কিত যেকোনো জিজ্ঞাসায়.*?16910",
            # Header noise patterns
            r"PAGE \d+ \| LANGUAGE:.*?QUALITY: \d+%",
            r"={20,}",
            r"-{20,}",
        ]

        # Answer key pattern - more flexible
        self.answer_key_pattern = r"SL\s+Ans(?:\s+SL\s+Ans)*.*?(?:\d+\s+[কখগঘABCDabcd](?:\s+\d+\s+[কখগঘABCDabcd])*)"

    def clean_embedding_text(self, text: str) -> str:
        """Remove noise patterns from embedding text"""
        cleaned_text = text

        # Remove noise patterns
        for pattern in self.noise_patterns:
            cleaned_text = re.sub(
                pattern, "", cleaned_text, flags=re.IGNORECASE | re.DOTALL
            )

        # Remove multiple consecutive whitespaces and newlines
        cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text)  # Keep paragraph breaks
        cleaned_text = re.sub(r" +", " ", cleaned_text)  # Remove multiple spaces
        cleaned_text = cleaned_text.strip()

        return cleaned_text

    def extract_answer_key(self, text: str, page_number: int) -> Optional[Dict]:
        """Extract answer key patterns from text"""
        # Look for the specific pattern you mentioned
        pattern = r"SL\s+Ans(?:\s+SL\s+Ans)*.*?(\d+\s+[কখগঘABCDabcd](?:\s+\d+\s+[কখগঘABCDabcd])*)"

        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)

        if matches:
            answers = {}
            full_match = matches[0]

            # Extract individual question-answer pairs
            answer_pairs = re.findall(r"(\d+)\s+([কখগঘABCDabcd])", full_match)

            for question_num, answer in answer_pairs:
                answers[int(question_num)] = answer.strip()

            if answers:  # Only return if we found actual answers
                return {
                    "page_number": page_number,
                    "answers": answers,
                    "raw_text": full_match,
                }

        return None

    def process_mcqs(
        self, mcqs: List[Dict], page_number: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """Separate MCQs with and without answers"""
        with_answers = []
        without_answers = []

        for mcq in mcqs:
            mcq_copy = {**mcq, "page_number": page_number}

            # Check multiple possible answer fields
            answer_fields = ["correct_answer", "answer", "ans"]
            has_answer = False

            for field in answer_fields:
                if field in mcq and mcq[field] and str(mcq[field]).strip():
                    # Normalize answer field name
                    if field != "correct_answer":
                        mcq_copy["correct_answer"] = mcq_copy.pop(field)
                    has_answer = True
                    break

            if has_answer:
                with_answers.append(mcq_copy)
            else:
                without_answers.append(mcq_copy)

        return with_answers, without_answers

    def create_chunks_from_structured_data(
        self, structured_content: Dict, page_number: int, content_type: str
    ) -> List[ContentChunk]:
        """Create embedding chunks from structured_content"""
        chunks = []

        if not structured_content:
            return chunks

        # Process each content type in structured_content
        for struct_type, content_list in structured_content.items():
            if not content_list:
                continue

            # Create a chunk for each content type
            chunk_content = self.format_structured_content_for_chunk(
                struct_type, content_list
            )

            if chunk_content:
                chunks.append(
                    ContentChunk(
                        type=struct_type,
                        content=chunk_content,
                        page=page_number,
                        metadata={
                            "source": "structured_data",
                            "original_type": struct_type,
                            "items_count": len(content_list),
                            "parent_content_type": content_type,
                        },
                    )
                )

        return chunks

    def format_structured_content_for_chunk(
        self, content_type: str, content_list: List
    ) -> str:
        """Format structured content into clean text for embedding"""
        if not content_list:
            return ""

        formatted_content = ""

        if content_type == "mcqs":
            for mcq in content_list:
                # Add question
                question_num = mcq.get("question_number", "")
                question_text = mcq.get("question", "")
                formatted_content += f"প্রশ্ন {question_num}: {question_text}\n"

                # Add options
                options = mcq.get("options", {})
                if isinstance(options, dict):
                    for option_key, option_value in options.items():
                        formatted_content += f"{option_key}) {option_value}\n"

                # Add answer if available
                if mcq.get("correct_answer") or mcq.get("answer"):
                    answer = mcq.get("correct_answer") or mcq.get("answer")
                    formatted_content += f"সঠিক উত্তর: {answer}\n"

                formatted_content += "\n"

        elif content_type == "learning_outcomes":
            formatted_content = "শিখনফল:\n"
            for i, outcome in enumerate(content_list, 1):
                if isinstance(outcome, dict):
                    outcome_text = outcome.get("outcome", "")
                else:
                    outcome_text = str(outcome)
                formatted_content += f"{i}. {outcome_text}\n"

        elif content_type == "vocabulary":
            formatted_content = "শব্দভাণ্ডার:\n"
            for vocab in content_list:
                if isinstance(vocab, dict):
                    word = vocab.get("word", "")
                    meaning = vocab.get("meaning", "")
                    formatted_content += f"• {word}: {meaning}\n"
                else:
                    formatted_content += f"• {vocab}\n"

        elif content_type == "narratives":
            for narrative in content_list:
                if isinstance(narrative, dict):
                    if narrative.get("title"):
                        formatted_content += f"শিরোনাম: {narrative['title']}\n"
                    if narrative.get("author"):
                        formatted_content += f"লেখক: {narrative['author']}\n"
                    if narrative.get("paragraphs"):
                        formatted_content += "\n".join(narrative["paragraphs"])
                    formatted_content += "\n\n"
                else:
                    formatted_content += f"{narrative}\n\n"

        elif content_type == "short_answers":
            for i, qa in enumerate(content_list, 1):
                if isinstance(qa, dict):
                    question = qa.get("question", "")
                    formatted_content += f"প্রশ্ন {i}: {question}\n"
                    if qa.get("expected_format"):
                        formatted_content += (
                            f"প্রত্যাশিত ফরম্যাট: {qa['expected_format']}\n"
                        )
                    if qa.get("marks"):
                        formatted_content += f"নম্বর: {qa['marks']}\n"
                    formatted_content += "\n"

        elif content_type == "comprehension":
            for comp in content_list:
                if isinstance(comp, dict):
                    if comp.get("passage"):
                        formatted_content += f"অনুচ্ছেদ:\n{comp['passage']}\n\n"
                    if comp.get("questions"):
                        formatted_content += "প্রশ্নসমূহ:\n"
                        for i, q in enumerate(comp["questions"], 1):
                            question_text = q.get("question", "")
                            formatted_content += f"{i}. {question_text}\n"
                    formatted_content += "\n"

        elif content_type == "grammar_rules":
            for rule in content_list:
                if isinstance(rule, dict):
                    rule_text = rule.get("rule", "")
                    formatted_content += f"নিয়ম: {rule_text}\n"
                    if rule.get("examples"):
                        formatted_content += "উদাহরণ:\n"
                        for example in rule["examples"]:
                            formatted_content += f"• {example}\n"
                    formatted_content += "\n"

        elif content_type == "fill_blanks":
            for fill in content_list:
                if isinstance(fill, dict):
                    sentence = fill.get("sentence", "")
                    formatted_content += f"বাক্য: {sentence}\n"
                    if fill.get("options"):
                        formatted_content += (
                            "বিকল্প: " + ", ".join(fill["options"]) + "\n"
                        )
                    formatted_content += "\n"

        elif content_type == "matching_items":
            for match in content_list:
                if isinstance(match, dict):
                    if match.get("instructions"):
                        formatted_content += f"নির্দেশনা: {match['instructions']}\n"
                    if match.get("left_items") and match.get("right_items"):
                        formatted_content += "বাম কলাম:\n"
                        for item in match["left_items"]:
                            formatted_content += f"• {item}\n"
                        formatted_content += "ডান কলাম:\n"
                        for item in match["right_items"]:
                            formatted_content += f"• {item}\n"
                    formatted_content += "\n"

        elif content_type == "tables":
            for table in content_list:
                if isinstance(table, dict):
                    if table.get("headers"):
                        formatted_content += " | ".join(table["headers"]) + "\n"
                        formatted_content += (
                            "-" * (len(" | ".join(table["headers"]))) + "\n"
                        )
                    if table.get("rows"):
                        for row in table["rows"]:
                            formatted_content += (
                                " | ".join(str(cell) for cell in row) + "\n"
                            )
                    formatted_content += "\n"

        elif content_type == "summaries":
            for summary in content_list:
                if isinstance(summary, dict):
                    if summary.get("title"):
                        formatted_content += f"সারসংক্ষেপ: {summary['title']}\n"
                    if summary.get("content"):
                        formatted_content += f"{summary['content']}\n"
                    if summary.get("key_points"):
                        formatted_content += "মূল বিষয়সমূহ:\n"
                        for point in summary["key_points"]:
                            formatted_content += f"• {point}\n"
                    formatted_content += "\n"

        elif content_type == "instructions":
            for instruction in content_list:
                if isinstance(instruction, dict):
                    step = instruction.get("step", "")
                    inst_text = instruction.get("instruction", "")
                    formatted_content += f"ধাপ {step}: {inst_text}\n"
                    if instruction.get("details"):
                        formatted_content += f"বিস্তারিত: {instruction['details']}\n"
                    formatted_content += "\n"

        else:
            # Generic formatting for other content types
            for item in content_list:
                if isinstance(item, dict):
                    # Try to extract meaningful text from dict
                    text_fields = [
                        "content",
                        "text",
                        "rule",
                        "instruction",
                        "sentence",
                        "outcome",
                    ]
                    for field in text_fields:
                        if field in item and item[field]:
                            formatted_content += f"{item[field]}\n"
                            break
                    else:
                        formatted_content += f"{str(item)}\n"
                else:
                    formatted_content += f"{item}\n"

        return formatted_content.strip()

    def create_embedding_chunks(
        self,
        page_number: int,
        content_type: str,
        structured_content: Dict,
        embedding_text: str = "",
    ) -> List[ContentChunk]:
        """Create embedding chunks, prioritizing structured_content over embedding_text"""
        chunks = []

        # First priority: Use structured_content if available
        if structured_content:
            chunks = self.create_chunks_from_structured_data(
                structured_content, page_number, content_type
            )

            if chunks:  # If we successfully created chunks from structured data
                return chunks

        # Fallback: Use embedding_text if structured_content is empty or failed
        if embedding_text.strip():
            cleaned_text = self.clean_embedding_text(embedding_text)

            # Process with Gemini for better formatting
            try:
                processed_text = self.process_embedding_text_sync(
                    cleaned_text, page_number, content_type
                )
            except:
                processed_text = cleaned_text

            chunks.append(
                ContentChunk(
                    type=content_type,
                    content=processed_text,
                    page=page_number,
                    metadata={
                        "source": "embedding_text",
                        "original_type": content_type,
                        "fallback_used": True,
                    },
                )
            )

        return chunks

    def process_embedding_text_sync(
        self, text: str, page_number: int, content_type: str
    ) -> str:
        """Synchronous version of Gemini processing for embedding text"""
        prompt = f"""
        Clean and format the following {content_type} educational content for knowledge base embedding.
        
        INSTRUCTIONS:
        1. Remove promotional text, noise, headers, footers
        2. Keep ALL educational content intact
        3. Format content clearly and logically
        4. Maintain original language exactly
        5. Return only clean educational content
        
        Content from Page {page_number}:
        {text}
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini processing error for page {page_number}: {e}")
            return text

    def process_structured_content(
        self, structured_content: Dict, page_number: int, content_type: str
    ):
        """Process structured content and store by type"""

        for struct_type, content_list in structured_content.items():
            if not content_list:  # Skip empty lists
                continue

            if struct_type == "mcqs":
                with_ans, without_ans = self.process_mcqs(content_list, page_number)
                self.content_storage["mcqs_with_answers"].extend(with_ans)
                self.content_storage["mcqs_without_answers"].extend(without_ans)
            else:
                # Add page number to each item
                enhanced_content = []
                for item in content_list:
                    if isinstance(item, dict):
                        enhanced_content.append({**item, "page_number": page_number})
                    else:
                        enhanced_content.append(
                            {"content": item, "page_number": page_number}
                        )

                # Store in appropriate category
                if struct_type in self.content_storage:
                    self.content_storage[struct_type].extend(enhanced_content)
                else:
                    # For unknown types, store in mixed_content
                    self.content_storage["mixed_content"].extend(enhanced_content)

    def assign_answers_to_mcqs(self):
        """Assign answers from answer keys to unanswered MCQs"""
        print(f"Found {len(self.content_storage['answer_keys'])} answer keys")
        print(f"Unanswered MCQs: {len(self.content_storage['mcqs_without_answers'])}")

        for answer_key in self.content_storage["answer_keys"]:
            answer_page = answer_key["page_number"]
            answers = answer_key["answers"]

            print(
                f"Processing answer key from page {answer_page} with {len(answers)} answers"
            )

            # Find MCQs without answers that could match
            for mcq in self.content_storage["mcqs_without_answers"][:]:
                if mcq.get("answer_assigned"):
                    continue

                mcq_page = mcq["page_number"]
                question_num = mcq.get("question_number")

                if question_num and question_num in answers:
                    # Calculate page distance
                    page_distance = abs(answer_page - mcq_page)

                    # Assign answer if within reasonable distance
                    if page_distance <= 5:  # Within 5 pages
                        mcq["correct_answer"] = answers[question_num]
                        mcq["answer_source_page"] = answer_page
                        mcq["answer_assigned"] = True

                        # Move to answered MCQs
                        self.content_storage["mcqs_with_answers"].append(mcq)
                        print(
                            f"Assigned answer {answers[question_num]} to Q{question_num} from page {mcq_page}"
                        )

        # Remove answered MCQs from unanswered list
        self.content_storage["mcqs_without_answers"] = [
            mcq
            for mcq in self.content_storage["mcqs_without_answers"]
            if not mcq.get("answer_assigned")
        ]

    async def process_page(self, page_data: Dict) -> Optional[ProcessedContent]:
        """Process a single page"""
        page_number = page_data["page_number"]
        content_type = page_data.get("content_type", "unknown")
        language = page_data.get("language", "unknown")
        embedding_text = page_data.get("embedding_text", "")
        structured_content = page_data.get("structured_content", {})

        # Check for answer keys in the embedding text
        if embedding_text.strip():
            cleaned_text = self.clean_embedding_text(embedding_text)
            answer_key = self.extract_answer_key(cleaned_text, page_number)
            if answer_key:
                self.content_storage["answer_keys"].append(answer_key)
                print(
                    f"Found answer key on page {page_number} with {len(answer_key['answers'])} answers"
                )

        # Process structured content if exists
        if structured_content:
            self.process_structured_content(
                structured_content, page_number, content_type
            )

        # Create embedding chunks (prioritizes structured_content)
        chunks = self.create_embedding_chunks(
            page_number, content_type, structured_content, embedding_text
        )
        self.embedding_chunks.extend(chunks)

        # Create a summary of processed content for the main file
        if chunks:
            processed_text = "\n\n".join([chunk.content for chunk in chunks])
        else:
            processed_text = (
                self.clean_embedding_text(embedding_text) if embedding_text else ""
            )

        return ProcessedContent(
            page_number=page_number,
            content_type=content_type,
            language=language,
            processed_text=processed_text,
            original_content=structured_content,
            quality_score=page_data.get("quality_score", 0),
            confidence_score=page_data.get("confidence_score", 0),
        )

    async def process_json_file(
        self, json_file_path: str, output_dir: str = "processed_knowledge"
    ):
        """Process the entire JSON file"""

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Load JSON data
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pages = data.get("pages", [])
        print(f"Processing {len(pages)} pages...")

        # Process each page
        processed_pages = []
        processed_count = 0

        for i, page_data in enumerate(pages):
            try:
                processed_page = await self.process_page(page_data)
                if processed_page:
                    processed_pages.append(processed_page)
                    processed_count += 1
                    print(
                        f"Processed page {processed_page.page_number} ({processed_count}/{len(pages)})"
                    )

            except Exception as e:
                print(f"Error processing page {page_data.get('page_number', i+1)}: {e}")

        # Assign answers to unanswered MCQs
        print("\nAssigning answers to MCQs...")
        self.assign_answers_to_mcqs()

        # Save processed content by content type
        await self.save_content_by_type(output_path, processed_pages)

        # Print summary
        print(f"\nProcessing completed!")
        print(f"Total pages processed: {len(processed_pages)}")
        print(f"Total embedding chunks created: {len(self.embedding_chunks)}")
        print(f"MCQs with answers: {len(self.content_storage['mcqs_with_answers'])}")
        print(
            f"MCQs without answers: {len(self.content_storage['mcqs_without_answers'])}"
        )
        print(f"Answer keys found: {len(self.content_storage['answer_keys'])}")

        # Print content type summary
        for content_type, items in self.content_storage.items():
            if items and content_type not in ["answer_keys"]:
                print(f"{content_type}: {len(items)} items")

        # Print chunk source summary
        structured_chunks = sum(
            1
            for chunk in self.embedding_chunks
            if chunk.metadata.get("source") == "structured_data"
        )
        embedding_chunks = sum(
            1
            for chunk in self.embedding_chunks
            if chunk.metadata.get("source") == "embedding_text"
        )
        print(f"\nChunk Sources:")
        print(f"From structured_data: {structured_chunks}")
        print(f"From embedding_text (fallback): {embedding_chunks}")

    async def save_content_by_type(
        self, output_path: Path, processed_pages: List[ProcessedContent]
    ):
        """Save processed content organized by content type"""

        # Save embedding chunks with enhanced metadata
        embedding_data = {}
        for chunk in self.embedding_chunks:
            if chunk.page not in embedding_data:
                embedding_data[chunk.page] = {"page": chunk.page, "chunks": []}

            embedding_data[chunk.page]["chunks"].append(
                {
                    "type": chunk.type,
                    "content": chunk.content,
                    "metadata": chunk.metadata or {},
                }
            )

        # Save embedding format
        embedding_file = output_path / "embedding_chunks.json"
        with open(embedding_file, "w", encoding="utf-8") as f:
            # Convert to list format sorted by page number
            embedding_list = [
                embedding_data[page] for page in sorted(embedding_data.keys())
            ]
            json.dump(embedding_list, f, ensure_ascii=False, indent=2)

        print(f"Embedding chunks saved to: {embedding_file}")

        # Save main processed content
        main_file = output_path / "processed_knowledge_base.txt"
        with open(main_file, "w", encoding="utf-8") as f:
            f.write("PROCESSED KNOWLEDGE BASE\n")
            f.write("=" * 50 + "\n\n")

            for page in sorted(processed_pages, key=lambda x: x.page_number):
                f.write(f"PAGE {page.page_number}\n")
                f.write(page.processed_text)
                f.write("\n\n")

        # Save content by type in separate files
        for content_type, items in self.content_storage.items():
            if not items:
                continue

            filename = output_path / f"{content_type}.json"

            # Convert to JSON serializable format
            json_items = []
            for item in items:
                if hasattr(item, "__dict__"):
                    json_items.append(item.__dict__)
                else:
                    json_items.append(item)

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(json_items, f, ensure_ascii=False, indent=2)

            print(f"Saved {len(items)} {content_type} items to {filename}")

        print(f"Main knowledge base saved to: {main_file}")


# Usage example
async def main():
    # Initialize processor with your Gemini API key
    processor = KnowledgeBaseProcessor(
        gemini_api_key="AIzaSyADj1E0thtPcIW4d9yz7XdM4Bl9pm8PVis"  # Replace with your actual API key
    )

    # Process the JSON file
    await processor.process_json_file(
        json_file_path="data/processed/extracted_data.json",
        output_dir="processed_knowledge_base",
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
