"""Data enhancement module for processing JSON data before embedding"""

import json
import re
from typing import List, Dict, Any, Set
from pathlib import Path
import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from loguru import logger


class DataEnhancer:
    def __init__(self):
        # Comprehensive Bengali keywords organized by scope
        self.bengali_keywords = {
            # Main Characters
            "অপরিচিতা",
            "অনুপম",
            "কল্যাণী",
            "মামা",
            "শম্ভুনাথ",
            "হরিশ",
            "বিনুদাদা",
            "শম্ভুনাথ সেন",
            "ডাক্তার",
            "শিক্ষক",
            "ছাত্রী",
            # Central Themes
            "বিয়ে",
            "বিবাহ",
            "পণপ্রথা",
            "যৌতুক",
            "যৌতুক প্রথা",
            "নারীর মর্যাদা",
            "ব্যক্তিত্ব",
            "ব্যক্তিত্বহীনতা",
            "পিতৃতন্ত্র",
            "পারিবারিক নিয়ন্ত্রণ",
            "আত্মসম্মান",
            "আত্মত্যাগ",
            "প্রতিবাদ",
            "অপমান",
            "অসহায়",
            # Family Relations
            "বাবা",
            "মা",
            "বন্ধু",
            "অভিভাবক",
            "পরিবার",
            "সংসার",
            "মাতৃ-আজ্ঞা",
            # Key Objects & Elements
            "গহনা",
            "সোনা",
            "টাকা",
            "গহনা যাচাই",
            "সেকরা",
            "কষ্টিপাথর",
            "এয়ারিং",
            "মঙ্গলঘট",
            "রসনচৌকি",
            # Locations
            "কানপুর",
            "কলকাতা",
            "ট্রেন",
            "রেলগাড়ি",
            "অন্তঃপুর",
            "অন্দরমহল",
            # Professions & Status
            "ওকালতি",
            "ডাক্তারি",
            "মাস্টারি",
            "ব্যবসা",
            "কলেজ",
            "এমএ পাশ",
            "গরিব",
            "ধনী",
            "সম্পত্তি",
            # Attributes & Qualities
            "বয়স",
            "ভাগ্য",
            "দেবতা",
            "সুপুরুষ",
            "সুন্দর",
            "পুরুষ",
            "নারী",
            "আত্মা",
            "হৃদয়",
            "মন",
            "লজ্জা",
            # Literary Context
            "গল্প",
            "লেখক",
            "রবীন্দ্রনাথ ঠাকুর",
            "সবুজপত্র",
            "গল্পগুচ্ছ",
            "প্রেম",
            "অপেক্ষা",
            "সমাজ",
            # Cultural & Religious Terms
            "আশীর্বাদ",
            "লগ্ন",
            "সভা",
            "বরযাত্রী",
            "ফল্গু",
            "স্বয়ম্বরা",
            "উমেদারি",
            "দক্ষযজ্ঞ",
            "প্রদোষ",
            "মঞ্জরী",
            "জড়িমা",
            # Education & Empowerment
            "মেয়েদের শিক্ষা",
            "মেয়েদের শিক্ষার ব্রত",
            "শিক্ষা",
            "ব্রত",
            # Common Bengali words for better detection
            "গণ্ডুষ",
            "মাতৃভূমি",
            "ভেতরবাড়ি",
        }

        # Create keyword categories for better enhancement
        self.keyword_categories = {
            "characters": {
                "অনুপম",
                "কল্যাণী",
                "মামা",
                "শম্ভুনাথ",
                "হরিশ",
                "বিনুদাদা",
                "শম্ভুনাথ সেন",
                "ডাক্তার",
                "শিক্ষক",
                "ছাত্রী",
            },
            "themes": {
                "যৌতুক",
                "পণপ্রথা",
                "যৌতুক প্রথা",
                "নারীর মর্যাদা",
                "ব্যক্তিত্ব",
                "ব্যক্তিত্বহীনতা",
                "আত্মসম্মান",
                "প্রতিবাদ",
                "পিতৃতন্ত্র",
            },
            "objects": {
                "গহনা",
                "সোনা",
                "টাকা",
                "সেকরা",
                "কষ্টিপাথর",
                "এয়ারিং",
                "মঙ্গলঘট",
                "রসনচৌকি",
            },
            "locations": {"কানপুর", "কলকাতা", "ট্রেন", "রেলগাড়ি", "অন্তঃপুর", "অন্দরমহল"},
            "cultural": {
                "বিয়ে",
                "বিবাহ",
                "আশীর্বাদ",
                "লগ্ন",
                "সভা",
                "বরযাত্রী",
                "স্বয়ম্বরা",
                "দক্ষযজ্ঞ",
            },
        }

    def detect_language(self, text: str) -> str:
        """Detect language of the text"""
        try:
            # Check for Bengali characters
            bengali_pattern = r"[\u0980-\u09FF]"
            if re.search(bengali_pattern, text):
                return "bn"

            # Use langdetect for other languages
            detected = detect(text)
            return detected if detected in ["bn", "en"] else "bn"
        except (LangDetectException, Exception):
            # Default to Bengali if detection fails
            return "bn"

    def extract_keywords(self, text: str, language: str = "bn") -> List[str]:
        """Extract keywords from text"""
        if language == "bn":
            return self._extract_bengali_keywords(text)
        else:
            return self._extract_english_keywords(text)

    def _extract_bengali_keywords(self, text: str) -> List[str]:
        """Extract Bengali keywords with comprehensive matching"""
        keywords = []

        # Extract known important Bengali words
        for keyword in self.bengali_keywords:
            if keyword in text:
                keywords.append(keyword)

        # Extract numbers (important for age, years, etc.)
        bengali_numbers = re.findall(r"[\u09E6-\u09EF]+", text)  # Bengali digits
        english_numbers = re.findall(r"\d+", text)  # English digits
        keywords.extend(bengali_numbers + english_numbers)

        # Extract quoted text (often contains important names/phrases)
        quoted_text = re.findall(r'"([^"]*)"', text)
        keywords.extend(quoted_text)

        # Extract Bengali words that might be names (capitalized)
        bengali_names = re.findall(r"[\u0980-\u09FF]+", text)
        keywords.extend([name for name in bengali_names if len(name) >= 2])

        return list(set(keywords))  # Remove duplicates

    def _extract_english_keywords(self, text: str) -> List[str]:
        """Extract English keywords"""
        # Simple keyword extraction for English
        words = re.findall(r"\b[A-Za-z]{3,}\b", text.lower())

        # Filter out common stop words
        stop_words = {
            "the",
            "and",
            "but",
            "for",
            "are",
            "this",
            "that",
            "with",
            "was",
            "will",
            "have",
            "has",
            "had",
            "can",
            "could",
            "would",
            "should",
            "may",
            "might",
        }
        keywords = [word for word in words if word not in stop_words]

        # Extract numbers
        numbers = re.findall(r"\d+", text)
        keywords.extend(numbers)

        return list(set(keywords))

    def categorize_keywords(self, keywords: List[str]) -> Dict[str, List[str]]:
        """Categorize extracted keywords by type"""
        categorized = {
            "characters": [],
            "themes": [],
            "objects": [],
            "locations": [],
            "cultural": [],
            "other": [],
        }

        for keyword in keywords:
            categorized_flag = False
            for category, category_keywords in self.keyword_categories.items():
                if keyword in category_keywords:
                    categorized[category].append(keyword)
                    categorized_flag = True
                    break

            if not categorized_flag:
                categorized["other"].append(keyword)

        return categorized

    def categorize_content(self, content_type: str) -> str:
        """Categorize content for better retrieval"""
        category_mapping = {
            # Educational content
            "learning_outcome": "educational",
            "instruction": "educational",
            "summary": "educational",
            # Assessment content
            "mcq": "assessment",
            "short_answer": "assessment",
            "matching": "assessment",
            "fill_in_the_blank": "assessment",
            "comprehension": "assessment",
            # Reference content
            "vocabulary": "reference",
            "grammar": "reference",
            "table": "reference",
            # Story/narrative content
            "narrative": "story",
            "literary_prose": "story",
            # Creative content
            "poetry": "creative",
            # Conversation content
            "dialogue": "conversation",
            # Informational content
            "description": "informational",
            # Mixed content
            "mixed": "general",
        }
        return category_mapping.get(content_type, "general")

    def calculate_content_complexity(self, text: str) -> float:
        """Calculate content complexity score"""
        # Simple complexity based on length and sentence structure
        sentences = text.split("।")  # Bengali sentence delimiter
        if not sentences:
            sentences = text.split(".")  # English sentence delimiter

        avg_sentence_length = len(text) / max(len(sentences), 1)

        # Normalize to 0-1 scale
        complexity = min(avg_sentence_length / 100, 1.0)
        return round(complexity, 2)

    def enhance_json_data(
        self, json_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enhance JSON data with additional metadata"""
        enhanced_data = []

        logger.info(f"Enhancing {len(json_data)} items...")

        for item in json_data:
            try:
                content = item.get("content", "")
                content_type = item.get("content_type", "general")

                # Detect language
                language = self.detect_language(content)

                # Extract keywords
                keywords = self.extract_keywords(content, language)

                # Categorize keywords
                categorized_keywords = self.categorize_keywords(keywords)

                # Create enhanced item
                enhanced_item = {
                    **item,  # Keep original data
                    "content_length": len(content),
                    "language": language,
                    "search_keywords": keywords,
                    "categorized_keywords": categorized_keywords,
                    "content_category": self.categorize_content(content_type),
                    "complexity_score": self.calculate_content_complexity(content),
                    "embedding_id": f"{item.get('id', 'unknown')}_emb",
                    "word_count": len(content.split()),
                    "has_question": "?" in content
                    or "কি" in content
                    or "কে" in content,
                    "has_answer": content_type == "mcq" and "correct_answer" in item,
                    "is_dialogue": '"' in content
                    or "বলিলেন" in content
                    or "বললেন" in content,
                    "character_mentions": len(categorized_keywords["characters"]),
                    "theme_relevance": len(categorized_keywords["themes"]),
                    "cultural_elements": len(categorized_keywords["cultural"]),
                }

                # Add specific metadata for MCQs
                if content_type == "mcq" and "options" in item:
                    enhanced_item["option_count"] = len(item["options"])
                    question_text = item.get("question", "")
                    enhanced_item["question_keywords"] = self.extract_keywords(
                        question_text, language
                    )
                    enhanced_item["question_categorized_keywords"] = (
                        self.categorize_keywords(enhanced_item["question_keywords"])
                    )

                enhanced_data.append(enhanced_item)

            except Exception as e:
                logger.error(f"Error enhancing item {item.get('id', 'unknown')}: {e}")
                # Keep original item if enhancement fails
                enhanced_data.append(item)

        logger.info(f"Successfully enhanced {len(enhanced_data)} items")
        return enhanced_data

    def save_enhanced_data(self, enhanced_data: List[Dict[str, Any]], output_path: str):
        """Save enhanced data to JSON file"""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Enhanced data saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving enhanced data: {e}")
            raise

    def generate_statistics(
        self, enhanced_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate statistics about the enhanced dataset"""
        df = pd.DataFrame(enhanced_data)

        # Basic stats
        stats = {
            "total_items": len(enhanced_data),
            "content_types": df["content_type"].value_counts().to_dict(),
            "languages": df["language"].value_counts().to_dict(),
            "content_categories": df["content_category"].value_counts().to_dict(),
            "avg_content_length": float(df["content_length"].mean()),
            "avg_word_count": float(df["word_count"].mean()),
            "avg_complexity": float(df["complexity_score"].mean()),
            "pages_covered": int(df["page_number"].nunique()),
            "items_with_questions": int(df["has_question"].sum()),
            "items_with_answers": int(df["has_answer"].sum()),
            "dialogue_items": int(df["is_dialogue"].sum()),
        }

        # Enhanced stats with keyword analysis
        if "character_mentions" in df.columns:
            stats.update(
                {
                    "avg_character_mentions": float(df["character_mentions"].mean()),
                    "avg_theme_relevance": float(df["theme_relevance"].mean()),
                    "avg_cultural_elements": float(df["cultural_elements"].mean()),
                    "items_with_characters": int((df["character_mentions"] > 0).sum()),
                    "items_with_themes": int((df["theme_relevance"] > 0).sum()),
                    "items_with_cultural_elements": int(
                        (df["cultural_elements"] > 0).sum()
                    ),
                }
            )

        return stats


def main():
    """Main function to enhance data"""
    from app.config.config import Config

    # Initialize enhancer
    enhancer = DataEnhancer()

    # Load original JSON data
    input_path = Config.JSON_DATA_PATH

    if not Path(input_path).exists():
        logger.error(f"Input file not found: {input_path}")
        return

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)

        logger.info(f"Loaded {len(original_data)} items from {input_path}")

        # Enhance data
        enhanced_data = enhancer.enhance_json_data(original_data)

        # Save enhanced data
        enhanced_output_path = input_path.replace(".json", "_enhanced.json")
        enhancer.save_enhanced_data(enhanced_data, enhanced_output_path)

        # Generate and save statistics
        stats = enhancer.generate_statistics(enhanced_data)
        stats_path = input_path.replace(".json", "_stats.json")

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info("Data enhancement completed successfully!")
        logger.info(f"Statistics: {stats}")

    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise


if __name__ == "__main__":
    main()
