# pipeline/data_extraction.py

import os
from pathlib import Path
from config.config import IMAGE_DPI, PDF_FILE

from app.data_pipeline.pdf_reader import pdf_to_images
from app.data_pipeline.analyzers.narrative_analyzer import NarrativeAnalyzer
from app.data_pipeline.analyzers.mcq_analyzer import MCQAnalyzer
from app.data_pipeline.analyzers.glossary_analyzer import GlossaryAnalyzer
from app.data_pipeline.analyzers.lesson_intro_analyzer import LessonIntroAnalyzer
from app.data_pipeline.analyzers.author_info_analyzer import AuthorInfoAnalyzer

# === CONFIG ===
PDF_PATH = PDF_FILE
OUTPUT_DIR = Path("app/data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output files
FILES = {
    "narrative": OUTPUT_DIR / "narrative.txt",
    "mcq": OUTPUT_DIR / "mcq_ans.txt",
    "glossary": OUTPUT_DIR / "glossary.txt",
    "lesson_intro": OUTPUT_DIR / "lesson_intro.txt",
    "author_info": OUTPUT_DIR / "author.txt",
}

# Clear existing outputs
for path in FILES.values():
    path.write_text("", encoding="utf-8")

# === INIT ANALYZERS ===
narrative_analyzer = NarrativeAnalyzer()
mcq_analyzer = MCQAnalyzer()
glossary_analyzer = GlossaryAnalyzer()
lesson_intro_analyzer = LessonIntroAnalyzer()
author_info_analyzer = AuthorInfoAnalyzer()


def process_range(analyzer, page_range, out_path, analyze_fn):
    """Helper to process a page range with given analyzer + output."""
    images = pdf_to_images(PDF_PATH, dpi=IMAGE_DPI, page_range=page_range)
    with open(out_path, "a", encoding="utf-8") as f:
        for img, pg in images:
            try:
                print(f"[✓] Processing page {pg}...")
                text = analyze_fn(analyzer, img)
                f.write(f"\n{text}")
            except Exception as e:
                print(f"[!] Failed on page {pg}: {e}")


def run_pipeline():
    print("🔄 Starting full extraction pipeline...\n")

    # Narrative: pages 6–17
    process_range(
        narrative_analyzer,
        (6, 17),
        FILES["narrative"],
        lambda a, i: a.analyze_narrative(i),
    )

    process_range(mcq_analyzer, (23, 32), FILES["mcq"], lambda a, i: a.analyze_mcq(i))

    # Glossary: pages 3–5
    process_range(
        glossary_analyzer, (3, 5), FILES["glossary"], lambda a, i: a.analyze_glossary(i)
    )

    # Lesson Intro: page 19
    process_range(
        lesson_intro_analyzer,
        (19, 19),
        FILES["lesson_intro"],
        lambda a, i: a.analyze_intro(i),
    )

    # Author Info: page 18
    process_range(
        author_info_analyzer,
        (18, 18),
        FILES["author_info"],
        lambda a, i: a.analyze_author_info(i),
    )

    print("\n✅ Extraction complete. All outputs saved to 'output/' directory.")


if __name__ == "__main__":
    run_pipeline()
