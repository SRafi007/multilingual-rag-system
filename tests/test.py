output_path = "app/data/output/narrative.txt"
output_path_mcq = "app/data/output/mcq_ans.txt"
output_path_glossary = "app/data/output/glossary.txt"
output_path_lesson = "app/data/output/lesson_intro.txt"
output_path_author = "app/data/output/author.txt"
from app.data_pipeline.pdf_reader import pdf_to_images

"""
import os
from app.data_pipeline.analyzers.narrative_analyzer import NarrativeAnalyzer
from app.data_pipeline.pdf_reader import pdf_to_images

# === Ensure output directory exists ===
output_path = "app/data/output/narrative.txt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# === Initialize analyzer ===
analyzer = NarrativeAnalyzer()

# === Load page images ===
images = pdf_to_images(
    "app/data/raw/HSC26-Bangla1st-Paper.pdf", dpi=300, page_range=(6, 8)
)

# === Analyze each image and write to output ===
with open(output_path, "w", encoding="utf-8") as f:
    for img, pg in images:
        try:
            print(f"Processing page {pg}...")
            text = analyzer.analyze_narrative(img)
            f.write(f"\n{text}")
        except Exception as e:
            print(f"[!] Failed to process page {pg}: {e}")


from app.data_pipeline.analyzers.mcq_analyzer import MCQAnalyzer

analyzer = MCQAnalyzer()
images = pdf_to_images(
    "app/data/raw/HSC26-Bangla1st-Paper.pdf", dpi=300, page_range=(23, 32)
)

with open(output_path_mcq, "w", encoding="utf-8") as f:
    for img, pg in images:
        print(f"Processing page {pg}...")
        text = analyzer.analyze_mcq(img)
        f.write(f"\n{text}")


# ------------
from app.data_pipeline.analyzers.glossary_analyzer import GlossaryAnalyzer

analyzer = GlossaryAnalyzer()
images = pdf_to_images(
    "app/data/raw/HSC26-Bangla1st-Paper.pdf", dpi=300, page_range=(3, 5)
)

with open(output_path_glossary, "w", encoding="utf-8") as f:
    for img, pg in images:
        try:
            print(f"Processing page {pg}...")
            text = analyzer.analyze_glossary(img)
            f.write(f"\n{text}")
        except Exception as e:
            print(f"[!] Failed to process page {pg}: {e}")


# ------------------------
from app.data_pipeline.analyzers.lesson_intro_analyzer import LessonIntroAnalyzer

analyzer = LessonIntroAnalyzer()
images = pdf_to_images(
    "app/data/raw/HSC26-Bangla1st-Paper.pdf", dpi=300, page_range=(19, 19)
)

with open(output_path_lesson, "w", encoding="utf-8") as f:
    for img, pg in images:
        try:
            print(f"Processing page {pg}...")
            text = analyzer.analyze_intro(img)
            f.write(f"\n{text}")
        except Exception as e:
            print(f"[!] Failed to process page {pg}: {e}")

"""
# -----------------------
from app.data_pipeline.analyzers.author_info_analyzer import AuthorInfoAnalyzer

analyzer = AuthorInfoAnalyzer()
images = pdf_to_images(
    "app/data/raw/HSC26-Bangla1st-Paper.pdf", dpi=300, page_range=(18, 18)
)

with open(output_path_author, "w", encoding="utf-8") as f:
    for img, pg in images:
        try:
            print(f"Processing page {pg}...")
            text = analyzer.analyze_author_info(img)
            f.write(f"\n{text}")
        except Exception as e:
            print(f"[!] Failed to process page {pg}: {e}")
