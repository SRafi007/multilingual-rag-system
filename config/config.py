# config/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# === GENERAL PATHS ===
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "app" / "data"
OUTPUT_DIR = BASE_DIR / "output"
RESOURCES_DIR = BASE_DIR / "resources"

# === PDF SETTINGS ===
PDF_FILE = Path("app") / "data" / "raw" / "HSC26-Bangla1st-Paper.pdf"
PAGE_RANGE = (2, 6)  # Only extract pages 2–40

# === TESSERACT OCR ===
TESSERACT_PATH = r"D:\softwares\code_related\Tesseract-OCR\tesseract.exe"
TESSDATA_PATH = r"D:\softwares\code_related\Tesseract-OCR\tessdata"

# === GEMINI API ===

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY_BACKUP_1 = os.getenv("GEMINI_API_KEY_BACKUP_1")
GEMINI_API_KEY_BACKUP_2 = os.getenv("GEMINI_API_KEY_BACKUP_2")

GEMINI_MODEL = "gemini-2.5-flash"  # Or "gemini-2.0-flash"

# === IMAGE SETTINGS ===
IMAGE_DPI = 300
