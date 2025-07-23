"setting.py"

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# === Base Project Path ===
BASE_DIR = Path(__file__).resolve().parent.parent

# === Environment Variables ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
TESSDATA_PATH = os.getenv("TESSDATA_PATH")

# === Static Configs (non-sensitive) ===
PDF_PATH = BASE_DIR / "data" / "raw" / "HSC26-Bangla1st-Paper.pdf"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# === Validations ===
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY is not set in .env")

if not TESSERACT_PATH or not TESSDATA_PATH:
    raise ValueError("❌ TESSERACT_PATH or TESSDATA_PATH is not set in .env")
