✅ Requirements
* Python 3.8+
* Poppler
* Tesseract OCR with Bangla language support (`ben.traineddata`)

🛠️ 1. Install Python dependencies

```bash
pip install pdf2image pytesseract pillow
```

🧩 2. Install Poppler
* Download: https://github.com/oschwartz10612/poppler-windows/releases
* Extract and add `.../Library/bin` to your system `PATH`

🔤 3. Install Tesseract OCR
* Download: https://github.com/UB-Mannheim/tesseract/wiki
* Install it to a known path (e.g. `D:\softwares\code_related\Tesseract-OCR`)
* Ensure `ben.traineddata` is in the `tessdata` folder

⚙️ 4. Set Tesseract path in code
In `pdf_ocr_processor.py`, update:

```python
pytesseract.pytesseract.tesseract_cmd = r"D:\softwares\code_related\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"D:\softwares\code_related\Tesseract-OCR\tessdata"
```

▶️ 5. Run the OCR script

```bash
python -m app.data.pdf_ocr_processor
```

It will extract Bangla text from the PDF and save it to:

```bash
data/processed/bangla_extracted_text.txt
```