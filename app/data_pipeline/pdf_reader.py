# pipeline/pdf_reader.py

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import io
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def pdf_to_images(
    pdf_path: str, dpi: int = 300, page_range: Tuple[int, int] = None
) -> List[Tuple[np.ndarray, int]]:
    """
    Convert PDF pages into high-quality images.

    Args:
        pdf_path: Path to PDF file
        dpi: Image resolution
        page_range: (start_page, end_page), 1-indexed inclusive

    Returns:
        List of tuples (image_array, page_number)
    """
    images = []
    pdf_doc = fitz.open(pdf_path)

    for page_index in range(pdf_doc.page_count):
        page_number = page_index + 1

        # Apply page range filter
        if page_range:
            start, end = page_range
            if not (start <= page_number <= end):
                continue

        page = pdf_doc[page_index]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        img_data = pix.tobytes("ppm")
        img = Image.open(io.BytesIO(img_data))
        img_array = np.array(img)

        images.append((img_array, page_number))
        logger.info(f"Converted page {page_number}")

    pdf_doc.close()
    return images
