# app/data/chunking.py
import re
from typing import List, Dict


class Chunker:
    def __init__(self, chunk_size: int = 600, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs based on double line breaks or Bengali full-stops.
        """
        paras = re.split(r"(?:\n{2,}|।\s*)", text)
        return [p.strip() for p in paras if len(p.strip()) > 0]

    def chunk_paragraphs(self, paragraphs: List[str]) -> List[Dict]:
        """
        Convert paragraphs into overlapping fixed-length chunks with metadata.
        """
        chunks = []
        for i, para in enumerate(paragraphs):
            start = 0
            while start < len(para):
                end = start + self.chunk_size
                chunk_text = para[start:end]
                if len(chunk_text) < 100:
                    break

                meta = self.extract_metadata(chunk_text)
                if self.is_valid_chunk(chunk_text):
                    chunks.append(
                        {
                            "text": chunk_text.strip(),
                            "para_index": i,
                            "char_start": start,
                            "metadata": meta,
                        }
                    )

                start += self.chunk_size - self.overlap
        return chunks

    def is_valid_chunk(self, text: str) -> bool:
        """
        Check if a chunk is valid and not gibberish.
        """
        return len(text) >= 100 and len(re.findall(r"[\u0980-\u09FF]", text)) > 20

    def extract_metadata(self, text: str) -> Dict:
        """
        Optionally extract metadata such as character mentions.
        """
        names = re.findall(r"[অ-ঔক-হ][া-ৗ]*", text)
        return {
            "keywords": list(set(names))[:10],  # top 10 unique Bengali words
        }

    def chunk_text(self, full_text: str) -> List[Dict]:
        paragraphs = self.split_paragraphs(full_text)
        return self.chunk_paragraphs(paragraphs)
