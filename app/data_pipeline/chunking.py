from bnltk.tokenize import Tokenizers
import json
from pathlib import Path
import re

# Initialize the tokenizer
bn_tokenizer = Tokenizers()


def simple_bengali_sentence_split(text):
    """Simple sentence splitting for Bengali text based on punctuation"""
    # Bengali sentence ending punctuation marks
    sentence_endings = r"[।!?।।]+"
    sentences = re.split(sentence_endings, text)
    # Clean up and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def chunk_bengali_text(
    text: str, source: str, source_type: str, max_sentences_per_chunk=2
):
    """Split Bengali text into sentence-based chunks"""
    try:
        # Try to use BNLTK tokenizer if it has sentence tokenization
        if hasattr(bn_tokenizer, "bn_sent_tokenizer"):
            sentences = bn_tokenizer.bn_sent_tokenizer(text)
            print("using bn_sent")
        elif hasattr(bn_tokenizer, "sent_tokenizer"):
            print("using bn_token")
            sentences = bn_tokenizer.sent_tokenizer(text)
        else:
            # Fallback to simple sentence splitting
            print("using bn_fall")
            sentences = simple_bengali_sentence_split(text)
    except Exception as e:
        print(f"Error with BNLTK tokenizer: {e}")
        print("Using simple sentence splitting...")
        sentences = simple_bengali_sentence_split(text)

    chunks = []
    chunk = []
    chunk_index = 0

    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if sentence:  # Only add non-empty sentences
            chunk.append(sentence)

        if len(chunk) >= max_sentences_per_chunk or i == len(sentences) - 1:
            if chunk:  # Only create chunk if it has content
                chunks.append(
                    {
                        "text": " ".join(chunk),
                        "source": source,
                        "source_type": source_type,
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1
                chunk = []  # reset

    return chunks


# Example usage
if __name__ == "__main__":
    # Example text block (you can read this from a file too)
    raw_text = """১৭ বছর বয়সে ব্যারিস্টারি পড়তে ইংল্যান্ডে গেলেও কোর্স সম্পন্ন করা সম্ভব হয়নি। তবে গৃহশিক্ষকের কাছ থেকে জ্ঞানার্জনে রবীন্দ্রনাথ ঠাকুরের কোনো ত্রুটি হয়নি। ১৮৮৪ খ্রিস্টাব্দ থেকে রবীন্দ্রনাথ ঠাকুর পিতার আদেশে বিষয়কর্ম পরিদর্শনে নিযুক্ত হন এবং ১৮৯০ খ্রিস্টাব্দ থেকে দেশের বিভিন্ন অঞ্চলে জমিদারি দেখাশুনা করেন। এ সূত্রে রবীন্দ্রনাথ ঠাকুর শিলাইদহ ও সিরাজগঞ্জের শাহজাদপুরে দীর্ঘ সময় অবস্থান করেন।"""

    chunks = chunk_bengali_text(
        raw_text, source="cleaned_author.txt", source_type="author"
    )

    # Save to file (optional)
    with open("output_chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Print for preview
    for c in chunks:
        print(json.dumps(c, ensure_ascii=False, indent=2))
