# chunking_translation_embedding.py

import os
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from app.data_pipeline.data_preprocessing import load_text_file, normalize_bengali_text
from app.data_pipeline.character_reference_resolution import resolve_narrator_reference

# --- Configuration ---
DATA_DIR = "app/data/normalized"
FILES = {
    "story": "resolved_narrative.txt",
    "glossary": "normalized_glossary.txt",
    "author_info": "normalized_author.txt",
    "story_introduction": "normalized_lesson_intro.txt",
    "small_knowledge": "normalized_mcq_ans.txt",
}
NARRATOR_NAME_BN = "অনুপম"
NARRATOR_NAME_EN = (
    "Anupam"  # Explicitly define English name for clarity in translation context
)

CHUNK_SIZE = 256  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap to maintain context

# --- Models ---
# Initialize Bangla-English NMT model
# You might need to install 'sacremoses' if you get an error: pip install sacremoses
# Note: opus-mt-bn-en is a good general purpose model, but for very nuanced literary text,
# larger models or fine-tuning might be beneficial.
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-bn-en")

# Initialize LaBSE model for embeddings
# This model is specifically designed for multilingual sentence embeddings
embedder = SentenceTransformer("sentence-transformers/LaBSE")

# --- Helper Functions ---


def translate_bangla_to_english(text_bn):
    """Translates Bangla text to English using the NMT model."""
    if not text_bn:
        return ""
    # The translator pipeline takes a list of strings and returns a list of dicts
    # We join them to ensure complete sentences are translated together
    # For very long texts, you might need to chunk before sending to translator
    # or handle potential max_length issues.
    try:
        # Split into sentences for better translation quality, then join back
        sentences_bn = [s.strip() for s in text_bn.split(". ") if s.strip()]
        if not sentences_bn:
            return ""

        # NMT models often have a max input length. Process in batches if text is very long.
        # This is a simplification; for extremely long texts, you'd need more sophisticated batching.
        translated_sentences = []
        for sent_bn in sentences_bn:
            # Ensure the input is within model's typical max length.
            # Opus-MT usually handles up to 512 tokens.
            if len(sent_bn) > 400:  # Arbitrary threshold, adjust based on model limits
                # Handle very long sentences by splitting further or truncating
                # For this project, we'll assume most sentences fit.
                print(
                    f"Warning: Very long sentence being translated, might be truncated by model: {sent_bn[:100]}..."
                )

            result = translator(sent_bn, max_length=512)
            translated_sentences.append(result[0]["translation_text"])
        return " ".join(translated_sentences)
    except Exception as e:
        print(f"Error during translation: {e}")
        return ""


def chunk_text(text, chunk_size, chunk_overlap):
    """
    Splits text into chunks of specified character size with overlap.
    Aims to split at sentence boundaries but defaults to character split if needed.
    """
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        # Try to find a good breaking point (e.g., end of a sentence)
        if end < len(text):
            # Look for a period, question mark, or exclamation mark within the overlap range
            # to break more naturally.
            natural_break_point = -1
            search_start = max(start, end - chunk_overlap)  # Search within overlap
            search_end = end
            for i in range(search_end - 1, search_start, -1):
                if text[i] in [".", "?", "!"]:
                    natural_break_point = i + 1
                    break

            if natural_break_point != -1:
                chunk = text[start:natural_break_point]
                start = natural_break_point  # Next chunk starts after the natural break
            else:
                start += chunk_size - chunk_overlap  # Move by chunk_size minus overlap
        else:
            start += (
                chunk_size - chunk_overlap
            )  # Last chunk, no overlap needed for next

        chunks.append(chunk.strip())
        if end == len(text):  # Reached end of text
            break
    return [c for c in chunks if c]  # Filter out empty chunks


def process_all_files():
    """
    Loads, normalizes, resolves coreferences, translates, and chunks all text files.
    Returns a DataFrame with original Bangla text, translated English text, and chunks.
    """
    all_chunks_data = []

    for file_type, filename in FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        print(f"\n--- Processing {file_type} from {filepath} ---")

        bangla_content = load_text_file(filepath)
        if not bangla_content:
            continue

        # 1. Normalize Bengali Text
        normalized_bn_content = normalize_bengali_text(bangla_content)
        print(f"Normalized Bangla (first 200 chars): {normalized_bn_content[:200]}...")

        # 2. Character Reference Resolution (only for story.txt)
        if file_type == "story":
            processed_bn_content = resolve_narrator_reference(
                normalized_bn_content, NARRATOR_NAME_BN
            )
            print(f"Resolved Bangla (first 200 chars): {processed_bn_content[:200]}...")
        else:
            processed_bn_content = normalized_bn_content

        # 3. Translate to English
        translated_en_content = translate_bangla_to_english(processed_bn_content)
        print(f"Translated English (first 200 chars): {translated_en_content[:200]}...")

        # 4. Chunk the translated English content
        # Chunks are created from the English translation because embeddings will be generated from English.
        chunks_en = chunk_text(translated_en_content, CHUNK_SIZE, CHUNK_OVERLAP)

        # 5. Store information for each chunk
        for i, chunk_en in enumerate(chunks_en):
            # For robust retrieval, also store the corresponding Bangla chunk if possible.
            # This is simplified; a more complex system might align Bangla chunks to English chunks.
            # For now, we'll store the full processed Bangla content with each English chunk.
            all_chunks_data.append(
                {
                    "source_file": file_type,
                    "chunk_id": f"{file_type}_{i}",
                    "original_bn_content": processed_bn_content,  # Full processed Bangla text (for context if needed)
                    "chunk_en": chunk_en,
                    "narrator_name_en": NARRATOR_NAME_EN,  # Add narrator info as metadata
                    "narrator_name_bn": NARRATOR_NAME_BN,
                    "context_type": file_type,  # e.g., 'story', 'glossary', etc.
                }
            )

    return pd.DataFrame(all_chunks_data)


def generate_embeddings(dataframe):
    """Generates embeddings for the 'chunk_en' column of the DataFrame."""
    if "chunk_en" not in dataframe.columns or dataframe.empty:
        print("No English chunks to embed.")
        return dataframe

    print(f"\n--- Generating Embeddings for {len(dataframe)} chunks ---")

    # Get all English chunks
    texts_to_embed = dataframe["chunk_en"].tolist()

    # Generate embeddings
    embeddings = embedder.encode(texts_to_embed, show_progress_bar=True)

    # Add embeddings as a new column
    dataframe["embedding"] = list(embeddings)
    print("Embeddings generated successfully.")

    return dataframe


if __name__ == "__main__":
    # Ensure data directory exists
    if not os.path.exists(DATA_DIR):
        print(
            f"Error: Data directory '{DATA_DIR}' not found. Please create it and place your text files inside."
        )
    else:
        # Step 1-4: Process files and create chunks DataFrame
        processed_data_df = process_all_files()

        if not processed_data_df.empty:
            print("\n--- Processed Data Head ---")
            print(processed_data_df[["source_file", "chunk_id", "chunk_en"]].head())
            print(f"\nTotal chunks generated: {len(processed_data_df)}")

            # Step 5: Generate Embeddings
            final_df = generate_embeddings(processed_data_df)

            if "embedding" in final_df.columns:
                print("\n--- Final DataFrame with Embeddings ---")
                print(final_df.head())
                print(f"Shape of embeddings: {final_df['embedding'].iloc[0].shape}")

                # You would typically save this DataFrame or directly insert it into a vector DB
                # Example: Saving to a parquet file for later use
                # final_df.to_parquet('knowledge_base_embeddings.parquet', index=False)
                # print("\nSaved knowledge_base_embeddings.parquet")
