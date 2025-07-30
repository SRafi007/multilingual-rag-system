import json
import os
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# --- Config ---
CHUNK_FILE = "app/data/chunks/processed_chunks.json"
DB_DIR = "app/data/chroma_db"
COLLECTION_NAME = "multilingual_story_chunks"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    # --- Init (Updated for new ChromaDB) ---
    # Create the directory if it doesn't exist
    os.makedirs(DB_DIR, exist_ok=True)

    # Use the new PersistentClient instead of the deprecated Client with Settings
    print("[+] Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path=DB_DIR)

    # Delete existing collection if it exists (optional - remove this if you want to append)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"[+] Deleted existing collection '{COLLECTION_NAME}'")
    except:
        pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print(f"[+] Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    # --- Load Chunks ---
    print(f"[+] Loading chunks from: {CHUNK_FILE}")
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[+] Loaded {len(chunks)} chunks.")

    # --- Process and Add to ChromaDB ---
    valid_chunks = 0
    skipped_chunks = 0

    # Prepare batch data for efficient insertion
    documents = []
    embeddings = []
    metadatas = []
    ids = []

    print("[+] Processing chunks and generating embeddings...")
    for chunk in tqdm(chunks, desc="Processing chunks"):
        # Use English text for embedding (as it's translated)
        text_en = chunk.get("text_en")
        text_bn = chunk.get("text_bn", "")

        if not text_en:
            print(
                f"[!] Skipping chunk {chunk.get('chunk_id', 'unknown')} - no English text"
            )
            skipped_chunks += 1
            continue

        # Generate embedding from English text
        embedding = embedder.encode(text_en).tolist()

        # Prepare metadata with both original and translated text
        metadata = {
            "chunk_id": chunk["chunk_id"],
            "language": chunk["metadata"]["language"],
            "section": chunk["metadata"]["section"],
            "source": chunk["source"],
            "index": chunk["index"],
            "translated": chunk["metadata"].get("translated", False),
            "text_bn": text_bn,  # Store original Bengali text in metadata
            "text_length": len(text_en),
        }

        # Add to batch
        documents.append(text_en)
        embeddings.append(embedding)
        metadatas.append(metadata)
        ids.append(chunk["chunk_id"])

        valid_chunks += 1

    # Batch insert for better performance
    if documents:
        print(f"[+] Adding {len(documents)} chunks to ChromaDB...")
        collection.add(
            documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids
        )

    print(f"✅ Successfully processed {valid_chunks} chunks")
    print(f"⚠️  Skipped {skipped_chunks} chunks due to missing English text")
    print(f"✅ All chunks embedded and stored in ChromaDB at '{DB_DIR}'")

    # Verify the collection
    count = collection.count()
    print(f"📊 Collection '{COLLECTION_NAME}' now contains {count} documents")


if __name__ == "__main__":
    main()
