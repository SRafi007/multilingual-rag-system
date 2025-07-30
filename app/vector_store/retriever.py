from sentence_transformers import SentenceTransformer
import chromadb

# --- Config ---
DB_DIR = "app/data/chroma_db"
COLLECTION_NAME = "multilingual_story_chunks"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Initialize
print("[+] Loading ChromaDB and model...")
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection(name=COLLECTION_NAME)
embedder = SentenceTransformer(EMBED_MODEL)


def search(query, top_k=5):
    """Search for relevant chunks"""
    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    print(f"\n🔍 Results for: '{query}'")
    print("=" * 60)

    # Check if we have results
    if not results["documents"][0]:
        print("No results found!")
        return

    seen_chunks = set()  # To avoid duplicates
    result_count = 0

    for i in range(len(results["documents"][0])):
        if result_count >= top_k:
            break

        # Get data
        document = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        # Skip duplicates
        chunk_id = metadata.get("chunk_id", f"chunk_{i}")
        if chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)

        # Convert distance to similarity score (lower distance = higher similarity)
        similarity = max(0, 1 - distance / 2)  # Rough normalization

        result_count += 1

        print(f"\n[{result_count}] Chunk ID: {chunk_id}")
        print(f"    Similarity: {similarity:.3f} (Distance: {distance:.4f})")
        print(f"    Bengali:  {metadata.get('text_bn', 'N/A')}")
        print(f"    English:  {document}")
        print(f"    Source:   {metadata.get('source', 'N/A')}")
        print(f"    Section:  {metadata.get('section', 'N/A')}")
        print(f"    Index:    {metadata.get('index', 'N/A')}")


def search_multilingual(queries, top_k=3):
    """Search with multiple queries"""
    print("\n" + "=" * 80)
    print("MULTILINGUAL SEARCH DEMO")
    print("=" * 80)

    for query in queries:
        search(query, top_k)
        print("\n" + "-" * 60)


def get_collection_stats():
    """Show collection statistics"""
    try:
        count = collection.count()
        print(f"\n📊 Collection Statistics:")
        print(f"    Total documents: {count}")

        # Get a sample to show variety
        sample = collection.get(limit=10, include=["metadatas"])

        if sample["metadatas"]:
            sources = set()
            sections = set()
            languages = set()

            for meta in sample["metadatas"]:
                sources.add(meta.get("source", "unknown"))
                sections.add(meta.get("section", "unknown"))
                languages.add(meta.get("language", "unknown"))

            print(f"    Sources: {', '.join(sources)}")
            print(f"    Sections: {', '.join(sections)}")
            print(f"    Languages: {', '.join(languages)}")

    except Exception as e:
        print(f"Error getting stats: {e}")


# Example usage
if __name__ == "__main__":
    # Show collection info
    get_collection_stats()

    # Single search
    search("Who is Anupam?", top_k=5)

    # Multiple searches in different languages
    search_multilingual(
        [
            "অনুপম কে?",  # Bengali: Who is Anupam?
            "life value",  # English
            "জীবনের মূল্য",  # Bengali: life's value
            "uncle mama",  # English
            "গল্পের নায়ক",  # Bengali: story's hero
        ],
        top_k=3,
    )
