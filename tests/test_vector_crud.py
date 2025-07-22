from app.embeddings.vector_store import VectorStore
from app.embeddings.embedding_manager import EmbeddingManager


def main():
    store = VectorStore()
    store.load()

    # Test query
    query_text = "অনুপমের বয়স কত ছিল?"
    embedder = EmbeddingManager()
    vector = embedder.embed([query_text])[0]

    print("\n[🔍] Top 3 results without filters:")
    results = store.query(vector, top_k=3)
    for r in results:
        print(f"\n>> {r['text'][:200]}")

    print("\n[🔍] Top 3 results with filter: metadata['para_index'] == 2")
    filtered = store.query_with_filter(vector, top_k=3, filters={"para_index": 2})
    for r in filtered:
        print(f"\n>> {r['text'][:200]}")

    # Example: delete all chunks from para_index 2
    # store.delete_by_filter("para_index", 2)


if __name__ == "__main__":
    main()
