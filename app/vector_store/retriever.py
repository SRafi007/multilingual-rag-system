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
    query_embedding = embedder.encode(query).tolist()

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    print(f"\n🔍 Results for: '{query}'\n")
    for i in range(top_k):
        print(f"[{i+1}] Score: {results['distances'][0][i]:.4f}")
        print(f"    BN: {results['metadatas'][0][i]['text_bn']}")
        print(f"    EN: {results['documents'][0][i]}")
        print(f"    Source: {results['metadatas'][0][i]['source']}")
        print(f"    Section: {results['metadatas'][0][i]['section']}")
        print()


# Example usage:
if __name__ == "__main__":
    search("Who is Anupam?")
