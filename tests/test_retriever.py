from app.vector_store.retriever import Retriever


def main():
    retriever = Retriever()

    queries = [
        "Why didn’t Anupam marry Kalyani?",  # English
        "অনুপম কল্যাণীকে বিয়ে করেনি কেন?",  # Bangla
        "What is the meaning of ‘পাকযন্ত্র’?",  # Glossary
        "Who is the author of the story?",  # Author metadata
        "What did the narrator feel about the marriage?",  # Story content
    ]

    for query in queries:
        print(f"\n🔎 Query: {query}")
        results = retriever.search(query, top_k=3)
        for idx, res in enumerate(results):
            print(f"\nResult {idx+1} [{res['source_type']}]:\n{res['text'][:300]}...\n")


if __name__ == "__main__":
    main()
