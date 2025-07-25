# app/data/data_patches.py
import json
from collections import defaultdict
from app.data_utils.mcq_patches import assign_answer


def add_page_18_prefix(item):

    prefix = "লেখক পরিচিতি - রবীন্দ্রনাথ ঠাকুর। : "
    item["content"] = prefix + item["content"]
    return item


def should_remove_short_answer_with_uddipok(item):
    if (
        item.get("content_type") in ("short_answer", "comprehension")
        and "content" in item
    ):
        if "উদ্দীপক" in item["content"]:
            return True
    return False


def should_remove_mcq_with_uddipok(item):
    if item.get("content_type") == "mcq" and "question" in item:
        if "উদ্দীপক" in item["question"]:
            return True
    return False


def main():
    # Read input JSON file
    with open("data/processed/chunk_structured_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Process each item
    data = [item for item in data if item["page_number"] != 41]
    data = [item for item in data if not should_remove_short_answer_with_uddipok(item)]
    data = [item for item in data if not should_remove_mcq_with_uddipok(item)]

    # Group items by page number to track item numbers
    page_items = defaultdict(int)

    for item in data:
        page_num = item["page_number"]
        content_type = item.get("content_type", "")

        # Increment item counter for this page
        page_items[page_num] += 1

        # Assign ID as page_number + item_number_on_that_page
        item["id"] = f"{page_num}_{page_items[page_num]}"

        # Remove language field if it exists
        if "language" in item:
            del item["language"]

        if item["page_number"] == 18 and "content" in item:
            add_page_18_prefix(item)

        # Skip items without question_number for MCQ processing
        if "question_number" not in item:
            continue

        question_num = int(item["question_number"])

        # Check conditions for MCQ answer assignment
        if content_type == "mcq":
            if (33 <= page_num <= 40) or (page_num == 32 and 1 <= question_num <= 10):
                assign_answer(item)

    # Write output JSON file
    with open("app/data/processed/structured_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
