# character_reference_resolution.py
import re
from app.data_pipeline.data_preprocessing import load_text_file, normalize_bengali_text


def resolve_narrator_reference(story_text, narrator_name_bn="অনুপম"):
    """
    Replaces Bengali first-person pronouns 'আমি' (ami) and 'আমার' (amar)
    with the narrator's name to resolve coreference.
    This is a rule-based approach specific to the known narrator.
    """
    if not story_text:
        return ""

    # Normalize name for consistent replacement
    narrator_name_bn_normalized = normalize_bengali_text(narrator_name_bn).strip()

    # Common forms of "I" and "my" in Bengali
    # Using regex to ensure whole word replacement
    # 'আমি' - I (nominative)
    # 'আমার' - my (possessive)
    # 'আমাকে' - me (objective)
    # 'আমায়' - me (objective, poetic/short form)
    # 'আমাকেই' - me (emphatic objective)
    # 'আমিই' - I (emphatic nominative)

    # Order of replacement matters: longer forms first to avoid partial matches
    replacements = {
        r"\bআমাকেই\b": f"{narrator_name_bn_normalized}-কেই",  # Emphatic 'me'
        r"\bআমায়\b": f"{narrator_name_bn_normalized}-কে",  # Poetic 'me'
        r"\bআমাকে\b": f"{narrator_name_bn_normalized}-কে",  # 'me'
        r"\bআমিই\b": f"{narrator_name_bn_normalized}-ই",  # Emphatic 'I'
        r"\bআমার\b": f"{narrator_name_bn_normalized}-এর",  # 'my' (possessive suffix 'এর' added to name)
        r"\bআমি\b": narrator_name_bn_normalized,  # 'I'
    }

    resolved_text = story_text
    for old, new in replacements.items():
        # Use re.sub with word boundaries to ensure whole word replacement
        resolved_text = re.sub(old, new, resolved_text)

    return resolved_text


if __name__ == "__main__":
    input_path = "app/data/normalized/normalized_narrative.txt"
    output_path = "app/data/normalized/resolved_narrative.txt"
    narrator_name_bn = "অনুপম"

    print(f"🔍 Loading file: {input_path}")
    story_content = load_text_file(input_path)

    if story_content:
        normalized_story = normalize_bengali_text(story_content)
        resolved_story = resolve_narrator_reference(
            normalized_story, narrator_name_bn=narrator_name_bn
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(resolved_story)
        print(f"✅ Resolved story saved to: {output_path}")
    else:
        print("❌ Failed to load story content.")
