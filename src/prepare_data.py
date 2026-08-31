"""
prepare_data.py
Loads a preference dataset, filters it to coding-explanation prompts,
and saves it in the prompt/chosen/rejected format DPOTrainer expects.
"""

from datasets import load_dataset, Dataset

# Keywords that indicate a coding-related prompt
CODE_KEYWORDS = [
    "code", "function", "python", "javascript", "java ", "c++", "algorithm",
    "debug", "error", "compile", "variable", "loop", "array", "class ",
    "syntax", "script", "programming", "sql", "api", "regex", "recursion"
]


def is_coding_prompt(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in CODE_KEYWORDS)


def extract_text(field):
    """UltraFeedback's chosen/rejected are lists of {role, content} dicts."""
    if isinstance(field, list):
        for turn in field:
            if turn.get("role") == "assistant":
                return turn.get("content", "")
    return str(field)


def main():
    print("Loading ultrafeedback_binarized dataset...")
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

    print(f"Total examples before filtering: {len(dataset)}")

    filtered_rows = []
    for row in dataset:
        prompt = row["prompt"]
        if is_coding_prompt(prompt):
            chosen_text = extract_text(row["chosen"])
            rejected_text = extract_text(row["rejected"])

            if not chosen_text or not rejected_text:
                continue
            if chosen_text.strip() == rejected_text.strip():
                continue

            filtered_rows.append({
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
            })

    print(f"Coding-related examples found: {len(filtered_rows)}")

    MAX_EXAMPLES = 500
    if len(filtered_rows) > MAX_EXAMPLES:
        filtered_rows = filtered_rows[:MAX_EXAMPLES]
        print(f"Capped to {MAX_EXAMPLES} examples for faster training")

    final_dataset = Dataset.from_list(filtered_rows)
    split_dataset = final_dataset.train_test_split(test_size=0.1, seed=42)

    split_dataset["train"].to_json("data/train_prefs.jsonl")
    split_dataset["test"].to_json("data/eval_prefs.jsonl")

    print(f"Saved {len(split_dataset['train'])} training examples to data/train_prefs.jsonl")
    print(f"Saved {len(split_dataset['test'])} eval examples to data/eval_prefs.jsonl")


if __name__ == "__main__":
    main()
