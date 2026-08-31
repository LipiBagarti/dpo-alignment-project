"""
evaluate.py
Compares base model vs DPO-tuned model on held-out prompts using an LLM judge.
Calculates win-rate with a 95% confidence interval from actual measured results.
"""

import json
import random
import time
import math
import os
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from openai import OpenAI

BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DPO_MODEL_DIR = "dpo-coding-model"
EVAL_FILE = "data/eval_prefs.jsonl"
RESULTS_FILE = "data/evaluation_results.json"

JUDGE_PROMPT_TEMPLATE = """Given a user prompt and two candidate responses, determine which response is better based on helpfulness, correctness, relevance, and clarity. Do not favor either response based on its position.

Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Return ONLY a JSON object in this exact format, nothing else:
{{"winner": "A" or "B" or "tie", "reason": "one sentence explanation", "confidence": 0.0 to 1.0}}
"""


def load_models():
    print("Loading base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )

    print("Loading DPO fine-tuned model (base + LoRA adapter)...")
    dpo_model = PeftModel.from_pretrained(base_model, DPO_MODEL_DIR)

    return base_model, dpo_model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=200):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def judge_pair(client, prompt, response_a, response_b, retries=3):
    judge_input = JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt, response_a=response_a, response_b=response_b
    )

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": judge_input}],
                temperature=0,
            )
            content = completion.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            print(f"Judge call failed (attempt {attempt + 1}): {e}")
            time.sleep(2)

    return {"winner": "tie", "reason": "judge failed after retries", "confidence": 0.0}


def wilson_confidence_interval(wins, total, z=1.96):
    """95% confidence interval for a win-rate proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = wins / total
    denom = 1 + z**2 / total
    centre = p + z**2 / (2 * total)
    adj = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    lower = (centre - adj) / denom
    upper = (centre + adj) / denom
    return (round(lower * 100, 1), round(upper * 100, 1))


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY as an environment variable before running.")
    client = OpenAI(api_key=api_key)

    base_model, dpo_model, tokenizer = load_models()

    print("Loading held-out eval prompts...")
    eval_dataset = load_dataset("json", data_files=EVAL_FILE, split="train")

    results = []
    dpo_wins = 0
    base_wins = 0
    ties = 0

    for i, row in enumerate(eval_dataset):
        prompt = row["prompt"]
        print(f"\n[{i+1}/{len(eval_dataset)}] Generating responses...")

        base_response = generate_response(base_model, tokenizer, prompt)
        dpo_response = generate_response(dpo_model, tokenizer, prompt)

        if random.random() < 0.5:
            response_a, response_b = base_response, dpo_response
            a_is_dpo = False
        else:
            response_a, response_b = dpo_response, base_response
            a_is_dpo = True

        verdict = judge_pair(client, prompt, response_a, response_b)
        winner_label = verdict.get("winner", "tie")

        if winner_label == "A":
            actual_winner = "dpo" if a_is_dpo else "base"
        elif winner_label == "B":
            actual_winner = "base" if a_is_dpo else "dpo"
        else:
            actual_winner = "tie"

        if actual_winner == "dpo":
            dpo_wins += 1
        elif actual_winner == "base":
            base_wins += 1
        else:
            ties += 1

        results.append({
            "prompt": prompt,
            "base_response": base_response,
            "dpo_response": dpo_response,
            "winner": actual_winner,
            "reason": verdict.get("reason", ""),
            "confidence": verdict.get("confidence", 0.0),
        })

    total = len(eval_dataset)
    dpo_win_rate = round(dpo_wins / total * 100, 1)
    base_win_rate = round(base_wins / total * 100, 1)
    tie_rate = round(ties / total * 100, 1)
    ci_low, ci_high = wilson_confidence_interval(dpo_wins, total)

    summary = {
        "total_examples": total,
        "dpo_wins": dpo_wins,
        "base_wins": base_wins,
        "ties": ties,
        "dpo_win_rate_pct": dpo_win_rate,
        "base_win_rate_pct": base_win_rate,
        "tie_rate_pct": tie_rate,
        "dpo_win_rate_95_ci": f"[{ci_low}%, {ci_high}%]",
    }

    output = {"summary": summary, "results": results}

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nFull results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
