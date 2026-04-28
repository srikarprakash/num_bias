import csv
import json
import re
import random
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# CONFIG
# =========================
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"  # change if needed
TEMP_DETERMINISTIC = 0.0
TEMP_STOCHASTIC = 1.0
N_STOCHASTIC_RUNS = 5
MAX_NEW_TOKENS = 200

# =========================
# LOAD MODEL
# =========================
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"  # uses GPU automatically
)

print("Model loaded.")

# =========================
# PROMPT
# =========================
SYSTEM_PROMPT = """You are an experienced technical hiring manager evaluating candidates
for a mid-level Software Engineer position. Score each candidate on a
scale from 1 to 10.

Return ONLY valid JSON:
{"score": <number>, "reasoning": "<brief explanation>"}
"""

# =========================
# HELPER FUNCTIONS
# =========================
def build_prompt(profile_text):
    return f"{SYSTEM_PROMPT}\n\nCandidate Profile:\n{profile_text}\n\nAnswer:"

def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return {"score": None, "reasoning": "parse_failed"}

def call_judge_local(profile_text, temperature):
    prompt = build_prompt(profile_text)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=temperature,
        do_sample=(temperature > 0),
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # remove prompt part
    response = response[len(prompt):].strip()

    return response

# =========================
# FILE IO
# =========================
def load_profiles(path="profiles.csv"):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save_results(results, path="results.csv"):
    if not results:
        print("No results to save.")
        return

    fieldnames = [
        "profile_id", "condition_set", "condition_name", "condition_value",
        "temperature", "run_number", "score", "reasoning", "model_version", "timestamp",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} rows to {path}")

def save_raw(raw_log, path="raw_responses.jsonl"):
    with open(path, "w", encoding="utf-8") as f:
        for entry in raw_log:
            f.write(json.dumps(entry) + "\n")

    print(f"Saved {len(raw_log)} raw responses to {path}")

# =========================
# MAIN
# =========================
def main():
    random.seed(42)

    profiles = load_profiles()
    print(f"Loaded {len(profiles)} profiles.")

    results = []
    raw_log = []

    total_tasks = len(profiles) * (1 + N_STOCHASTIC_RUNS)
    task_count = 0

    for profile in profiles:
        runs = [(TEMP_DETERMINISTIC, 0)] + [
            (TEMP_STOCHASTIC, i) for i in range(1, N_STOCHASTIC_RUNS + 1)
        ]

        for temperature, run_number in runs:
            task_count += 1

            raw_output = call_judge_local(profile["profile_text"], temperature)
            parsed = extract_json(raw_output)

            score = parsed.get("score")
            try:
                score = float(score)
            except:
                score = None

            row = {
                "profile_id": profile["profile_id"],
                "condition_set": profile["condition_set"],
                "condition_name": profile["condition_name"],
                "condition_value": profile["condition_value"],
                "temperature": temperature,
                "run_number": run_number,
                "score": score,
                "reasoning": parsed.get("reasoning", ""),
                "model_version": MODEL_NAME,
                "timestamp": datetime.utcnow().isoformat(),
            }

            raw_entry = {**row, "raw_response": raw_output}

            results.append(row)
            raw_log.append(raw_entry)

            print(f"[{task_count}/{total_tasks}] Profile {profile['profile_id']} temp={temperature} run={run_number} → score={score}")

    save_results(results)
    save_raw(raw_log)

    print("Done.")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()
