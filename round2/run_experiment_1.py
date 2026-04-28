"""
run_experiment_local.py  (LOCAL GPU VERSION)

Runs binary and pairwise LLM judging tasks using a local open-source model
(DeepSeek, LLaMA, Mistral, etc.) via Hugging Face Transformers.

Usage:
  python run_experiment_local.py --calibrate
  python run_experiment_local.py --binary
  python run_experiment_local.py --pairwise
  python run_experiment_local.py

Reads:
  binary_tasks.csv
  pairwise_tasks.csv

Writes:
  results_binary.csv
  results_pairwise.csv
  raw_responses_r2.jsonl
"""

import argparse
import csv
import json
import random
import re
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"  # Change to any HF model
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 300
N_REPS = 20
CALIBRATION_REPS = 20

# Set to True if GPU memory is limited (<16GB VRAM)
USE_4BIT = False


# ─────────────────────────────────────────────────────────────
# PROMPTS (UNCHANGED LOGIC)
# ─────────────────────────────────────────────────────────────

BINARY_SYSTEM = """You are an experienced technical hiring manager screening resumes for
a mid-level Software Engineer position at a competitive tech company.

Based on the candidate profile below, would you advance this candidate
to a first-round interview?

Respond with a JSON object:
{"decision": "YES" or "NO", "confidence": <0-100>, "reasoning": "<brief explanation>"}"""

PAIRWISE_SYSTEM = """You are an experienced technical hiring manager. You can advance only
one of the following two candidates to a first-round interview for a
mid-level Software Engineer position.

Which candidate would you advance? You must choose one.

Respond with a JSON object:
{"choice": "A" or "B", "confidence": <50-100>, "reasoning": "<brief explanation>"}"""


# ─────────────────────────────────────────────────────────────
# MODEL LOADING (LOCAL GPU)
# ─────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading model: {MODEL_NAME}")

    if USE_4BIT:
        print("Using 4-bit quantization (low VRAM mode)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            quantization_config=bnb_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# GENERATION HELPER
# ─────────────────────────────────────────────────────────────

def generate_response(model, tokenizer, system_prompt, user_prompt):
    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt portion
    generated = text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    return generated.strip()


# ─────────────────────────────────────────────────────────────
# ROBUST JSON PARSER (Fixes messy outputs)
# ─────────────────────────────────────────────────────────────

def extract_json(text):
    try:
        return json.loads(text)
    except:
        pass

    # Try to extract first {...} block
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return {}


# ─────────────────────────────────────────────────────────────
# BINARY TASK
# ─────────────────────────────────────────────────────────────

def run_binary_call(model, tokenizer, task, run_number, raw_log):
    raw = generate_response(model, tokenizer, BINARY_SYSTEM, task["profile_text"])
    parsed = extract_json(raw)

    decision = str(parsed.get("decision", "")).strip().upper()
    confidence = float(parsed.get("confidence", -1))
    reasoning = str(parsed.get("reasoning", ""))

    valid = decision in ("YES", "NO")

    row = {
        "task_id": task["task_id"],
        "condition_set": task["condition_set"],
        "condition_name": task["condition_name"],
        "condition_value": task["condition_value"],
        "task_type": "binary",
        "run_number": run_number,
        "decision": decision if valid else "INVALID",
        "confidence": confidence,
        "reasoning": reasoning,
        "valid": valid,
        "model_version": MODEL_NAME,
        "timestamp": datetime.utcnow().isoformat(),
    }

    raw_log.append({**row, "raw": raw, "profile_text": task["profile_text"]})
    return row


def run_binary_tasks(model, tokenizer, tasks, raw_log):
    results = []
    total = len(tasks) * N_REPS
    print(f"Running {total} binary calls...")

    count = 0
    for task in tasks:
        for run in range(1, N_REPS + 1):
            row = run_binary_call(model, tokenizer, task, run, raw_log)
            results.append(row)
            count += 1
            if count % 20 == 0:
                print(f"[{count}/{total}] {row['task_id']} → {row['decision']}")

    return results


# ─────────────────────────────────────────────────────────────
# PAIRWISE TASK
# ─────────────────────────────────────────────────────────────

def make_pairwise_user_msg(profile_a, profile_b):
    return f"Candidate A:\n{profile_a}\n\nCandidate B:\n{profile_b}"


def run_pairwise_tasks(model, tokenizer, pairs, raw_log, seed=42):
    rng = random.Random(seed)
    results = []
    total = len(pairs) * N_REPS
    print(f"Running {total} pairwise calls...")

    count = 0
    for pair in pairs:
        for run in range(1, N_REPS + 1):

            if rng.random() < 0.5:
                position_a = pair["framing_1_name"]
                profile_a = pair["profile_1_text"]
                profile_b = pair["profile_2_text"]
            else:
                position_a = pair["framing_2_name"]
                profile_a = pair["profile_2_text"]
                profile_b = pair["profile_1_text"]

            raw = generate_response(
                model, tokenizer,
                PAIRWISE_SYSTEM,
                make_pairwise_user_msg(profile_a, profile_b)
            )

            parsed = extract_json(raw)

            choice = str(parsed.get("choice", "")).strip().upper()
            confidence = float(parsed.get("confidence", -1))
            reasoning = str(parsed.get("reasoning", ""))

            valid = choice in ("A", "B")

            if valid:
                winner = position_a if choice == "A" else (
                    pair["framing_2_name"]
                    if position_a == pair["framing_1_name"]
                    else pair["framing_1_name"]
                )
            else:
                winner = "INVALID"

            row = {
                "pair_id": pair["pair_id"],
                "condition_set": pair["condition_set"],
                "pair_name": pair["pair_name"],
                "framing_1_name": pair["framing_1_name"],
                "framing_2_name": pair["framing_2_name"],
                "run_number": run,
                "position_a": position_a,
                "choice": choice if valid else "INVALID",
                "winner_framing": winner,
                "confidence": confidence,
                "reasoning": reasoning,
                "valid": valid,
                "model_version": MODEL_NAME,
                "timestamp": datetime.utcnow().isoformat(),
            }

            raw_log.append({**row, "raw": raw})
            results.append(row)

            count += 1
            if count % 20 == 0:
                print(f"[{count}/{total}] {pair['pair_id']} → {choice}")

    return results


# ─────────────────────────────────────────────────────────────
# SAVE HELPERS (UNCHANGED)
# ─────────────────────────────────────────────────────────────

def save_csv(results, path):
    if not results:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} rows to {path}")


def save_raw(raw_log):
    with open("raw_responses_r2.jsonl", "w", encoding="utf-8") as f:
        for entry in raw_log:
            f.write(json.dumps(entry, default=str) + "\n")
    print(f"Saved {len(raw_log)} raw responses")


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--pairwise", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model()
    raw_log = []

    binary_tasks = load_csv("binary_tasks.csv")
    pairwise_tasks = load_csv("pairwise_tasks.csv")

    run_binary = args.binary or not (args.binary or args.pairwise)
    run_pairwise = args.pairwise or not (args.binary or args.pairwise)

    if run_binary:
        non_cal = [t for t in binary_tasks if t["condition_set"] != "CAL"]
        results = run_binary_tasks(model, tokenizer, non_cal, raw_log)
        save_csv(results, "results_binary.csv")

    if run_pairwise:
        results = run_pairwise_tasks(model, tokenizer, pairwise_tasks, raw_log)
        save_csv(results, "results_pairwise.csv")

    save_raw(raw_log)
    print("Done.")


if __name__ == "__main__":
    main()
