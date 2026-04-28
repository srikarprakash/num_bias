"""
run_experiment_local.py  (Round 2 - Local GPU Version)
Runs binary and pairwise LLM judging tasks using local open-source models.

Usage:
  python run_experiment_local.py --calibrate          # calibration check only
  python run_experiment_local.py --binary             # binary tasks only
  python run_experiment_local.py --pairwise           # pairwise tasks only
  python run_experiment_local.py                      # all (binary + pairwise)

Reads:  binary_tasks.csv, pairwise_tasks.csv
Writes: results_binary.csv, results_pairwise.csv, raw_responses_r2.jsonl
"""

import argparse
import csv
import json
import random
import re
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Model Configuration ───────────────────────────────────────────────────────
# Choose one of these models:
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"  # DeepSeek 7B (recommended)
# MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"    # LLaMA 2 7B
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # LLaMA 3 8B

# Inference settings
TEMPERATURE = 0.7
N_REPS = 20
CALIBRATION_REPS = 20
MAX_NEW_TOKENS = 512  # Max tokens for model response
USE_4BIT = False  # Set to True if you have GPU memory issues

# System prompts (same as original)
BINARY_SYSTEM = """\
You are an experienced technical hiring manager screening resumes for
a mid-level Software Engineer position at a competitive tech company.

Based on the candidate profile below, would you advance this candidate
to a first-round interview?

Respond with a JSON object:
{"decision": "YES" or "NO", "confidence": <0-100>, "reasoning": "<brief explanation>"}"""

PAIRWISE_SYSTEM = """\
You are an experienced technical hiring manager. You can advance only
one of the following two candidates to a first-round interview for a
mid-level Software Engineer position.

Which candidate would you advance? You must choose one.

Respond with a JSON object:
{"choice": "A" or "B", "confidence": <50-100>, "reasoning": "<brief explanation>"}"""


# ── Model Initialization ──────────────────────────────────────────────────────

class LocalLLM:
    """Wrapper for local Hugging Face model inference on GPU."""
    
    def __init__(self, model_name, use_4bit=False):
        print(f"Loading model: {model_name}")
        print(f"GPU available: {torch.cuda.is_available()}")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load model with appropriate precision
        if use_4bit:
            # 4-bit quantization for lower memory usage (requires bitsandbytes)
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            print("Model loaded in 4-bit mode")
        else:
            # Standard float16 loading
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",  # Automatically use GPU
                trust_remote_code=True
            )
            print("Model loaded in float16 mode")
        
        self.model.eval()  # Set to evaluation mode
        print(f"Model loaded on device: {self.model.device}")
    
    def generate(self, system_prompt, user_message, temperature=0.7, max_tokens=512):
        """
        Generate response from the model.
        Formats prompt as a chat conversation.
        """
        # Format as chat (DeepSeek/LLaMA chat format)
        if "deepseek" in self.model_name.lower():
            # DeepSeek chat format
            prompt = f"User: {system_prompt}\n\n{user_message}\n\nAssistant:"
        elif "llama-2" in self.model_name.lower():
            # LLaMA 2 chat format
            prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]"
        elif "llama-3" in self.model_name.lower() or "Meta-Llama-3" in self.model_name:
            # LLaMA 3 chat format
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        else:
            # Generic format
            prompt = f"System: {system_prompt}\n\nUser: {user_message}\n\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,  # Use sampling if temperature > 0
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens (exclude input prompt)
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()


# ── Response Parsing ──────────────────────────────────────────────────────────

def extract_json_from_response(response_text):
    """
    Robust JSON extraction from model output.
    Handles cases where model returns text before/after JSON.
    """
    # Try to parse the entire response first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object using regex
    json_match = re.search(r'\{[^{}]*\}', response_text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Try to extract fields manually from text
    decision_match = re.search(r'"decision"\s*:\s*"(YES|NO)"', response_text, re.IGNORECASE)
    choice_match = re.search(r'"choice"\s*:\s*"([AB])"', response_text, re.IGNORECASE)
    confidence_match = re.search(r'"confidence"\s*:\s*(\d+)', response_text)
    
    result = {}
    if decision_match:
        result["decision"] = decision_match.group(1).upper()
    if choice_match:
        result["choice"] = choice_match.group(1).upper()
    if confidence_match:
        result["confidence"] = int(confidence_match.group(1))
    
    if result:
        return result
    
    # Last resort: return empty dict
    return {}


# ── Binary Task ───────────────────────────────────────────────────────────────

def run_binary_call(model, task, run_number, raw_log):
    """
    Run a single binary judgment call.
    NOTE: This is now synchronous (no async) since we're using local inference.
    """
    try:
        # First attempt
        raw = model.generate(BINARY_SYSTEM, task["profile_text"], TEMPERATURE, MAX_NEW_TOKENS)
        parsed = extract_json_from_response(raw)
        
        decision = str(parsed.get("decision", "")).strip().upper()
        confidence = float(parsed.get("confidence", -1))
        reasoning = str(parsed.get("reasoning", ""))
        
        # Retry logic if decision is invalid
        if decision not in ("YES", "NO"):
            retry_msg = task["profile_text"] + "\n\nReminder: respond ONLY with {\"decision\": \"YES\" or \"NO\", ...}"
            raw = model.generate(BINARY_SYSTEM, retry_msg, TEMPERATURE, MAX_NEW_TOKENS)
            parsed = extract_json_from_response(raw)
            decision = str(parsed.get("decision", "INVALID")).strip().upper()
            confidence = float(parsed.get("confidence", -1))
            reasoning = str(parsed.get("reasoning", ""))
        
        valid = decision in ("YES", "NO")
    
    except Exception as e:
        decision, confidence, reasoning, raw = "INVALID", -1, str(e), ""
        valid = False
    
    row = {
        "task_id":        task["task_id"],
        "condition_set":  task["condition_set"],
        "condition_name": task["condition_name"],
        "condition_value":task["condition_value"],
        "task_type":      "binary",
        "run_number":     run_number,
        "decision":       decision,
        "confidence":     confidence,
        "reasoning":      reasoning,
        "valid":          valid,
        "model_version":  model.model_name,
        "timestamp":      datetime.utcnow().isoformat(),
    }
    raw_log.append({**row, "raw": raw, "profile_text": task["profile_text"]})
    return row


def run_binary_tasks(model, tasks, raw_log):
    """
    Run all binary tasks sequentially.
    NOTE: No async/await - runs one at a time on GPU.
    """
    results = []
    total = len(tasks) * N_REPS
    done = 0
    
    print(f"  Running {total} binary calls...")
    
    for task in tasks:
        for run in range(1, N_REPS + 1):
            row = run_binary_call(model, task, run, raw_log)
            results.append(row)
            done += 1
            
            if done % 20 == 0:
                print(f"  [{done}/{total}] last: {row['task_id']} run={row['run_number']} → {row['decision']} ({row['confidence']}%)")
    
    return results


# ── Pairwise Task ─────────────────────────────────────────────────────────────

def make_pairwise_user_msg(profile_a, profile_b):
    return f"Candidate A:\n{profile_a}\n\nCandidate B:\n{profile_b}"


def run_pairwise_call(model, pair, run_number, raw_log, rng):
    """Run a single pairwise comparison call."""
    # Randomize which framing is shown as A vs B
    if rng.random() < 0.5:
        position_a = pair["framing_1_name"]
        profile_a  = pair["profile_1_text"]
        profile_b  = pair["profile_2_text"]
    else:
        position_a = pair["framing_2_name"]
        profile_a  = pair["profile_2_text"]
        profile_b  = pair["profile_1_text"]
    
    user_msg = make_pairwise_user_msg(profile_a, profile_b)
    
    try:
        raw = model.generate(PAIRWISE_SYSTEM, user_msg, TEMPERATURE, MAX_NEW_TOKENS)
        parsed = extract_json_from_response(raw)
        
        choice = str(parsed.get("choice", "")).strip().upper()
        confidence = float(parsed.get("confidence", -1))
        reasoning = str(parsed.get("reasoning", ""))
        
        # Retry logic
        if choice not in ("A", "B"):
            retry_msg = user_msg + "\n\nReminder: respond ONLY with {\"choice\": \"A\" or \"B\", ...}"
            raw = model.generate(PAIRWISE_SYSTEM, retry_msg, TEMPERATURE, MAX_NEW_TOKENS)
            parsed = extract_json_from_response(raw)
            choice = str(parsed.get("choice", "INVALID")).strip().upper()
            confidence = float(parsed.get("confidence", -1))
            reasoning = str(parsed.get("reasoning", ""))
        
        valid = choice in ("A", "B")
        if valid:
            winner = position_a if choice == "A" else (
                pair["framing_2_name"] if position_a == pair["framing_1_name"]
                else pair["framing_1_name"]
            )
        else:
            winner = "INVALID"
    
    except Exception as e:
        choice, confidence, reasoning, raw = "INVALID", -1, str(e), ""
        valid, winner = False, "INVALID"
    
    row = {
        "pair_id":         pair["pair_id"],
        "condition_set":   pair["condition_set"],
        "pair_name":       pair["pair_name"],
        "framing_1_name":  pair["framing_1_name"],
        "framing_2_name":  pair["framing_2_name"],
        "run_number":      run_number,
        "position_a":      position_a,
        "choice":          choice,
        "winner_framing":  winner,
        "confidence":      confidence,
        "reasoning":       reasoning,
        "valid":           valid,
        "model_version":   model.model_name,
        "timestamp":       datetime.utcnow().isoformat(),
    }
    raw_log.append({**row, "raw": raw})
    return row


def run_pairwise_tasks(model, pairs, raw_log, seed=42):
    """Run all pairwise tasks sequentially."""
    rng = random.Random(seed)
    results = []
    total = len(pairs) * N_REPS
    done = 0
    
    print(f"  Running {total} pairwise calls...")
    
    for pair in pairs:
        for run in range(1, N_REPS + 1):
            row = run_pairwise_call(model, pair, run, raw_log, rng)
            results.append(row)
            done += 1
            
            if done % 20 == 0:
                print(f"  [{done}/{total}] {row['pair_id']} run={row['run_number']} → chose {row['choice']} (winner: {row['winner_framing']})")
    
    return results


# ── Calibration ───────────────────────────────────────────────────────────────

def run_calibration(model, binary_tasks, raw_log):
    """Run calibration check on base profile."""
    cal_tasks = [t for t in binary_tasks if t["condition_set"] == "CAL"]
    if not cal_tasks:
        print("No calibration task found in binary_tasks.csv.")
        return
    
    cal_task = cal_tasks[0]
    print(f"Running calibration: {CALIBRATION_REPS} binary calls on base profile...")
    
    results = []
    for run in range(1, CALIBRATION_REPS + 1):
        row = run_binary_call(model, cal_task, run, raw_log)
        results.append(row)
    
    valid = [r for r in results if r["valid"]]
    yes_count = sum(1 for r in valid if r["decision"] == "YES")
    yes_rate = yes_count / len(valid) if valid else float("nan")
    
    print(f"\n── Calibration Result ─────────────────────────────────────────")
    print(f"  Valid responses: {len(valid)} / {CALIBRATION_REPS}")
    print(f"  YES: {yes_count}  NO: {len(valid) - yes_count}")
    print(f"  YES rate: {yes_rate:.1%}")
    
    if 0.35 <= yes_rate <= 0.65:
        print("  ✓ Profile is in the target borderline zone (35%-65%). Good to proceed.")
    elif yes_rate > 0.65:
        print("  ✗ Profile is TOO STRONG (YES rate > 65%). Weaken a non-numerical feature.")
    else:
        print("  ✗ Profile is TOO WEAK (YES rate < 35%). Strengthen a non-numerical feature.")
    
    print()
    return yes_rate


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_binary(results, path="results_binary.csv"):
    if not results:
        return
    fields = [
        "task_id", "condition_set", "condition_name", "condition_value",
        "task_type", "run_number", "decision", "confidence", "reasoning",
        "valid", "model_version", "timestamp",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} rows to {path}")


def save_pairwise(results, path="results_pairwise.csv"):
    if not results:
        return
    fields = [
        "pair_id", "condition_set", "pair_name", "framing_1_name", "framing_2_name",
        "run_number", "position_a", "choice", "winner_framing", "confidence",
        "reasoning", "valid", "model_version", "timestamp",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} rows to {path}")


def save_raw(raw_log, path="raw_responses_r2.jsonl"):
    with open(path, "w", encoding="utf-8") as f:
        for entry in raw_log:
            f.write(json.dumps(entry, default=str) + "\n")
    print(f"Saved {len(raw_log)} raw responses to {path}")


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    """
    Main function - now synchronous (no async).
    Loads model once and runs all tasks sequentially.
    """
    # Initialize local model on GPU
    model = LocalLLM(MODEL_NAME, use_4bit=USE_4BIT)
    raw_log = []
    
    # Load task data
    binary_tasks  = load_csv("binary_tasks.csv")
    pairwise_tasks = load_csv("pairwise_tasks.csv")
    
    # Always run calibration first
    if args.calibrate or not (args.binary or args.pairwise):
        run_calibration(model, binary_tasks, raw_log)
        if args.calibrate:
            save_raw(raw_log, "raw_responses_r2.jsonl")
            return
    
    # Main experiment tasks
    run_binary_tasks_flag   = args.binary   or not (args.binary or args.pairwise)
    run_pairwise_tasks_flag = args.pairwise or not (args.binary or args.pairwise)
    
    binary_results   = []
    pairwise_results = []
    
    if run_binary_tasks_flag:
        non_cal = [t for t in binary_tasks if t["condition_set"] != "CAL"]
        print(f"\nRunning binary tasks ({len(non_cal)} tasks × {N_REPS} reps)...")
        binary_results = run_binary_tasks(model, non_cal, raw_log)
        save_binary(binary_results)
    
    if run_pairwise_tasks_flag:
        print(f"\nRunning pairwise tasks ({len(pairwise_tasks)} pairs × {N_REPS} reps)...")
        pairwise_results = run_pairwise_tasks(model, pairwise_tasks, raw_log)
        save_pairwise(pairwise_results)
    
    save_raw(raw_log)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Round 2 LLM judge experiment (Local GPU).")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run calibration check only (20 calls on base profile).")
    parser.add_argument("--binary",    action="store_true",
                        help="Run binary tasks only.")
    parser.add_argument("--pairwise",  action="store_true",
                        help="Run pairwise tasks only.")
    args = parser.parse_args()
    main(args)  # Note: No asyncio.run() - this is synchronous now
