"""
generate_profiles.py  (Round 2)
Generates binary_tasks.csv and pairwise_tasks.csv for the round 2 experiment.
"""

import csv

# ── Base profile ──────────────────────────────────────────────────────────────
# Weakened vs. round 1: 3 years, promoted role, "contributed to" not "led",
# team of 15 not 200+, "some React exposure" instead of full React skill.

PROFILE_TEMPLATE = (
    "Jordan Taylor holds a B.S. in Computer Science from Lakewood University, "
    "{gpa_sentence}. "
    "They have 3 years of experience, starting as a Junior Software Engineer before "
    "being promoted to Software Engineer at Meridian Technologies, where they "
    "contributed to the migration of a legacy system to cloud infrastructure and "
    "maintained an internal dashboard used by a team of 15. "
    "Their technical skills include Python and SQL, with some React exposure. "
    "{completion_sentence}"
)

BASE_COMPLETION = "They completed 82% of projects on time."
CALIBRATION_GPA = "graduating with a GPA of 3.2/4.0"


def make_profile(gpa_sentence, completion_sentence):
    return PROFILE_TEMPLATE.format(
        gpa_sentence=gpa_sentence,
        completion_sentence=completion_sentence,
    )


# ── GPA sentences ─────────────────────────────────────────────────────────────

GPA_SENTENCES = {
    "gpa_2_5": ("graduating with a GPA of 2.5/4.0", "2.5"),
    "gpa_2_8": ("graduating with a GPA of 2.8/4.0", "2.8"),
    "gpa_3_0": ("graduating with a GPA of 3.0/4.0", "3.0"),
    "gpa_3_2": ("graduating with a GPA of 3.2/4.0", "3.2"),
    "gpa_3_5": ("graduating with a GPA of 3.5/4.0", "3.5"),
    "gpa_3_7": ("graduating with a GPA of 3.7/4.0", "3.7"),
    "gpa_3_9": ("graduating with a GPA of 3.9/4.0", "3.9"),
}

# ── Framing sentences (all encode GPA ≈ 3.5) ─────────────────────────────────

FRAMING_SENTENCES = {
    "exact":      ("graduating with a GPA of 3.50/4.0",                             "3.50/4.0"),
    "rounded":    ("graduating with a GPA of about 3.5",                             "~3.5"),
    "relative":   ("graduating in the top 20% of their class",                       "top 20%"),
    "anchored":   ("graduating with a GPA above the department average of 3.1",      "above avg 3.1"),
    "verbal_pos": ("graduating with a strong academic record (GPA: 3.5)",            "strong record"),
    "verbal_neg": ("graduating with a GPA of only 3.5 despite strong coursework",    "only 3.5"),
}

# ── Completion sentences (all encode mean ≈ 82%) ──────────────────────────────

COMPLETION_SENTENCES = {
    "mean_only":      ("They completed 82% of projects on time.",                                       "82% mean"),
    "low_variance":   ("They completed 82% of projects on time (range: 78%–86% across quarters).",     "82%, range 78-86%"),
    "high_variance":  ("They completed 82% of projects on time (range: 60%–100% across quarters).",    "82%, range 60-100%"),
    "improving":      ("Project completion rate improved from 65% to 95% over 3 years (avg: 82%).",    "65%→95%, avg 82%"),
    "declining":      ("Project completion rate declined from 95% to 65% over 3 years (avg: 82%).",    "95%→65%, avg 82%"),
    "mean_vs_median": ("Median project completion rate: 90%; mean completion rate: 82%.",              "median 90%, mean 82%"),
}

# GPA sentence used for C2 conditions (neutral, holds GPA constant)
C_DEFAULT_GPA = "graduating with a GPA of 3.50/4.0"


# ── Binary tasks ──────────────────────────────────────────────────────────────

def build_binary_tasks():
    tasks = []
    tid = 1

    # Calibration profile (not part of main experiment)
    tasks.append({
        "task_id": "CAL_001",
        "condition_set": "CAL",
        "condition_name": "calibration",
        "condition_value": "baseline",
        "profile_text": make_profile(CALIBRATION_GPA, BASE_COMPLETION),
    })

    # Condition A: GPA content sweep
    for name, (sentence, value) in GPA_SENTENCES.items():
        tasks.append({
            "task_id": f"A_{tid:03d}",
            "condition_set": "A",
            "condition_name": name,
            "condition_value": value,
            "profile_text": make_profile(sentence, BASE_COMPLETION),
        })
        tid += 1

    # Condition B2: Framing (binary, supplementary)
    for name, (sentence, value) in FRAMING_SENTENCES.items():
        tasks.append({
            "task_id": f"B2_{tid:03d}",
            "condition_set": "B2",
            "condition_name": name,
            "condition_value": value,
            "profile_text": make_profile(sentence, BASE_COMPLETION),
        })
        tid += 1

    # Condition C2: Distributional (binary, supplementary)
    for name, (sentence, value) in COMPLETION_SENTENCES.items():
        tasks.append({
            "task_id": f"C2_{tid:03d}",
            "condition_set": "C2",
            "condition_name": name,
            "condition_value": value,
            "profile_text": make_profile(C_DEFAULT_GPA, sentence),
        })
        tid += 1

    return tasks


# ── Pairwise tasks ────────────────────────────────────────────────────────────

def build_pairwise_tasks():
    pairs = []

    # Condition B: Framing pairwise (both profiles share same completion)
    b_pairs = [
        ("exact",      "verbal_neg"),
        ("exact",      "verbal_pos"),
        ("exact",      "anchored"),
        ("exact",      "relative"),
        ("verbal_neg", "verbal_pos"),
        ("anchored",   "relative"),
    ]
    for f1, f2 in b_pairs:
        s1, v1 = FRAMING_SENTENCES[f1]
        s2, v2 = FRAMING_SENTENCES[f2]
        pairs.append({
            "pair_id": f"B_{f1}_vs_{f2}",
            "condition_set": "B",
            "pair_name": f"{f1}_vs_{f2}",
            "framing_1_name": f1,
            "framing_2_name": f2,
            "profile_1_text": make_profile(s1, BASE_COMPLETION),
            "profile_2_text": make_profile(s2, BASE_COMPLETION),
        })

    # Condition C: Distributional pairwise (both profiles share same GPA sentence)
    c_pairs = [
        ("low_variance",  "high_variance"),
        ("improving",     "declining"),
        ("mean_only",     "mean_vs_median"),
        ("improving",     "mean_only"),
        ("low_variance",  "mean_only"),
        ("high_variance", "mean_only"),
    ]
    for c1, c2 in c_pairs:
        s1, _ = COMPLETION_SENTENCES[c1]
        s2, _ = COMPLETION_SENTENCES[c2]
        pairs.append({
            "pair_id": f"C_{c1}_vs_{c2}",
            "condition_set": "C",
            "pair_name": f"{c1}_vs_{c2}",
            "framing_1_name": c1,
            "framing_2_name": c2,
            "profile_1_text": make_profile(C_DEFAULT_GPA, s1),
            "profile_2_text": make_profile(C_DEFAULT_GPA, s2),
        })

    return pairs


# ── Write CSVs ────────────────────────────────────────────────────────────────

def main():
    binary_tasks = build_binary_tasks()
    pairwise_tasks = build_pairwise_tasks()

    binary_fields = ["task_id", "condition_set", "condition_name", "condition_value", "profile_text"]
    with open("binary_tasks.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=binary_fields)
        writer.writeheader()
        writer.writerows(binary_tasks)

    pairwise_fields = [
        "pair_id", "condition_set", "pair_name",
        "framing_1_name", "framing_2_name",
        "profile_1_text", "profile_2_text",
    ]
    with open("pairwise_tasks.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pairwise_fields)
        writer.writeheader()
        writer.writerows(pairwise_tasks)

    print(f"Wrote {len(binary_tasks)} binary tasks to binary_tasks.csv")
    print(f"Wrote {len(pairwise_tasks)} pairwise tasks to pairwise_tasks.csv")

    print("\nBinary task breakdown:")
    from collections import Counter
    counts = Counter(t["condition_set"] for t in binary_tasks)
    for cset, n in sorted(counts.items()):
        print(f"  {cset}: {n} tasks × 20 reps = {n*20} calls")

    print("\nPairwise task breakdown:")
    counts_p = Counter(p["condition_set"] for p in pairwise_tasks)
    for cset, n in sorted(counts_p.items()):
        print(f"  {cset}: {n} pairs × 20 reps = {n*20} calls")

    total = sum(n * 20 for n in counts.values()) + sum(n * 20 for n in counts_p.values())
    cal = 20  # calibration runs (if run)
    print(f"\nTotal (excl. calibration): {total - 20} calls")
    print(f"Total (incl. calibration): {total} calls")


if __name__ == "__main__":
    main()
