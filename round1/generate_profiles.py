"""
generate_profiles.py
Generate synthetic resume profiles for the numerical sensitivity experiment.
Outputs: profiles.csv
"""

import csv
import itertools

BASE_GPA_SENTENCE = "Graduated with a GPA of {value}/4.0"
BASE_COMPLETION_SENTENCE = "They completed 82% of projects on time."

PROFILE_TEMPLATE = (
    "Jordan Taylor holds a B.S. in Computer Science from Lakewood University, "
    "{gpa_sentence}. "
    "They have 4 years of experience as a Software Engineer at Meridian Technologies, "
    "where they led the migration of a legacy system to cloud infrastructure and built "
    "an internal dashboard used by over 200 employees. "
    "Their technical skills include Python, SQL, React, and AWS. "
    "{completion_sentence}"
)

# Condition Set A: Numerical Content Sensitivity
CONDITION_A = [
    {"condition_name": f"gpa_{str(v).replace('.', '_')}", "condition_value": str(v),
     "gpa_sentence": f"graduating with a GPA of {v}/4.0",
     "completion_sentence": BASE_COMPLETION_SENTENCE}
    for v in [2.5, 2.8, 3.0, 3.2, 3.5, 3.7, 3.9]
]

# Condition Set B: Numerical Framing Sensitivity (GPA fixed at 3.5)
CONDITION_B = [
    {
        "condition_name": "exact",
        "condition_value": "3.50/4.0",
        "gpa_sentence": "graduating with a GPA of 3.50/4.0",
        "completion_sentence": BASE_COMPLETION_SENTENCE,
    },
    {
        "condition_name": "rounded",
        "condition_value": "about 3.5",
        "gpa_sentence": "graduating with a GPA of about 3.5",
        "completion_sentence": BASE_COMPLETION_SENTENCE,
    },
    {
        "condition_name": "relative",
        "condition_value": "top 20%",
        "gpa_sentence": "graduating in the top 20% of their class",
        "completion_sentence": BASE_COMPLETION_SENTENCE,
    },
    {
        "condition_name": "anchored",
        "condition_value": "above dept avg 3.1",
        "gpa_sentence": "graduating with a GPA above the department average of 3.1",
        "completion_sentence": BASE_COMPLETION_SENTENCE,
    },
    {
        "condition_name": "verbal_pos",
        "condition_value": "strong record (3.5)",
        "gpa_sentence": "graduating with a strong academic record (GPA: 3.5)",
        "completion_sentence": BASE_COMPLETION_SENTENCE,
    },
    {
        "condition_name": "verbal_neg",
        "condition_value": "only 3.5",
        "gpa_sentence": "graduating with a GPA of only 3.5 despite strong coursework",
        "completion_sentence": BASE_COMPLETION_SENTENCE,
    },
]

# Condition Set C: Statistical Reasoning (mean fixed at 82%)
CONDITION_C = [
    {
        "condition_name": "mean_only",
        "condition_value": "82% mean",
        "gpa_sentence": "graduating with a GPA of 3.50/4.0",
        "completion_sentence": "They completed 82% of projects on time.",
    },
    {
        "condition_name": "low_variance",
        "condition_value": "82% mean, range 78-86%",
        "gpa_sentence": "graduating with a GPA of 3.50/4.0",
        "completion_sentence": (
            "They completed 82% of projects on time "
            "(range: 78%-86% across quarters)."
        ),
    },
    {
        "condition_name": "high_variance",
        "condition_value": "82% mean, range 60-100%",
        "gpa_sentence": "graduating with a GPA of 3.50/4.0",
        "completion_sentence": (
            "They completed 82% of projects on time "
            "(range: 60%-100% across quarters)."
        ),
    },
    {
        "condition_name": "improving",
        "condition_value": "65% to 95%, avg 82%",
        "gpa_sentence": "graduating with a GPA of 3.50/4.0",
        "completion_sentence": (
            "Project completion rate improved from 65% to 95% over 4 years (avg: 82%)."
        ),
    },
    {
        "condition_name": "declining",
        "condition_value": "95% to 65%, avg 82%",
        "gpa_sentence": "graduating with a GPA of 3.50/4.0",
        "completion_sentence": (
            "Project completion rate declined from 95% to 65% over 4 years (avg: 82%)."
        ),
    },
    {
        "condition_name": "mean_vs_median",
        "condition_value": "median 90%, mean 82%",
        "gpa_sentence": "graduating with a GPA of 3.50/4.0",
        "completion_sentence": (
            "Median project completion rate: 90%; mean completion rate: 82%."
        ),
    },
]


def build_profiles():
    rows = []
    profile_id = 1

    for condition_set, conditions in [("A", CONDITION_A), ("B", CONDITION_B), ("C", CONDITION_C)]:
        for cond in conditions:
            text = PROFILE_TEMPLATE.format(
                gpa_sentence=cond["gpa_sentence"],
                completion_sentence=cond["completion_sentence"],
            )
            rows.append({
                "profile_id": f"{condition_set}_{profile_id:03d}",
                "condition_set": condition_set,
                "condition_name": cond["condition_name"],
                "condition_value": cond["condition_value"],
                "profile_text": text,
            })
            profile_id += 1

    return rows


def main():
    profiles = build_profiles()
    out_path = "profiles.csv"
    fieldnames = ["profile_id", "condition_set", "condition_name", "condition_value", "profile_text"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(profiles)

    print(f"Wrote {len(profiles)} profiles to {out_path}")
    for p in profiles:
        print(f"  [{p['profile_id']}] {p['condition_set']} / {p['condition_name']}")
        print(f"    {p['profile_text'][:120]}...")
        print()


if __name__ == "__main__":
    main()
