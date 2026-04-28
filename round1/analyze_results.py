"""
analyze_results.py
Analyze experiment results and produce figures + summary statistics.
Reads: results.csv
Outputs: results_figure.png, analysis_stats.json
"""

import csv
import json
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

# ── Colorblind-friendly palette (Wong 2011) ──────────────────────────────────
COLORS = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
}
PALETTE = list(COLORS.values())


def load_results(path="results.csv"):
    df = pd.read_csv(path)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def condition_means(df, condition_set, temp=1.0):
    """Return mean and std per condition_name for a given condition set and temperature."""
    sub = df[(df["condition_set"] == condition_set) & (df["temperature"] == temp)]
    return sub.groupby("condition_name")["score"].agg(["mean", "std", "count"]).reset_index()


def cohens_d(a, b):
    """Compute Cohen's d between two arrays."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled_sd = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_sd


def scores_for(df, condition_set, condition_name, temp=1.0):
    mask = (
        (df["condition_set"] == condition_set)
        & (df["condition_name"] == condition_name)
        & (df["temperature"] == temp)
    )
    return df.loc[mask, "score"].values


# ── Analysis A: Content Sensitivity ──────────────────────────────────────────

def analysis_a(df, ax):
    gpa_map = {
        "gpa_2_5": 2.5, "gpa_2_8": 2.8, "gpa_3_0": 3.0,
        "gpa_3_2": 3.2, "gpa_3_5": 3.5, "gpa_3_7": 3.7, "gpa_3_9": 3.9,
    }

    sub_t0  = df[(df["condition_set"] == "A") & (df["temperature"] == 0.0)]
    sub_t1  = df[(df["condition_set"] == "A") & (df["temperature"] == 1.0)]

    gpa_vals, means, stds, det_scores = [], [], [], []
    for cname, gpa in sorted(gpa_map.items(), key=lambda x: x[1]):
        s_t1 = sub_t1[sub_t1["condition_name"] == cname]["score"].values
        s_t0 = sub_t0[sub_t0["condition_name"] == cname]["score"].values
        if len(s_t1) == 0:
            continue
        gpa_vals.append(gpa)
        means.append(np.mean(s_t1))
        stds.append(np.std(s_t1, ddof=1) if len(s_t1) > 1 else 0)
        det_scores.append(float(s_t0[0]) if len(s_t0) > 0 else np.nan)

    gpa_arr = np.array(gpa_vals)
    mean_arr = np.array(means)
    std_arr  = np.array(stds)

    # Error bars (stochastic runs)
    ax.errorbar(gpa_arr, mean_arr, yerr=std_arr, fmt="o-",
                color=COLORS["blue"], linewidth=2, markersize=6,
                capsize=4, label="mean ± std (temp=1.0)")

    # Deterministic scores
    ax.scatter(gpa_arr, det_scores, marker="D", color=COLORS["orange"],
               s=50, zorder=5, label="temp=0 score")

    # Linear regression
    if len(gpa_arr) >= 2 and not np.all(np.isnan(mean_arr)):
        slope, intercept, r, p, se = stats.linregress(gpa_arr, mean_arr)
        x_fit = np.linspace(gpa_arr.min(), gpa_arr.max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, "--",
                color=COLORS["red"], linewidth=1.5,
                label=f"linear fit (R²={r**2:.2f}, p={p:.3f})")

    ax.set_xlabel("GPA", fontsize=10)
    ax.set_ylabel("Judge score", fontsize=10)
    ax.set_title("A: Content Sensitivity", fontsize=11, fontweight="bold")
    ax.set_xlim(2.3, 4.1)
    ax.set_ylim(0, 11)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # Return stats for JSON output
    result = {}
    if len(gpa_arr) >= 2 and not np.all(np.isnan(mean_arr)):
        slope, intercept, r, p, se = stats.linregress(gpa_arr, mean_arr)
        result = {"slope": slope, "intercept": intercept, "r_squared": r**2, "p_value": p}
    return result


# ── Analysis B: Framing Effects ───────────────────────────────────────────────

FRAMING_ORDER = ["exact", "rounded", "relative", "anchored", "verbal_pos", "verbal_neg"]

def analysis_b(df, ax):
    reference = "exact"
    results = {}

    cond_means = []
    cond_errs  = []
    effect_sizes = []
    labels = []
    ref_scores = scores_for(df, "B", reference)

    for cname in FRAMING_ORDER:
        s = scores_for(df, "B", cname)
        if len(s) == 0:
            continue
        m = np.mean(s)
        err = np.std(s, ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0  # SEM
        d = cohens_d(s, ref_scores) if cname != reference else 0.0
        labels.append(cname)
        cond_means.append(m)
        cond_errs.append(err)
        effect_sizes.append(d)
        results[cname] = {"mean": m, "sem": err, "cohens_d_vs_exact": d}

    # ANOVA
    groups = [scores_for(df, "B", c) for c in labels if len(scores_for(df, "B", c)) > 0]
    if len(groups) >= 2:
        f_stat, anova_p = stats.f_oneway(*groups)
        results["anova"] = {"f_stat": f_stat, "p_value": anova_p}
    else:
        anova_p = float("nan")
        results["anova"] = {}

    # Horizontal bar chart of effect sizes
    y = np.arange(len(labels))
    colors = [COLORS["blue"] if d >= 0 else COLORS["red"] for d in effect_sizes]
    bars = ax.barh(y, effect_sizes, color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Cohen's d vs. 'exact'", fontsize=10)
    ax.set_title(
        f"B: Framing Effects\n(ANOVA p={anova_p:.3f})" if not np.isnan(anova_p) else "B: Framing Effects",
        fontsize=11, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)

    # Annotate bars with mean scores
    for i, (bar, m) in enumerate(zip(bars, cond_means)):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"μ={m:.1f}", va="center", fontsize=8)

    return results


# ── Analysis C: Distributional Reasoning ─────────────────────────────────────

DIST_ORDER = ["mean_only", "low_variance", "high_variance", "improving", "declining", "mean_vs_median"]

def analysis_c(df, ax):
    results = {}
    labels, means, errs = [], [], []

    for cname in DIST_ORDER:
        s = scores_for(df, "C", cname)
        if len(s) == 0:
            continue
        m = np.mean(s)
        err = np.std(s, ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0
        labels.append(cname)
        means.append(m)
        errs.append(err)
        results[cname] = {"mean": m, "sem": err}

    x = np.arange(len(labels))
    color_list = PALETTE[:len(labels)]
    bars = ax.bar(x, means, yerr=errs, color=color_list, alpha=0.85,
                  capsize=4, error_kw={"linewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Judge score", fontsize=10)
    ax.set_title("C: Distributional Reasoning", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 11)
    ax.grid(axis="y", alpha=0.3)

    # Hypothesis tests
    hypotheses = [
        ("H1", "low_variance",  "high_variance",  "low_var > high_var"),
        ("H2", "improving",     "declining",       "improving > declining"),
        ("H3", "mean_vs_median","mean_only",       "median info ≠ mean_only"),
        ("H4", "improving",     "mean_only",       "improving > mean_only"),
    ]
    hyp_results = {}
    for hname, cond1, cond2, desc in hypotheses:
        a = scores_for(df, "C", cond1)
        b = scores_for(df, "C", cond2)
        if len(a) < 2 or len(b) < 2:
            hyp_results[hname] = {"description": desc, "note": "insufficient data"}
            continue
        t_stat, p_val = stats.ttest_ind(a, b, alternative="two-sided")
        d = cohens_d(a, b)
        hyp_results[hname] = {
            "description": desc,
            "mean_cond1": float(np.mean(a)),
            "mean_cond2": float(np.mean(b)),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(d),
            "supported": bool(p_val < 0.05 and np.mean(a) > np.mean(b))
            if hname != "H3" else bool(p_val < 0.05),
        }
    results["hypotheses"] = hyp_results
    return results


# ── Analysis D: Reasoning Text ────────────────────────────────────────────────

KEYWORDS = ["strong", "solid", "concerning", "impressive", "excellent",
            "weak", "good", "average", "outstanding", "low", "high",
            "variance", "trend", "trajectory", "distribution", "median",
            "range", "consistent", "inconsistent", "decline", "improve"]

FEATURE_MENTIONS = {
    "A": ["gpa", "grade", "academic"],
    "B": ["gpa", "grade", "academic", "top", "average", "strong", "only"],
    "C": ["variance", "range", "trend", "trajectory", "median", "mean", "completion", "improve", "decline"],
}

def analysis_d(df):
    results = {}
    for cset in ["A", "B", "C"]:
        sub = df[(df["condition_set"] == cset) & (df["temperature"] == 0.0)]
        cset_results = {}
        for _, row in sub.iterrows():
            reasoning = str(row.get("reasoning", "")).lower()
            kw_found = {kw: (kw in reasoning) for kw in KEYWORDS}
            feature_mentioned = any(fm in reasoning for fm in FEATURE_MENTIONS.get(cset, []))
            cset_results[row["condition_name"]] = {
                "score": row["score"],
                "reasoning_snippet": str(row.get("reasoning", ""))[:200],
                "feature_mentioned_in_reasoning": feature_mentioned,
                "keywords": {k: v for k, v in kw_found.items() if v},
            }
        results[cset] = cset_results
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_results()
    print(f"Loaded {len(df)} rows from results.csv")
    print(df.groupby(["condition_set", "temperature"])["score"].describe().to_string())

    plt.style.use("seaborn-v0_8-white")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Numerical Sensitivity in LLM Resume Judges (GPT-4o-mini)",
        fontsize=13, fontweight="bold", y=1.02
    )

    stats_a = analysis_a(df, axes[0])
    stats_b = analysis_b(df, axes[1])
    stats_c = analysis_c(df, axes[2])
    stats_d = analysis_d(df)

    plt.tight_layout()
    fig.savefig("results_figure.png", dpi=150, bbox_inches="tight")
    print("Saved results_figure.png")

    all_stats = {
        "analysis_a_regression": stats_a,
        "analysis_b_framing":    stats_b,
        "analysis_c_distributional": stats_c,
        "analysis_d_reasoning":  stats_d,
    }
    with open("analysis_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print("Saved analysis_stats.json")

    # Print hypothesis summary
    print("\n── Hypothesis Tests (Condition C) ─────────────────────────────")
    hyps = stats_c.get("hypotheses", {})
    for hname, hr in hyps.items():
        supported = hr.get("supported", "?")
        p = hr.get("p_value", float("nan"))
        d = hr.get("cohens_d", float("nan"))
        print(f"  {hname}: {hr['description']}")
        print(f"    p={p:.3f}  Cohen's d={d:.2f}  supported={supported}")

    print("\n── Framing Effects (Condition B) ──────────────────────────────")
    for cname, fr in stats_b.items():
        if cname == "anova":
            continue
        d = fr.get("cohens_d_vs_exact", 0)
        m = fr.get("mean", 0)
        print(f"  {cname:<14} mean={m:.2f}  d={d:+.2f}")

    print("\n── Reasoning Coverage (Condition C, temp=0) ───────────────────")
    for cname, cr in stats_d.get("C", {}).items():
        flag = "YES" if cr["feature_mentioned_in_reasoning"] else " NO"
        print(f"  [{flag}] {cname:<18} score={cr['score']}  keywords={list(cr['keywords'].keys())}")


if __name__ == "__main__":
    main()
