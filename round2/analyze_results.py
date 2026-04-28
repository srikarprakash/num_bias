"""
analyze_results.py  (Round 2)
Reads results_binary.csv and results_pairwise.csv.
Outputs: results_round2.png, analysis_stats_r2.json
"""

import json
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import expit

warnings.filterwarnings("ignore")

# Colorblind-friendly palette (Wong 2011)
C = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#000000",
}
PALETTE = [C["blue"], C["orange"], C["green"], C["red"], C["purple"], C["sky"]]

GPA_ORDER = ["gpa_2_5", "gpa_2_8", "gpa_3_0", "gpa_3_2", "gpa_3_5", "gpa_3_7", "gpa_3_9"]
GPA_VALUES = [2.5, 2.8, 3.0, 3.2, 3.5, 3.7, 3.9]

FRAMING_ORDER = ["exact", "rounded", "relative", "anchored", "verbal_pos", "verbal_neg"]
DIST_ORDER    = ["mean_only", "low_variance", "high_variance", "improving", "declining", "mean_vs_median"]

B_PAIRS = [
    "exact_vs_verbal_neg", "exact_vs_verbal_pos", "exact_vs_anchored",
    "exact_vs_relative", "verbal_neg_vs_verbal_pos", "anchored_vs_relative",
]
C_PAIRS = [
    "low_variance_vs_high_variance", "improving_vs_declining",
    "mean_only_vs_mean_vs_median", "improving_vs_mean_only",
    "low_variance_vs_mean_only", "high_variance_vs_mean_only",
]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_binary(path="results_binary.csv"):
    df = pd.read_csv(path)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["valid"] = df["valid"].astype(str).str.lower() == "true"
    df["yes"] = (df["decision"].str.upper() == "YES") & df["valid"]
    return df[df["valid"]]


def load_pairwise(path="results_pairwise.csv"):
    df = pd.read_csv(path)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["valid"] = df["valid"].astype(str).str.lower() == "true"
    return df[df["valid"]]


# ── Analysis A: Psychometric curve ───────────────────────────────────────────

def logistic(x, a, b):
    return expit(a * x + b)


def bootstrap_ci(data, n_boot=2000, ci=0.95):
    """Bootstrap 95% CI on proportion of True values."""
    n = len(data)
    if n == 0:
        return float("nan"), float("nan")
    boot = [np.mean(np.random.choice(data, n, replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boot, (1 - ci) / 2 * 100)
    hi = np.percentile(boot, (1 + ci) / 2 * 100)
    return lo, hi


def analysis_a(df_binary, ax):
    sub = df_binary[df_binary["condition_set"] == "A"]
    adv_rates, lo_cis, hi_cis, yes_conf, no_conf = [], [], [], [], []

    for name in GPA_ORDER:
        rows = sub[sub["condition_name"] == name]
        if len(rows) == 0:
            adv_rates.append(np.nan); lo_cis.append(np.nan); hi_cis.append(np.nan)
            yes_conf.append(np.nan); no_conf.append(np.nan)
            continue
        rate = rows["yes"].mean()
        lo, hi = bootstrap_ci(rows["yes"].values)
        adv_rates.append(rate)
        lo_cis.append(lo)
        hi_cis.append(hi)
        yes_conf.append(rows.loc[rows["yes"], "confidence"].mean())
        no_conf.append(rows.loc[~rows["yes"], "confidence"].mean())

    gpa_arr  = np.array(GPA_VALUES)
    rate_arr = np.array(adv_rates)
    lo_arr   = np.array(lo_cis)
    hi_arr   = np.array(hi_cis)

    # Error bars from bootstrap CIs
    valid = ~np.isnan(rate_arr)
    ax.errorbar(
        gpa_arr[valid], rate_arr[valid],
        yerr=[rate_arr[valid] - lo_arr[valid], hi_arr[valid] - rate_arr[valid]],
        fmt="o", color=C["blue"], linewidth=2, markersize=7, capsize=5,
        label="Advancement rate (95% CI)",
    )

    # Logistic fit
    result_stats = {}
    if valid.sum() >= 3:
        try:
            popt, _ = curve_fit(logistic, gpa_arr[valid], rate_arr[valid],
                                 p0=[2.0, -6.0], maxfev=10000)
            a_fit, b_fit = popt
            x_fit = np.linspace(2.3, 4.1, 200)
            y_fit = logistic(x_fit, a_fit, b_fit)
            ax.plot(x_fit, y_fit, "--", color=C["red"], linewidth=1.8,
                    label=f"Logistic fit")
            threshold_gpa = -b_fit / a_fit
            ax.axvline(threshold_gpa, color=C["red"], alpha=0.4, linewidth=1,
                       linestyle=":")
            ax.text(threshold_gpa + 0.03, 0.08, f"threshold\n{threshold_gpa:.2f}",
                    color=C["red"], fontsize=7)
            result_stats = {"slope": a_fit, "intercept": b_fit, "threshold_gpa": threshold_gpa}
        except Exception:
            pass

    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("GPA", fontsize=10)
    ax.set_ylabel("Advancement rate", fontsize=10)
    ax.set_title("A: Psychometric Curve\n(Binary classification)", fontsize=10, fontweight="bold")
    ax.set_xlim(2.3, 4.1)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    return {
        "advancement_rates": dict(zip(GPA_ORDER, adv_rates)),
        "logistic_fit": result_stats,
    }


# ── Analysis B: Framing pairwise ──────────────────────────────────────────────

def position_adjusted_win_rate(df_pair, pair_name, framing_name):
    """Win rate adjusted for position bias: average over A-position and B-position trials."""
    sub = df_pair[df_pair["pair_name"] == pair_name]
    if len(sub) == 0:
        return float("nan"), 0

    as_a = sub[sub["position_a"] == framing_name]
    as_b = sub[sub["position_a"] != framing_name]

    wins_as_a = (as_a["winner_framing"] == framing_name).mean() if len(as_a) else float("nan")
    wins_as_b = (as_b["winner_framing"] == framing_name).mean() if len(as_b) else float("nan")

    rates = [r for r in [wins_as_a, wins_as_b] if not np.isnan(r)]
    adj_rate = np.mean(rates) if rates else float("nan")
    n = len(sub)
    return adj_rate, n


def binomial_test(n_wins, n_total, expected=0.5):
    if n_total == 0:
        return float("nan")
    result = stats.binomtest(int(round(n_wins)), n_total, expected, alternative="two-sided")
    return result.pvalue


def analysis_b_pairwise(df_pair, ax):
    sub = df_pair[df_pair["condition_set"] == "B"]
    pair_names, win_rates, p_vals, ns = [], [], [], []

    for pname in B_PAIRS:
        rows = sub[sub["pair_name"] == pname]
        if len(rows) == 0:
            continue
        f1 = rows["framing_1_name"].iloc[0]
        rate, n = position_adjusted_win_rate(sub, pname, f1)
        p = binomial_test(rate * n, n)
        pair_names.append(pname.replace("_vs_", " vs\n"))
        win_rates.append(rate)
        p_vals.append(p)
        ns.append(n)

    y = np.arange(len(pair_names))
    colors = [C["blue"] if r > 0.5 else C["orange"] for r in win_rates]
    ax.barh(y, win_rates, color=colors, alpha=0.8)
    ax.axvline(0.5, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(pair_names, fontsize=7)
    ax.set_xlabel("Win rate (framing 1, position-adjusted)", fontsize=9)
    ax.set_title("B: Framing Pairwise\nWin rates (null = 0.5)", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="x", alpha=0.25)

    for i, (r, p, n) in enumerate(zip(win_rates, p_vals, ns)):
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.text(r + 0.01, i, f"{r:.0%} {sig}", va="center", fontsize=7)

    return {"pairs": dict(zip(
        [pn.replace("\n", " ") for pn in pair_names],
        [{"win_rate_f1": r, "p_value": p, "n": n} for r, p, n in zip(win_rates, p_vals, ns)]
    ))}


# ── Analysis C: Distributional pairwise ──────────────────────────────────────

def analysis_c_pairwise(df_pair, ax):
    sub = df_pair[df_pair["condition_set"] == "C"]
    pair_names, win_rates, p_vals, ns = [], [], [], []

    for pname in C_PAIRS:
        rows = sub[sub["pair_name"] == pname]
        if len(rows) == 0:
            continue
        f1 = rows["framing_1_name"].iloc[0]
        rate, n = position_adjusted_win_rate(sub, pname, f1)
        p = binomial_test(rate * n, n)
        pair_names.append(pname.replace("_vs_", " vs\n"))
        win_rates.append(rate)
        p_vals.append(p)
        ns.append(n)

    y = np.arange(len(pair_names))
    colors = [C["green"] if r > 0.5 else C["red"] for r in win_rates]
    ax.barh(y, win_rates, color=colors, alpha=0.8)
    ax.axvline(0.5, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(pair_names, fontsize=7)
    ax.set_xlabel("Win rate (cond 1, position-adjusted)", fontsize=9)
    ax.set_title("C: Distributional Pairwise\nWin rates (null = 0.5)", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="x", alpha=0.25)

    for i, (r, p, n) in enumerate(zip(win_rates, p_vals, ns)):
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.text(r + 0.01, i, f"{r:.0%} {sig}", va="center", fontsize=7)

    return {"pairs": dict(zip(
        [pn.replace("\n", " ") for pn in pair_names],
        [{"win_rate_cond1": r, "p_value": p, "n": n} for r, p, n in zip(win_rates, p_vals, ns)]
    ))}


# ── Analysis C2: Distributional binary ───────────────────────────────────────

def analysis_c2_binary(df_binary, ax):
    sub = df_binary[df_binary["condition_set"] == "C2"]
    labels, means, cis_lo, cis_hi = [], [], [], []

    for cname in DIST_ORDER:
        rows = sub[sub["condition_name"] == cname]
        if len(rows) == 0:
            continue
        rate = rows["yes"].mean()
        lo, hi = bootstrap_ci(rows["yes"].values)
        labels.append(cname)
        means.append(rate)
        cis_lo.append(lo)
        cis_hi.append(hi)

    x = np.arange(len(labels))
    errs = [np.array(means) - np.array(cis_lo), np.array(cis_hi) - np.array(means)]
    ax.bar(x, means, color=PALETTE[:len(labels)], alpha=0.85, width=0.6)
    ax.errorbar(x, means, yerr=errs, fmt="none", color="black", capsize=4, linewidth=1.5)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Advancement rate", fontsize=9)
    ax.set_title("C2: Distributional Binary\nAdvancement rates (95% CI)", fontsize=10, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", alpha=0.25)

    # Annotate
    for xi, (m, lo, hi) in enumerate(zip(means, cis_lo, cis_hi)):
        ax.text(xi, hi + 0.03, f"{m:.0%}", ha="center", fontsize=7)

    return {"conditions": dict(zip(labels, [
        {"advancement_rate": m, "ci_lo": lo, "ci_hi": hi}
        for m, lo, hi in zip(means, cis_lo, cis_hi)
    ]))}


# ── Analysis D: Reasoning text ────────────────────────────────────────────────

CORRECT_ATTR_KEYWORDS = {
    "B": ["gpa", "grade", "academic", "top", "percentile", "average", "only", "strong record"],
    "C": ["variance", "range", "trend", "trajectory", "median", "mean", "improve", "decline",
          "consistent", "inconsistent"],
}
CONFAB_KEYWORDS = ["leadership", "communication", "team player", "interpersonal", "culture"]

def analysis_d(df_binary, df_pairwise):
    results = {}

    # Pairwise reasoning classification
    for cset in ["B", "C"]:
        sub = df_pairwise[df_pairwise["condition_set"] == cset]
        rows_out = []
        for _, row in sub.iterrows():
            r = str(row.get("reasoning", "")).lower()
            correct = any(kw in r for kw in CORRECT_ATTR_KEYWORDS.get(cset, []))
            confab  = any(kw in r for kw in CONFAB_KEYWORDS)
            equiv   = any(ph in r for ph in ["both candidates", "very similar", "identical", "same"])
            rows_out.append({
                "pair_name": row["pair_name"],
                "winner": row["winner_framing"],
                "correct_attribution": correct,
                "confabulation": confab,
                "equivalence_noted": equiv,
                "snippet": str(row.get("reasoning", ""))[:150],
            })
        sub_df = pd.DataFrame(rows_out)
        if len(sub_df):
            results[f"pairwise_{cset}"] = {
                "pct_correct_attribution": sub_df["correct_attribution"].mean(),
                "pct_confabulation":       sub_df["confabulation"].mean(),
                "pct_equivalence_noted":   sub_df["equivalence_noted"].mean(),
                "n": len(sub_df),
            }

    # Binary reasoning for C2
    sub_c2 = df_binary[df_binary["condition_set"] == "C2"]
    c2_results = {}
    for cname in DIST_ORDER:
        rows = sub_c2[sub_c2["condition_name"] == cname]
        if len(rows) == 0:
            continue
        snippets = rows["reasoning"].dropna().tolist()
        pct_mention = np.mean([
            any(kw in str(s).lower() for kw in CORRECT_ATTR_KEYWORDS["C"])
            for s in snippets
        ])
        c2_results[cname] = {
            "n": len(rows),
            "advancement_rate": rows["yes"].mean(),
            "pct_feature_mentioned": pct_mention,
        }
    results["binary_C2"] = c2_results

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df_binary   = load_binary()
    df_pairwise = load_pairwise()

    print(f"Loaded {len(df_binary)} valid binary rows, {len(df_pairwise)} valid pairwise rows.")
    print("\nBinary condition breakdown:")
    print(df_binary.groupby("condition_set")["yes"].agg(["count", "mean"]).to_string())

    plt.style.use("seaborn-v0_8-white")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "Round 2: Numerical Sensitivity in LLM Resume Judges\n(GPT-4o-mini, binary + pairwise, temp=0.7)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    stats_a  = analysis_a(df_binary, axes[0, 0])
    stats_b  = analysis_b_pairwise(df_pairwise, axes[0, 1])
    stats_c  = analysis_c_pairwise(df_pairwise, axes[1, 0])
    stats_c2 = analysis_c2_binary(df_binary, axes[1, 1])
    stats_d  = analysis_d(df_binary, df_pairwise)

    plt.tight_layout()
    fig.savefig("results_round2.png", dpi=150, bbox_inches="tight")
    print("\nSaved results_round2.png")

    all_stats = {
        "analysis_a": stats_a,
        "analysis_b_pairwise": stats_b,
        "analysis_c_pairwise": stats_c,
        "analysis_c2_binary":  stats_c2,
        "analysis_d_reasoning": stats_d,
    }
    with open("analysis_stats_r2.json", "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print("Saved analysis_stats_r2.json")

    # Print pairwise summary tables
    print("\n── Framing Pairwise (B) ────────────────────────────────────────")
    for pname, row in stats_b.get("pairs", {}).items():
        sig = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else "ns"))
        print(f"  {pname:<40} win={row['win_rate_f1']:.0%}  p={row['p_value']:.3f} {sig}")

    print("\n── Distributional Pairwise (C) ─────────────────────────────────")
    for pname, row in stats_c.get("pairs", {}).items():
        sig = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else "ns"))
        print(f"  {pname:<44} win={row['win_rate_cond1']:.0%}  p={row['p_value']:.3f} {sig}")

    print("\n── Reasoning Attribution (pairwise) ─────────────────────────────")
    for key, val in stats_d.items():
        if "pairwise" in key:
            print(f"  {key}: correct_attr={val['pct_correct_attribution']:.0%}  "
                  f"confab={val['pct_confabulation']:.0%}  "
                  f"equiv_noted={val['pct_equivalence_noted']:.0%}")


if __name__ == "__main__":
    main()
