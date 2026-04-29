"""
Stress test - verifies the system gives high scores to matching domains
and low scores to mismatched domains.

Analyzes:
1. Same-domain vs cross-domain score seperation
2. Easy vs hard irrelevant discrimination
3. Per-method threshold analysis (AUC)

Usage:
    python3 src/evaluation/stress_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import OUTPUT_DIR, RESULTS_DIR

DETAILED_SCORES_PATH = RESULTS_DIR / "detailed_scores.csv"


def domain_separation_analysis(df):
    """Check how well each method separtes relevant from irrelevant candidates."""
    print("\n" + "=" * 70)
    print("1. DOMAIN SEPARATION ANALYSIS")
    print("=" * 70)

    methods = [
        ("tfidf_score", "TF-IDF"),
        ("skill_idf_score", "Skill-IDF"),
        ("multi_agent_score", "Multi-Agent+IDF"),
    ]

    print(f"\n{'Method':<20} {'Rel Mean':>10} {'Irr Mean':>10} {'Gap':>10} {'Cohen d':>10} {'AUC':>8} {'p-value':>12}")
    print("-" * 82)

    results = []
    for col, name in methods:
        rel = df[df["relevance"] == 1][col].values
        irr = df[df["relevance"] == 0][col].values

        gap = rel.mean() - irr.mean()

        # cohens d for effect size
        pooled_std = np.sqrt((rel.std()**2 + irr.std()**2) / 2)
        cohens_d = gap / pooled_std if pooled_std > 0 else 0

        # AUC - probability that random relevant scores higher then random irrelevant
        y_true = df["relevance"].values
        y_scores = df[col].values
        auc = roc_auc_score(y_true, y_scores)

        # mann-whitney U test
        u_stat, p_val = stats.mannwhitneyu(rel, irr, alternative="greater")

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        print(f"{name:<20} {rel.mean():>10.4f} {irr.mean():>10.4f} {gap:>+10.4f} "
              f"{cohens_d:>10.3f} {auc:>8.4f} {p_val:>10.2e} {sig}")

        results.append({
            "method": name,
            "relevant_mean": round(rel.mean(), 4),
            "irrelevant_mean": round(irr.mean(), 4),
            "gap": round(gap, 4),
            "cohens_d": round(cohens_d, 3),
            "auc": round(auc, 4),
            "p_value": p_val,
            "significant": p_val < 0.05,
        })

    print("\n  Interpretation:")
    print("  Cohen's d: 0.2=small, 0.5=medium, 0.8=large effect")
    print("  AUC: 0.5=random, 0.7=acceptable, 0.8=good, 0.9=excellent")

    return results


def difficulty_analysis(df):
    """Check how well each method distinguishes easy vs hard irrelevant pairs."""
    print("\n" + "=" * 70)
    print("2. EASY vs HARD IRRELEVANT ANALYSIS")
    print("=" * 70)

    methods = [
        ("tfidf_score", "TF-IDF"),
        ("skill_idf_score", "Skill-IDF"),
        ("multi_agent_score", "Multi-Agent+IDF"),
    ]

    relevant = df[df["difficulty"] == "relevant"]
    hard = df[df["difficulty"] == "hard"]
    easy = df[df["difficulty"] == "easy"]

    print(f"\n  Counts: relevant={len(relevant)}, hard_irrelevant={len(hard)}, easy_irrelevant={len(easy)}")

    print(f"\n{'Method':<20} {'Relevant':>10} {'Hard Irr':>10} {'Easy Irr':>10} {'R-H Gap':>10} {'R-E Gap':>10} {'H-E Gap':>10}")
    print("-" * 82)

    results = []
    for col, name in methods:
        r_mean = relevant[col].mean()
        h_mean = hard[col].mean()
        e_mean = easy[col].mean()

        print(f"{name:<20} {r_mean:>10.4f} {h_mean:>10.4f} {e_mean:>10.4f} "
              f"{r_mean - h_mean:>+10.4f} {r_mean - e_mean:>+10.4f} {h_mean - e_mean:>+10.4f}")

        results.append({
            "method": name,
            "relevant_mean": round(r_mean, 4),
            "hard_irr_mean": round(h_mean, 4),
            "easy_irr_mean": round(e_mean, 4),
            "rel_hard_gap": round(r_mean - h_mean, 4),
            "rel_easy_gap": round(r_mean - e_mean, 4),
        })

    # check if the ordering is correct for each method
    print("\n  Expected pattern: Relevant > Hard Irrelevant > Easy Irrelevant")
    for col, name in methods:
        r = relevant[col].mean()
        h = hard[col].mean()
        e = easy[col].mean()
        if r > h > e:
            print(f"  [PASS] {name}: correct ordering (R={r:.4f} > H={h:.4f} > E={e:.4f})")
        elif r > h and r > e:
            print(f"  [WARN] {name}: relevant highest but hard/easy not ordered (R={r:.4f}, H={h:.4f}, E={e:.4f})")
        else:
            print(f"  [FAIL] {name}: incorrect ordering (R={r:.4f}, H={h:.4f}, E={e:.4f})")

    return results


def per_keyword_stress(df):
    """Per-keyword domain discrimination check for multi-agent."""
    print("\n" + "=" * 70)
    print("3. PER-KEYWORD DISCRIMINATION (Multi-Agent)")
    print("=" * 70)

    col = "multi_agent_score"
    keywords = sorted(df["keyword"].unique())

    print(f"\n{'Keyword':<15} {'Rel Mean':>10} {'Irr Mean':>10} {'Gap':>10} {'AUC':>8} {'Status':>8}")
    print("-" * 65)

    for kw in keywords:
        kw_df = df[df["keyword"] == kw]
        rel = kw_df[kw_df["relevance"] == 1][col].values
        irr = kw_df[kw_df["relevance"] == 0][col].values

        gap = rel.mean() - irr.mean()
        auc = roc_auc_score(kw_df["relevance"].values, kw_df[col].values)

        status = "PASS" if auc > 0.6 else "WEAK" if auc > 0.5 else "FAIL"
        print(f"{kw:<15} {rel.mean():>10.4f} {irr.mean():>10.4f} {gap:>+10.4f} {auc:>8.4f} [{status}]")


def plot_stress_test(df, output_dir):
    """Generate stress test charts."""
    methods = [
        ("tfidf_score", "TF-IDF", "#3498db"),
        ("skill_idf_score", "Skill-IDF", "#e74c3c"),
        ("multi_agent_score", "Multi-Agent+IDF", "#2ecc71"),
    ]

    # score distributions by difficulty level
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Score Distributions by Difficulty Level", fontsize=16, fontweight="bold")

    for idx, (col, name, color) in enumerate(methods):
        ax = axes[idx]
        for diff, diff_color, label in [
            ("relevant", "green", "Relevant"),
            ("hard", "orange", "Hard Irrelevant"),
            ("easy", "red", "Easy Irrelevant"),
        ]:
            subset = df[df["difficulty"] == diff][col]
            ax.hist(subset, bins=20, alpha=0.5, label=label, color=diff_color, edgecolor="black")

        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "stress_test_distributions.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"\n  Saved: {path}")
    plt.close()

    # AUC comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    method_names = []
    aucs = []
    colors = []
    for col, name, color in methods:
        auc = roc_auc_score(df["relevance"].values, df[col].values)
        method_names.append(name)
        aucs.append(auc)
        colors.append(color)

    bars = ax.bar(method_names, aucs, color=colors, edgecolor="black", linewidth=1.5)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("AUC (Area Under ROC Curve)", fontsize=13, fontweight="bold")
    ax.set_title("Domain Discrimination: AUC Comparison", fontsize=15, fontweight="bold")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random (0.5)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "stress_test_auc.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def main():
    print("=" * 70)
    print("STRESS TEST - DOMAIN DISCRIMINATION ANALYSIS")
    print("=" * 70)

    if not DETAILED_SCORES_PATH.exists():
        print(f"Error: {DETAILED_SCORES_PATH} not found. Run evaluate_all.py first.")
        sys.exit(1)

    df = pd.read_csv(DETAILED_SCORES_PATH)
    print(f"Loaded {len(df):,} scored pairs")

    # run all analyses
    separation_results = domain_separation_analysis(df)
    difficulty_results = difficulty_analysis(df)
    per_keyword_stress(df)

    # generate charts
    viz_dir = OUTPUT_DIR / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    plot_stress_test(df, viz_dir)

    # save results
    sep_df = pd.DataFrame(separation_results)
    sep_df.to_csv(RESULTS_DIR / "stress_test_separation.csv", index=False)
    print(f"\n  Separation results: {RESULTS_DIR / 'stress_test_separation.csv'}")

    diff_df = pd.DataFrame(difficulty_results)
    diff_df.to_csv(RESULTS_DIR / "stress_test_difficulty.csv", index=False)
    print(f"  Difficulty results: {RESULTS_DIR / 'stress_test_difficulty.csv'}")

    # markdown summary
    md_path = viz_dir / "stress_test_summary.md"
    with open(md_path, "w") as f:
        f.write("# Stress Test - Domain Discrimination\n\n")
        f.write("## Domain Separation\n\n")
        f.write(sep_df.drop(columns=["p_value"]).to_markdown(index=False))
        f.write("\n\n## Easy vs Hard Irrelevant\n\n")
        f.write(diff_df.to_markdown(index=False))
    print(f"  Summary: {md_path}")

    print("\n" + "=" * 70)
    print("STRESS TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
