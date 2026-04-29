"""
Statistical significance tests and improvment reporting.

Computes:
1. Paired Wilcoxon signed-rank test on per-JD NDCG values
2. Absolute and relative improvment over baselines
3. Confidence intervals via bootstrap

Usage:
    python3 src/evaluation/statistical_tests.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import ndcg_score

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import OUTPUT_DIR, RESULTS_DIR

DETAILED_SCORES_PATH = RESULTS_DIR / "detailed_scores.csv"


def compute_per_jd_ndcg(df, score_col, k=5):
    """Compute NDCG@k for each JD seperately."""
    ndcgs = {}
    for jd_id, group in df.groupby("jd_id"):
        scores = group[score_col].values.astype(float)
        relevance = group["relevance"].values.astype(float)
        try:
            ndcg = ndcg_score(relevance.reshape(1, -1), scores.reshape(1, -1), k=k)
            ndcgs[jd_id] = ndcg
        except:
            ndcgs[jd_id] = float("nan")
    return ndcgs


def paired_significance_tests(df, k=5):
    """Run paired statistical tests between all method pairs."""
    print("\n" + "=" * 70)
    print(f"1. PAIRED SIGNIFICANCE TESTS (per-JD NDCG@{k})")
    print("=" * 70)

    methods = [
        ("tfidf_score", "TF-IDF"),
        ("skill_idf_score", "Skill-IDF"),
        ("multi_agent_score", "Multi-Agent+IDF"),
    ]

    # compute per-JD NDCG for each method
    per_jd = {}
    for col, name in methods:
        per_jd[name] = compute_per_jd_ndcg(df, col, k)

    # align JD IDs so we compare the same JDs
    jd_ids = sorted(set.intersection(*[set(v.keys()) for v in per_jd.values()]))
    print(f"\n  JDs with valid NDCG for all methods: {len(jd_ids)}")

    print(f"\n  {'Method':<20} {'Mean NDCG':>10} {'Std':>10} {'Median':>10}")
    print("  " + "-" * 52)
    for col, name in methods:
        vals = [per_jd[name][jid] for jid in jd_ids if not np.isnan(per_jd[name][jid])]
        print(f"  {name:<20} {np.mean(vals):>10.4f} {np.std(vals):>10.4f} {np.median(vals):>10.4f}")

    # pairwise wilcoxon tests
    print(f"\n  Paired Wilcoxon Signed-Rank Tests:")
    print(f"  {'Comparison':<40} {'W-stat':>10} {'p-value':>12} {'Sig':>6}")
    print("  " + "-" * 70)

    test_results = []
    comparisons = [
        ("Multi-Agent+IDF", "TF-IDF"),
        ("Multi-Agent+IDF", "Skill-IDF"),
        ("Skill-IDF", "TF-IDF"),
    ]

    for method_a, method_b in comparisons:
        vals_a = np.array([per_jd[method_a][jid] for jid in jd_ids])
        vals_b = np.array([per_jd[method_b][jid] for jid in jd_ids])

        # remove NaN pairs
        mask = ~(np.isnan(vals_a) | np.isnan(vals_b))
        vals_a = vals_a[mask]
        vals_b = vals_b[mask]

        # one-sided test: is method a better then method b?
        diffs = vals_a - vals_b
        if np.all(diffs == 0):
            w_stat, p_val = 0, 1.0
        else:
            w_stat, p_val = stats.wilcoxon(vals_a, vals_b, alternative="greater")

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        label = f"{method_a} > {method_b}"
        print(f"  {label:<40} {w_stat:>10.0f} {p_val:>12.2e} {sig:>6}")

        # count wins/ties/losses
        wins = (vals_a > vals_b).sum()
        ties = (vals_a == vals_b).sum()
        losses = (vals_a < vals_b).sum()

        test_results.append({
            "comparison": label,
            "w_statistic": round(float(w_stat), 2),
            "p_value": float(p_val),
            "significant_005": p_val < 0.05,
            "wins": int(wins),
            "ties": int(ties),
            "losses": int(losses),
            "n_pairs": len(vals_a),
        })

        print(f"    wins={wins}, ties={ties}, losses={losses} (out of {len(vals_a)} JDs)")

    return test_results, per_jd, jd_ids


def improvement_report(df, k=5):
    """Compute absolute and relative improvments over baselines."""
    print("\n" + "=" * 70)
    print("2. IMPROVEMENT REPORT")
    print("=" * 70)

    methods = [
        ("tfidf_score", "TF-IDF"),
        ("skill_idf_score", "Skill-IDF"),
        ("multi_agent_score", "Multi-Agent+IDF"),
    ]

    # compute overall metrics using the same function as evaluate_all
    from evaluation.evaluate_all import evaluate_method
    metrics = {}
    for col, name in methods:
        metrics[name] = evaluate_method(df, col, k)

    metric_keys = [f"ndcg@{k}", f"prec@{k}", f"rec@{k}", "map"]

    print(f"\n  {'Method':<20}", end="")
    for mk in metric_keys:
        print(f" {mk.upper():>10}", end="")
    print()
    print("  " + "-" * 62)

    for col, name in methods:
        print(f"  {name:<20}", end="")
        for mk in metric_keys:
            print(f" {metrics[name][mk]:>10.4f}", end="")
        print()

    # show how much multi-agent improves over each baseline
    print(f"\n  Multi-Agent+IDF Improvement Over Baselines:")
    print(f"  {'Baseline':<15} {'Metric':<10} {'Baseline':>10} {'Multi-Agent':>12} {'Abs Gain':>10} {'Rel Gain':>10}")
    print("  " + "-" * 70)

    improvement_rows = []
    for baseline_name in ["TF-IDF", "Skill-IDF"]:
        for mk in metric_keys:
            base_val = metrics[baseline_name][mk]
            ma_val = metrics["Multi-Agent+IDF"][mk]
            abs_gain = ma_val - base_val
            rel_gain = (abs_gain / base_val * 100) if base_val > 0 else 0

            print(f"  {baseline_name:<15} {mk:<10} {base_val:>10.4f} {ma_val:>12.4f} "
                  f"{abs_gain:>+10.4f} {rel_gain:>+9.1f}%")

            improvement_rows.append({
                "baseline": baseline_name,
                "metric": mk,
                "baseline_value": base_val,
                "multi_agent_value": ma_val,
                "absolute_gain": round(abs_gain, 4),
                "relative_gain_pct": round(rel_gain, 1),
            })

    return improvement_rows


def bootstrap_confidence_intervals(df, k=5, n_bootstrap=1000, seed=42):
    """Compute bootstrap confidence intervals for NDCG@k."""
    print("\n" + "=" * 70)
    print(f"3. BOOTSTRAP CONFIDENCE INTERVALS (NDCG@{k}, {n_bootstrap} iterations)")
    print("=" * 70)

    rng = np.random.RandomState(seed)

    methods = [
        ("tfidf_score", "TF-IDF"),
        ("skill_idf_score", "Skill-IDF"),
        ("multi_agent_score", "Multi-Agent+IDF"),
    ]

    jd_ids = df["jd_id"].unique()

    print(f"\n  {'Method':<20} {'Mean NDCG':>10} {'95% CI':>20} {'Std':>10}")
    print("  " + "-" * 62)

    ci_results = []
    for col, name in methods:
        per_jd_ndcg = compute_per_jd_ndcg(df, col, k)
        jd_ndcgs = np.array([per_jd_ndcg[jid] for jid in jd_ids if not np.isnan(per_jd_ndcg.get(jid, float("nan")))])

        # resample and compute mean each time
        boot_means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(jd_ndcgs, size=len(jd_ndcgs), replace=True)
            boot_means.append(np.mean(sample))

        boot_means = np.array(boot_means)
        ci_low = np.percentile(boot_means, 2.5)
        ci_high = np.percentile(boot_means, 97.5)
        mean_val = np.mean(jd_ndcgs)

        print(f"  {name:<20} {mean_val:>10.4f} [{ci_low:.4f}, {ci_high:.4f}] {np.std(boot_means):>10.4f}")

        ci_results.append({
            "method": name,
            "mean_ndcg": round(mean_val, 4),
            "ci_lower": round(ci_low, 4),
            "ci_upper": round(ci_high, 4),
            "bootstrap_std": round(float(np.std(boot_means)), 4),
        })

    return ci_results


def main():
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE & IMPROVEMENT REPORT")
    print("=" * 70)

    if not DETAILED_SCORES_PATH.exists():
        print(f"Error: {DETAILED_SCORES_PATH} not found. Run evaluate_all.py first.")
        sys.exit(1)

    df = pd.read_csv(DETAILED_SCORES_PATH)
    print(f"Loaded {len(df):,} scored pairs, {df['jd_id'].nunique()} JDs")

    # run all analyses
    test_results, per_jd, jd_ids = paired_significance_tests(df)
    improvement_rows = improvement_report(df)
    ci_results = bootstrap_confidence_intervals(df)

    # save everything
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    viz_dir = OUTPUT_DIR / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(test_results).to_csv(RESULTS_DIR / "significance_tests.csv", index=False)
    print(f"\n  Significance tests: {RESULTS_DIR / 'significance_tests.csv'}")

    pd.DataFrame(improvement_rows).to_csv(RESULTS_DIR / "improvement_report.csv", index=False)
    print(f"  Improvement report: {RESULTS_DIR / 'improvement_report.csv'}")

    pd.DataFrame(ci_results).to_csv(RESULTS_DIR / "confidence_intervals.csv", index=False)
    print(f"  Confidence intervals: {RESULTS_DIR / 'confidence_intervals.csv'}")

    # markdown summary
    md_path = viz_dir / "statistical_summary.md"
    with open(md_path, "w") as f:
        f.write("# Statistical Analysis\n\n")
        f.write("## Paired Significance Tests (Wilcoxon Signed-Rank)\n\n")
        f.write(pd.DataFrame(test_results).to_markdown(index=False))
        f.write("\n\n## Bootstrap 95% Confidence Intervals (NDCG@5)\n\n")
        f.write(pd.DataFrame(ci_results).to_markdown(index=False))
        f.write("\n\n## Improvement Over Baselines\n\n")
        f.write(pd.DataFrame(improvement_rows).to_markdown(index=False))
    print(f"  Summary: {md_path}")

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
