"""
Ablation study - measures how much each component contributes to the final score.

Tests:
1. Full system (all features)
2. Without IDF weighting (equal skill weights)
3. Without experience score (skill + title only)
4. Without title score (skill + experience only)
5. Skill score only

Reads from data/test/eval_pairs.parquet. Use --n-jds to control how many JDs
to run (ablation runs pipeline 5x so keep this small, default is 10).

Usage:
    python3 src/evaluation/ablation_study.py
    python3 src/evaluation/ablation_study.py --n-jds 20 --k 5
"""
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agents.orchestrator import SkillMiningOrchestrator
from config import OUTPUT_DIR, WEIGHTS

EVAL_PAIRS_PATH = ROOT / "data" / "test" / "eval_pairs.parquet"

def compute_ndcg(scores, relevance, k):
    """Compute NDCG@k for a single JD."""
    try:
        return ndcg_score(relevance.reshape(1, -1), scores.reshape(1, -1), k=k)
    except:
        return float("nan")

def evaluate_variant(df, score_col, k):
    """Evaluate a variant across all JDs."""
    ndcgs = []
    for jd_id, group in df.groupby("jd_id"):
        s = group[score_col].values.astype(float)
        r = group["relevance"].values.astype(float)
        ndcg = compute_ndcg(s, r, k)
        ndcgs.append(ndcg)
    return np.nanmean(ndcgs)


def run_full_system(pairs):
    """Variant 1: Full system with all features."""
    print("  [1/5] Full system (all features)...")
    orchestrator = SkillMiningOrchestrator()

    by_jd = defaultdict(list)
    for p in pairs:
        by_jd[p["jd_id"]].append(p)

    scores = {}
    for jd_id, jd_pairs in by_jd.items():
        jd_text = jd_pairs[0]["jd_text"]
        candidates = [(p["resume_id"], p["resume_text"]) for p in jd_pairs]

        ranked = orchestrator.rank_candidates(
            jd_text=jd_text,
            candidates=candidates,
            jd_id=jd_id,
        )

        for r in ranked:
            scores[(r["resume_id"], jd_id)] = r["final_score"]

    return scores

def run_without_idf(pairs):
    """Variant 2: Without IDF weighting (equal skill weights)."""
    print("  [2/5] Without IDF weighting...")

    # swap IDF weights to uniform 1.0 temporarily then restore after
    import json
    from config import PROCESSED_DIR

    idf_path = PROCESSED_DIR / "skill_idf.json"

    if idf_path.exists():
        with open(idf_path, 'r') as f:
            original_weights = json.load(f)

        # Create uniform weights
        uniform_weights = {k: 1.0 for k in original_weights.keys()}

        with open(idf_path, 'w') as f:
            json.dump(uniform_weights, f)
    else:
        original_weights = None

    # Run evaluation
    orchestrator = SkillMiningOrchestrator()

    by_jd = defaultdict(list)
    for p in pairs:
        by_jd[p["jd_id"]].append(p)

    scores = {}
    for jd_id, jd_pairs in by_jd.items():
        jd_text = jd_pairs[0]["jd_text"]
        candidates = [(p["resume_id"], p["resume_text"]) for p in jd_pairs]

        ranked = orchestrator.rank_candidates(
            jd_text=jd_text,
            candidates=candidates,
            jd_id=jd_id,
        )

        for r in ranked:
            scores[(r["resume_id"], jd_id)] = r["final_score"]

    # Restore original weights
    if original_weights:
        with open(idf_path, 'w') as f:
            json.dump(original_weights, f)

    return scores

def run_without_experience(pairs):
    """Variant 3: Without experience score."""
    print("  [3/5] Without experience score...")

    # Temporarily set experience weight to 0
    original_weights = WEIGHTS.copy()
    WEIGHTS["experience"] = 0.0
    # Renormalize
    total = sum(WEIGHTS.values())
    for k in WEIGHTS:
        WEIGHTS[k] /= total

    orchestrator = SkillMiningOrchestrator()

    by_jd = defaultdict(list)
    for p in pairs:
        by_jd[p["jd_id"]].append(p)

    scores = {}
    for jd_id, jd_pairs in by_jd.items():
        jd_text = jd_pairs[0]["jd_text"]
        candidates = [(p["resume_id"], p["resume_text"]) for p in jd_pairs]

        ranked = orchestrator.rank_candidates(
            jd_text=jd_text,
            candidates=candidates,
            jd_id=jd_id,
        )

        for r in ranked:
            scores[(r["resume_id"], jd_id)] = r["final_score"]

    # Restore original weights
    WEIGHTS.update(original_weights)

    return scores

def run_without_title(pairs):
    """Variant 4: Without title score."""
    print("  [4/5] Without title score...")

    original_weights = WEIGHTS.copy()
    WEIGHTS["title"] = 0.0
    total = sum(WEIGHTS.values())
    for k in WEIGHTS:
        WEIGHTS[k] /= total

    orchestrator = SkillMiningOrchestrator()

    by_jd = defaultdict(list)
    for p in pairs:
        by_jd[p["jd_id"]].append(p)

    scores = {}
    for jd_id, jd_pairs in by_jd.items():
        jd_text = jd_pairs[0]["jd_text"]
        candidates = [(p["resume_id"], p["resume_text"]) for p in jd_pairs]

        ranked = orchestrator.rank_candidates(
            jd_text=jd_text,
            candidates=candidates,
            jd_id=jd_id,
        )

        for r in ranked:
            scores[(r["resume_id"], jd_id)] = r["final_score"]

    WEIGHTS.update(original_weights)
    return scores

def run_skill_only(pairs):
    """Variant 5: Skill score only (100% weight)."""
    print("  [5/5] Skill score only...")

    original_weights = WEIGHTS.copy()
    WEIGHTS["skill"] = 1.0
    WEIGHTS["experience"] = 0.0
    WEIGHTS["education"] = 0.0
    WEIGHTS["title"] = 0.0

    orchestrator = SkillMiningOrchestrator()

    by_jd = defaultdict(list)
    for p in pairs:
        by_jd[p["jd_id"]].append(p)

    scores = {}
    for jd_id, jd_pairs in by_jd.items():
        jd_text = jd_pairs[0]["jd_text"]
        candidates = [(p["resume_id"], p["resume_text"]) for p in jd_pairs]

        ranked = orchestrator.rank_candidates(
            jd_text=jd_text,
            candidates=candidates,
            jd_id=jd_id,
        )

        for r in ranked:
            scores[(r["resume_id"], jd_id)] = r["final_score"]

    WEIGHTS.update(original_weights)
    return scores

def plot_ablation_results(results, output_dir):
    """Bar chart comparing all variants."""
    _, ax = plt.subplots(figsize=(12, 6))

    variants = list(results.keys())
    ndcg_values = list(results.values())

    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#95a5a6']
    bars = ax.bar(variants, ndcg_values, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.4f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('NDCG@5', fontsize=13, fontweight='bold')
    ax.set_title('Ablation Study: Component Contribution', fontsize=15, fontweight='bold')
    ax.set_ylim(0, max(ndcg_values) * 1.15)
    ax.grid(axis='y', alpha=0.3)

    # Rotate x labels
    plt.xticks(rotation=15, ha='right')

    plt.tight_layout()
    output_path = output_dir / 'ablation_study.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="ablation study")
    parser.add_argument("--n-jds", type=int, default=10,
                        help="number of JDs to sample from eval_pairs (ablation runs pipeline 5x so keep small)")
    parser.add_argument("--k", type=int, default=5, help="top-k for NDCG")
    args = parser.parse_args()

    print("=" * 70)
    print("ABLATION STUDY - COMPONENT CONTRIBUTION")
    print("=" * 70)

    if not EVAL_PAIRS_PATH.exists():
        print(f"eval_pairs.parquet not found, run scripts/build_eval_dataset.py first")
        sys.exit(1)

    # load eval pairs and sample n-jds JDs to keep runtime managable
    all_pairs_df = pd.read_parquet(EVAL_PAIRS_PATH)
    jd_ids = all_pairs_df["jd_id"].drop_duplicates().sample(
        min(args.n_jds, all_pairs_df["jd_id"].nunique()), random_state=42
    ).tolist()
    sampled_df = all_pairs_df[all_pairs_df["jd_id"].isin(jd_ids)]
    pairs = sampled_df.to_dict("records")
    print(f"running on {len(jd_ids)} JDs, {len(pairs)} pairs")

    print("\n" + "=" * 70)
    print("RUNNING ABLATION VARIANTS")
    print("=" * 70)

    t0 = time.time()
    full_scores = run_full_system(pairs)
    print(f"    done in {time.time() - t0:.2f}s")

    t0 = time.time()
    no_idf_scores = run_without_idf(pairs)
    print(f"    done in {time.time() - t0:.2f}s")

    t0 = time.time()
    no_exp_scores = run_without_experience(pairs)
    print(f"    done in {time.time() - t0:.2f}s")

    t0 = time.time()
    no_title_scores = run_without_title(pairs)
    print(f"    done in {time.time() - t0:.2f}s")

    t0 = time.time()
    skill_only_scores = run_skill_only(pairs)
    print(f"    done in {time.time() - t0:.2f}s")

    rows = []
    for p in pairs:
        key = (p["resume_id"], p["jd_id"])
        rows.append({
            "resume_id":  p["resume_id"],
            "jd_id":      p["jd_id"],
            "relevance":  p["relevance"],
            "full":       full_scores.get(key, 0.0),
            "no_idf":     no_idf_scores.get(key, 0.0),
            "no_exp":     no_exp_scores.get(key, 0.0),
            "no_title":   no_title_scores.get(key, 0.0),
            "skill_only": skill_only_scores.get(key, 0.0),
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)

    results = {
        "Full System":        evaluate_variant(df, "full", args.k),
        "Without IDF":        evaluate_variant(df, "no_idf", args.k),
        "Without Experience": evaluate_variant(df, "no_exp", args.k),
        "Without Title":      evaluate_variant(df, "no_title", args.k),
        "Skill Only":         evaluate_variant(df, "skill_only", args.k),
    }

    full_ndcg = results["Full System"]
    print("\n{:<25} {:>10} {:>12}".format("Variant", f"NDCG@{args.k}", "vs Full"))
    print("-" * 50)
    for variant, ndcg in results.items():
        print("{:<25} {:>10.4f} {:>+12.4f}".format(variant, ndcg, ndcg - full_ndcg))

    viz_dir = OUTPUT_DIR / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    plot_ablation_results(results, viz_dir)

    output_path = OUTPUT_DIR / "results" / "ablation_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"variant": k, "ndcg@5": v, "delta": v - full_ndcg}
        for k, v in results.items()
    ]).to_csv(output_path, index=False)
    print(f"\nresults saved to: {output_path}")

    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
