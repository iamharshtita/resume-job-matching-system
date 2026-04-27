"""
Ablation study: measure contribution of each component.

Tests:
1. Full system (all features)
2. Without IDF weighting (equal skill weights)
3. Without experience score (skill + title only)
4. Without title score (skill + experience only)
5. Skill score only

Usage:
    PYTHONPATH=src python3 scripts/ablation_study.py --n-jobs 5 --k 5
"""
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ndcg_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agents.orchestrator import SkillMiningOrchestrator
from agents.matcher import MatchingAgent
from config import PROCESSED_DIR, OUTPUT_DIR, WEIGHTS

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

def build_test_set(resumes_df, jds_df, keyword, n_jobs, n_relevant, n_irrelevant):
    """Build test set (same as evaluate_all.py)."""
    target_jds = jds_df[jds_df["primary_keyword"] == keyword]
    target_jds = target_jds[target_jds["required_skills"].apply(len) >= 3].iloc[:n_jobs]

    if len(target_jds) == 0:
        raise ValueError(f"No JDs found for keyword '{keyword}'")

    rel_resumes = resumes_df[resumes_df["primary_keyword"] == keyword].iloc[:n_relevant]
    other_keywords = resumes_df[resumes_df["primary_keyword"] != keyword]["primary_keyword"].unique()
    other_keyword = other_keywords[0]
    irrel_resumes = resumes_df[resumes_df["primary_keyword"] == other_keyword].iloc[:n_irrelevant]

    all_resumes = pd.concat([rel_resumes, irrel_resumes]).reset_index(drop=True)

    pairs = []
    for _, jd in target_jds.iterrows():
        for _, resume in all_resumes.iterrows():
            pairs.append({
                "jd_id": str(jd["id"]),
                "jd_text": jd["raw_text"],
                "resume_id": str(resume["id"]),
                "resume_text": resume["raw_text"],
                "relevance": 1 if resume["primary_keyword"] == keyword else 0,
            })

    return pairs

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

    # Temporarily disable IDF by setting all weights to 1.0
    from agents.matcher import MatchingAgent
    import json
    from config import PROCESSED_DIR

    # Backup original IDF weights
    idf_path = PROCESSED_DIR / "skill_idf.json"
    backup_path = PROCESSED_DIR / "skill_idf_backup.json"

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
    fig, ax = plt.subplots(figsize=(12, 6))

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
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--keyword", type=str, default="Python", help="Target keyword")
    parser.add_argument("--n-jobs", type=int, default=5, help="Number of JDs")
    parser.add_argument("--n-relevant", type=int, default=10, help="Relevant resumes")
    parser.add_argument("--n-irrelevant", type=int, default=10, help="Irrelevant resumes")
    parser.add_argument("--k", type=int, default=5, help="Top-k for NDCG")
    args = parser.parse_args()

    print("=" * 70)
    print("ABLATION STUDY - COMPONENT CONTRIBUTION")
    print("=" * 70)

    # Load data
    print(f"\nLoading datasets...")
    resumes_df = pd.read_parquet(PROCESSED_DIR / "resumes_parsed.parquet")
    jds_df = pd.read_parquet(PROCESSED_DIR / "jds_parsed.parquet")

    # Build test set
    print(f"\nBuilding test set (keyword: {args.keyword})...")
    pairs = build_test_set(
        resumes_df, jds_df, args.keyword,
        args.n_jobs, args.n_relevant, args.n_irrelevant
    )
    print(f"  Test pairs: {len(pairs):,}")

    # Run variants
    print(f"\n" + "=" * 70)
    print("RUNNING ABLATION VARIANTS")
    print("=" * 70)

    variants = {}

    # 1. Full system
    t0 = time.time()
    full_scores = run_full_system(pairs)
    print(f"    Completed in {time.time() - t0:.2f}s")

    # 2. Without IDF
    t0 = time.time()
    no_idf_scores = run_without_idf(pairs)
    print(f"    Completed in {time.time() - t0:.2f}s")

    # 3. Without experience
    t0 = time.time()
    no_exp_scores = run_without_experience(pairs)
    print(f"    Completed in {time.time() - t0:.2f}s")

    # 4. Without title
    t0 = time.time()
    no_title_scores = run_without_title(pairs)
    print(f"    Completed in {time.time() - t0:.2f}s")

    # 5. Skill only
    t0 = time.time()
    skill_only_scores = run_skill_only(pairs)
    print(f"    Completed in {time.time() - t0:.2f}s")

    # Build dataframe
    rows = []
    for p in pairs:
        key = (p["resume_id"], p["jd_id"])
        rows.append({
            "resume_id": p["resume_id"],
            "jd_id": p["jd_id"],
            "relevance": p["relevance"],
            "full": full_scores.get(key, 0.0),
            "no_idf": no_idf_scores.get(key, 0.0),
            "no_exp": no_exp_scores.get(key, 0.0),
            "no_title": no_title_scores.get(key, 0.0),
            "skill_only": skill_only_scores.get(key, 0.0),
        })

    df = pd.DataFrame(rows)

    # Evaluate
    print(f"\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)

    results = {
        "Full System": evaluate_variant(df, "full", args.k),
        "Without IDF": evaluate_variant(df, "no_idf", args.k),
        "Without Experience": evaluate_variant(df, "no_exp", args.k),
        "Without Title": evaluate_variant(df, "no_title", args.k),
        "Skill Only": evaluate_variant(df, "skill_only", args.k),
    }

    print(f"\n{'Variant':<25} {'NDCG@'+str(args.k):>10} {'Δ vs Full':>12}")
    print("-" * 50)

    full_ndcg = results["Full System"]
    for variant, ndcg in results.items():
        delta = ndcg - full_ndcg
        print(f"{variant:<25} {ndcg:>10.4f} {delta:>+12.4f}")

    # Visualization
    print(f"\nGenerating visualization...")
    viz_dir = OUTPUT_DIR / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    plot_ablation_results(results, viz_dir)

    # Save results
    output_path = OUTPUT_DIR / "results" / "ablation_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([
        {"variant": k, "ndcg@5": v, "delta": v - full_ndcg}
        for k, v in results.items()
    ])
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
