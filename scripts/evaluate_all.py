"""
Unified evaluation script - compares all methods on the same test set.

Runs:
1. TF-IDF Baseline
2. Skill-IDF Baseline
3. Multi-Agent Pipeline (with IDF-weighted matching)

Outputs: comparison_results.csv with NDCG@5, Precision@5, Recall@5, MAP

Usage:
    PYTHONPATH=src python3 scripts/evaluate_all.py --n-jobs 5 --n-candidates 20 --k 5
"""
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from baselines.tfidf_baseline import TFIDFBaselineAgent
from agents.orchestrator import SkillMiningOrchestrator
from config import PROCESSED_DIR

def compute_metrics(scores, relevance, k):
    """Compute ranking metrics for a single JD."""
    order = np.argsort(-scores)
    prec = relevance[order][:k].sum() / k
    rec = relevance[order][:k].sum() / max(relevance.sum(), 1)

    try:
        ndcg = ndcg_score(relevance.reshape(1, -1), scores.reshape(1, -1), k=k)
    except:
        ndcg = float("nan")

    # MAP
    relevant_positions = np.where(relevance[order] == 1)[0]
    if len(relevant_positions) > 0:
        precisions_at_k = [(i + 1) / (pos + 1) for i, pos in enumerate(relevant_positions) if pos < k]
        map_score = sum(precisions_at_k) / max(len(relevant_positions), 1) if precisions_at_k else 0.0
    else:
        map_score = 0.0

    return {"ndcg": ndcg, "prec": prec, "rec": rec, "map": map_score}

def evaluate_method(df, score_col, k):
    """Evaluate a method across all JDs in the dataframe."""
    ndcgs, precs, recs, maps = [], [], [], []

    for jd_id, group in df.groupby("jd_id"):
        s = group[score_col].values.astype(float)
        r = group["relevance"].values.astype(float)

        m = compute_metrics(s, r, k)
        ndcgs.append(m["ndcg"])
        precs.append(m["prec"])
        recs.append(m["rec"])
        maps.append(m["map"])

    return {
        f"ndcg@{k}": round(np.nanmean(ndcgs), 4),
        f"prec@{k}": round(np.nanmean(precs), 4),
        f"rec@{k}": round(np.nanmean(recs), 4),
        "map": round(np.nanmean(maps), 4),
    }

def build_test_set(resumes_df, jds_df, keyword, n_jobs, n_relevant, n_irrelevant):
    """Build test set: relevant (same keyword) vs irrelevant (different keyword)."""

    # Pick JDs with this keyword and at least 3 required skills
    target_jds = jds_df[jds_df["primary_keyword"] == keyword]
    target_jds = target_jds[target_jds["required_skills"].apply(len) >= 3].iloc[:n_jobs]

    if len(target_jds) == 0:
        raise ValueError(f"No JDs found for keyword '{keyword}'")

    # Pick resumes: relevant (same keyword) and irrelevant (different keyword)
    rel_resumes = resumes_df[resumes_df["primary_keyword"] == keyword].iloc[:n_relevant]

    # Get irrelevant from other keywords
    other_keywords = resumes_df[resumes_df["primary_keyword"] != keyword]["primary_keyword"].unique()
    if len(other_keywords) == 0:
        raise ValueError(f"No other keywords found for irrelevant resumes")

    other_keyword = other_keywords[0]
    irrel_resumes = resumes_df[resumes_df["primary_keyword"] == other_keyword].iloc[:n_irrelevant]

    # Combine
    all_resumes = pd.concat([rel_resumes, irrel_resumes]).reset_index(drop=True)

    # Build pairs
    pairs = []
    for _, jd in target_jds.iterrows():
        for _, resume in all_resumes.iterrows():
            pairs.append({
                "jd_id": str(jd["id"]),
                "jd_title": jd["title"],
                "jd_text": jd["raw_text"],
                "jd_keyword": jd["primary_keyword"],
                "resume_id": str(resume["id"]),
                "resume_text": resume["raw_text"],
                "resume_keyword": resume["primary_keyword"],
                "relevance": 1 if resume["primary_keyword"] == keyword else 0,
            })

    return pairs

def run_tfidf_baseline(pairs):
    """Run TF-IDF baseline on all pairs."""
    print("  Running TF-IDF baseline...")
    agent = TFIDFBaselineAgent()

    # Fit on all texts
    all_texts = []
    for p in pairs:
        all_texts.append(p["resume_text"])
        all_texts.append(p["jd_text"])
    agent.fit(list(set(all_texts)))

    # Score
    batch = [(p["resume_id"], p["jd_id"], p["resume_text"], p["jd_text"]) for p in pairs]
    results = agent.score_batch(batch)

    return {(r["resume_id"], r["jd_id"]): r["tfidf_score"] for r in results}

def run_skill_idf_baseline(pairs):
    """Run Skill-IDF baseline (exact matching with IDF weights)."""
    print("  Running Skill-IDF baseline...")

    from agents.jd_parser import JDParserAgent
    from agents.resume_parser import ResumeParserAgent
    import json
    import math
    from collections import Counter

    # Load IDF weights
    idf_path = PROCESSED_DIR / "skill_idf.json"
    if idf_path.exists():
        with open(idf_path, 'r') as f:
            idf_weights = json.load(f)
    else:
        # Compute on-the-fly
        print("    Computing IDF weights...")
        all_jd_skills = []
        for p in pairs:
            jd_parser = JDParserAgent()
            parsed = jd_parser.process({"raw_text": p["jd_text"], "id": p["jd_id"]})
            skills = parsed["required_skills"] + parsed["preferred_skills"]
            all_jd_skills.append([str(s).lower() for s in skills])

        N = len(all_jd_skills)
        df_counts = Counter()
        for skills in all_jd_skills:
            for s in set(skills):
                df_counts[s] += 1

        idf_weights = {sk: math.log((N + 1) / (df + 1)) + 1.0 for sk, df in df_counts.items()}

    # Score pairs
    resume_parser = ResumeParserAgent()
    jd_parser = JDParserAgent()

    scores = {}
    for p in pairs:
        # Parse
        parsed_resume = resume_parser.process({"raw_text": p["resume_text"], "id": p["resume_id"]})
        parsed_jd = jd_parser.process({"raw_text": p["jd_text"], "id": p["jd_id"]})

        resume_skills = set(str(s).lower() for s in parsed_resume["raw_skills"])
        jd_skills = parsed_jd["required_skills"] + parsed_jd["preferred_skills"]

        # IDF-weighted matching
        total_weight = matched_weight = 0.0
        for skill in jd_skills:
            w = idf_weights.get(str(skill).lower(), 1.0)
            total_weight += w
            if str(skill).lower() in resume_skills:
                matched_weight += w

        score = matched_weight / total_weight if total_weight > 0 else 0.0
        scores[(p["resume_id"], p["jd_id"])] = score

    return scores

def run_multi_agent_pipeline(pairs):
    """Run Multi-Agent pipeline (with IDF-weighted matching)."""
    print("  Running Multi-Agent pipeline...")

    orchestrator = SkillMiningOrchestrator()

    # Group by JD to batch process
    by_jd = defaultdict(list)
    for p in pairs:
        by_jd[p["jd_id"]].append(p)

    scores = {}
    for jd_id, jd_pairs in by_jd.items():
        jd_text = jd_pairs[0]["jd_text"]
        candidates = [(p["resume_id"], p["resume_text"]) for p in jd_pairs]

        # Rank
        ranked = orchestrator.rank_candidates(
            jd_text=jd_text,
            candidates=candidates,
            jd_id=jd_id,
        )

        for r in ranked:
            scores[(r["resume_id"], jd_id)] = r["final_score"]

    return scores

def main():
    parser = argparse.ArgumentParser(description="Unified evaluation of all methods")
    parser.add_argument("--keyword", type=str, default="Python", help="Target keyword")
    parser.add_argument("--n-jobs", type=int, default=5, help="Number of JDs")
    parser.add_argument("--n-relevant", type=int, default=10, help="Relevant resumes per JD")
    parser.add_argument("--n-irrelevant", type=int, default=10, help="Irrelevant resumes per JD")
    parser.add_argument("--k", type=int, default=5, help="Top-k for metrics")
    args = parser.parse_args()

    print("=" * 70)
    print("UNIFIED EVALUATION - ALL METHODS")
    print("=" * 70)

    # Load data
    print(f"\nLoading datasets...")
    resumes_df = pd.read_parquet(PROCESSED_DIR / "resumes_parsed.parquet")
    jds_df = pd.read_parquet(PROCESSED_DIR / "jds_parsed.parquet")
    print(f"  Resumes: {len(resumes_df):,}")
    print(f"  JDs: {len(jds_df):,}")

    # Build test set
    print(f"\nBuilding test set (keyword: {args.keyword})...")
    pairs = build_test_set(
        resumes_df, jds_df, args.keyword,
        args.n_jobs, args.n_relevant, args.n_irrelevant
    )
    print(f"  Test pairs: {len(pairs):,}")
    print(f"  Relevant pairs: {sum(p['relevance'] for p in pairs):,}")
    print(f"  Irrelevant pairs: {sum(1-p['relevance'] for p in pairs):,}")

    # Run methods
    print(f"\n" + "=" * 70)
    print("RUNNING EVALUATIONS")
    print("=" * 70)

    results = {}

    # 1. TF-IDF
    t0 = time.time()
    tfidf_scores = run_tfidf_baseline(pairs)
    results["tfidf_time"] = time.time() - t0
    print(f"  TF-IDF completed in {results['tfidf_time']:.2f}s")

    # 2. Skill-IDF
    t0 = time.time()
    skill_idf_scores = run_skill_idf_baseline(pairs)
    results["skill_idf_time"] = time.time() - t0
    print(f"  Skill-IDF completed in {results['skill_idf_time']:.2f}s")

    # 3. Multi-Agent
    t0 = time.time()
    multi_agent_scores = run_multi_agent_pipeline(pairs)
    results["multi_agent_time"] = time.time() - t0
    print(f"  Multi-Agent completed in {results['multi_agent_time']:.2f}s")

    # Build results dataframe
    print(f"\nBuilding results dataframe...")
    rows = []
    for p in pairs:
        key = (p["resume_id"], p["jd_id"])
        rows.append({
            "resume_id": p["resume_id"],
            "jd_id": p["jd_id"],
            "relevance": p["relevance"],
            "tfidf_score": tfidf_scores.get(key, 0.0),
            "skill_idf_score": skill_idf_scores.get(key, 0.0),
            "multi_agent_score": multi_agent_scores.get(key, 0.0),
        })

    df = pd.DataFrame(rows)

    # Evaluate each method
    print(f"\n" + "=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)

    tfidf_metrics = evaluate_method(df, "tfidf_score", args.k)
    skill_idf_metrics = evaluate_method(df, "skill_idf_score", args.k)
    multi_agent_metrics = evaluate_method(df, "multi_agent_score", args.k)

    # Print comparison table
    print(f"\n{'Method':<20} {'NDCG@'+str(args.k):>10} {'P@'+str(args.k):>10} {'R@'+str(args.k):>10} {'MAP':>10} {'Time(s)':>10}")
    print("-" * 70)

    print(f"{'TF-IDF':<20} {tfidf_metrics[f'ndcg@{args.k}']:>10.4f} {tfidf_metrics[f'prec@{args.k}']:>10.4f} "
          f"{tfidf_metrics[f'rec@{args.k}']:>10.4f} {tfidf_metrics['map']:>10.4f} {results['tfidf_time']:>10.2f}")

    print(f"{'Skill-IDF':<20} {skill_idf_metrics[f'ndcg@{args.k}']:>10.4f} {skill_idf_metrics[f'prec@{args.k}']:>10.4f} "
          f"{skill_idf_metrics[f'rec@{args.k}']:>10.4f} {skill_idf_metrics['map']:>10.4f} {results['skill_idf_time']:>10.2f}")

    print(f"{'Multi-Agent+IDF':<20} {multi_agent_metrics[f'ndcg@{args.k}']:>10.4f} {multi_agent_metrics[f'prec@{args.k}']:>10.4f} "
          f"{multi_agent_metrics[f'rec@{args.k}']:>10.4f} {multi_agent_metrics['map']:>10.4f} {results['multi_agent_time']:>10.2f}")

    # Save results
    output_path = PROCESSED_DIR.parent / "outputs" / "results" / "comparison_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame([
        {"method": "TF-IDF", **tfidf_metrics, "time": results["tfidf_time"]},
        {"method": "Skill-IDF", **skill_idf_metrics, "time": results["skill_idf_time"]},
        {"method": "Multi-Agent+IDF", **multi_agent_metrics, "time": results["multi_agent_time"]},
    ])

    summary_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Save detailed scores
    detailed_path = output_path.parent / "detailed_scores.csv"
    df.to_csv(detailed_path, index=False)
    print(f"Detailed scores saved to: {detailed_path}")

if __name__ == "__main__":
    main()
