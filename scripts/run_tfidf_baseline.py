"""
run_tfidf_baseline.py

Compares the TF-IDF baseline against the full multi-agent pipeline.

Ground truth:
    same-keyword pair   -> relevance = 1
    cross-keyword pair  -> relevance = 0

Usage:
    python scripts/run_tfidf_baseline.py
    python scripts/run_tfidf_baseline.py --keyword Java --other Design
    python scripts/run_tfidf_baseline.py --n-jobs 10 --k 10
    python scripts/run_tfidf_baseline.py --auto   (picks top 2 keywords from dataset)
"""
import sys
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agents.tfidf_baseline import TFIDFBaselineAgent
from agents.orchestrator import SkillMiningOrchestrator

PROCESSED = ROOT / "data" / "processed"


def parse_args():
    parser = argparse.ArgumentParser(description="TF-IDF baseline vs multi-agent comparison")
    parser.add_argument("--keyword",       default="Python", help="Primary keyword (relevant resumes)")
    parser.add_argument("--other",         default="Design", help="Counter keyword (irrelevant resumes)")
    parser.add_argument("--n-jobs",        type=int, default=5, help="Number of JDs to evaluate")
    parser.add_argument("--n-relevant",    type=int, default=5, help="Relevant resumes per JD")
    parser.add_argument("--n-irrelevant",  type=int, default=5, help="Irrelevant resumes per JD")
    parser.add_argument("--k",             type=int, default=5, help="K for NDCG@K and Precision@K")
    parser.add_argument("--auto",          action="store_true", help="Auto-pick top 2 keywords from dataset")
    return parser.parse_args()


def auto_pick_keywords(jds):
    """Pick the two most common keywords from the JD dataset as keyword and counter."""
    top = jds["primary_keyword"].value_counts().head(3).index.tolist()
    keyword = top[0]
    other   = top[1] if top[1] != keyword else top[2]
    print(f"Auto-selected keywords: '{keyword}' (relevant) vs '{other}' (irrelevant)")
    return keyword, other


def load_data():
    resumes = pd.read_parquet(PROCESSED / "resumes_parsed.parquet")
    jds     = pd.read_parquet(PROCESSED / "jds_parsed.parquet")
    return resumes, jds


def build_pairs(resumes, jds, keyword, other, n_jobs, n_relevant, n_irrelevant):
    """
    Selects n_jobs JDs matching keyword and builds a candidate pool of
    n_relevant same-keyword resumes + n_irrelevant other-keyword resumes.
    Returns all (resume, JD) pairs with ground truth relevance labels.
    """
    target_jds = jds[jds["primary_keyword"] == keyword]
    target_jds = target_jds[target_jds["required_skills"].apply(len) >= 3]
    target_jds = target_jds.iloc[:n_jobs]

    if len(target_jds) == 0:
        raise ValueError(f"No JDs found for keyword '{keyword}'. Check --keyword value.")

    relevant_resumes   = resumes[resumes["primary_keyword"] == keyword].iloc[:n_relevant]
    irrelevant_resumes = resumes[resumes["primary_keyword"] == other].iloc[:n_irrelevant]

    if len(irrelevant_resumes) == 0:
        raise ValueError(f"No resumes found for counter keyword '{other}'. Check --other value.")

    candidate_pool = pd.concat([relevant_resumes, irrelevant_resumes]).reset_index(drop=True)

    pairs = []
    for _, jd in target_jds.iterrows():
        for _, resume in candidate_pool.iterrows():
            pairs.append({
                "jd_id":       str(jd["id"]),
                "jd_title":    jd["title"],
                "jd_text":     jd["raw_text"],
                "resume_id":   str(resume["id"]),
                "resume_text": resume["raw_text"],
                "relevance":   1 if resume["primary_keyword"] == keyword else 0,
            })

    return pairs, target_jds, candidate_pool


def score_tfidf(agent, pairs):
    """Score all pairs with TF-IDF cosine similarity."""
    t0      = time.perf_counter()
    batch   = [(p["resume_id"], p["jd_id"], p["resume_text"], p["jd_text"]) for p in pairs]
    results = agent.score_batch(batch)
    elapsed = time.perf_counter() - t0

    df = pd.DataFrame(results)
    df["relevance"] = [p["relevance"] for p in pairs]
    df["jd_title"]  = [p["jd_title"]  for p in pairs]

    print(f"TF-IDF scored {len(pairs)} pairs in {elapsed:.2f}s")
    return df


def score_multiagent(orchestrator, target_jds, candidate_pool):
    """Run all pairs through the full multi-agent pipeline."""
    t0         = time.perf_counter()
    candidates = [(str(row["id"]), row["raw_text"]) for _, row in candidate_pool.iterrows()]

    rows = []
    for _, jd in target_jds.iterrows():
        ranked = orchestrator.rank_candidates(
            jd_text=jd["raw_text"],
            candidates=candidates,
            jd_id=str(jd["id"]),
        )
        for r in ranked:
            rows.append({
                "jd_id":       str(jd["id"]),
                "jd_title":    jd["title"],
                "resume_id":   r["resume_id"],
                "ma_score":    r["final_score"],
                "matched":     r["matched_skills"],
                "explanation": r["explanation"],
            })

    elapsed = time.perf_counter() - t0
    print(f"Multi-agent scored {len(rows)} pairs in {elapsed:.2f}s")
    return pd.DataFrame(rows)


def compute_metrics(scores, relevance, k):
    """NDCG@K, Precision@K, and Spearman for one JD's candidate list."""
    order      = np.argsort(-scores)
    rel_sorted = relevance[order]
    prec_at_k  = rel_sorted[:k].sum() / k

    try:
        ndcg = ndcg_score(relevance.reshape(1, -1), scores.reshape(1, -1), k=k)
    except Exception:
        ndcg = float("nan")

    try:
        rho, _ = spearmanr(scores, relevance)
    except Exception:
        rho = float("nan")

    return {"ndcg": ndcg, "precision_at_k": prec_at_k, "spearman": rho}


def evaluate(df, score_col, k):
    """Average NDCG@K, Precision@K, Spearman across all JDs."""
    ndcgs, precs, spears = [], [], []

    for _, group in df.groupby("jd_id"):
        scores    = group[score_col].values.astype(float)
        relevance = group["relevance"].values.astype(float)
        m = compute_metrics(scores, relevance, k)
        ndcgs.append(m["ndcg"])
        precs.append(m["precision_at_k"])
        spears.append(m["spearman"])

    return {
        f"ndcg@{k}":      round(np.nanmean(ndcgs),  4),
        f"precision@{k}": round(np.nanmean(precs),  4),
        "spearman":       round(np.nanmean(spears), 4),
    }


def main():
    args = parse_args()

    resumes, jds = load_data()

    # Resolve keywords — auto mode picks from dataset, otherwise use CLI args
    if args.auto:
        keyword, other = auto_pick_keywords(jds)
    else:
        keyword = args.keyword
        other   = args.other

    n_jobs       = args.n_jobs
    n_relevant   = args.n_relevant
    n_irrelevant = args.n_irrelevant
    k            = args.k

    print("=" * 70)
    print("TF-IDF Baseline vs Multi-Agent Pipeline")
    print(f"Keyword: {keyword}  |  Counter: {other}")
    print(f"JDs: {n_jobs}  |  Relevant: {n_relevant}  |  Irrelevant: {n_irrelevant}  |  K={k}")
    print("=" * 70)

    pairs, target_jds, candidate_pool = build_pairs(
        resumes, jds, keyword, other, n_jobs, n_relevant, n_irrelevant
    )
    print(f"Total pairs: {len(pairs)}\n")

    # TF-IDF
    print("--- Fitting TF-IDF vectorizer ---")
    tfidf_agent = TFIDFBaselineAgent()
    tfidf_agent.fit_from_parquet(
        PROCESSED / "resumes_parsed.parquet",
        PROCESSED / "jds_parsed.parquet",
    )
    tfidf_df = score_tfidf(tfidf_agent, pairs)

    # Multi-agent
    print("\n--- Running multi-agent pipeline ---")
    orchestrator = SkillMiningOrchestrator()
    ma_df        = score_multiagent(orchestrator, target_jds, candidate_pool)

    # Merge both sets of scores
    merged = tfidf_df.merge(
        ma_df[["jd_id", "resume_id", "ma_score", "matched", "explanation"]],
        on=["jd_id", "resume_id"],
        how="left",
    )
    merged["ma_score"] = merged["ma_score"].fillna(0.0)

    # Metrics
    tfidf_metrics = evaluate(merged, "tfidf_score", k)
    ma_metrics    = evaluate(merged, "ma_score",    k)

    print(f"\n{'Metric':<20}  {'TF-IDF':>10}  {'Multi-Agent':>12}")
    print("-" * 46)
    for metric in tfidf_metrics:
        tf    = tfidf_metrics[metric]
        ma    = ma_metrics[metric]
        delta = ma - tf
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "–")
        print(f"{metric:<20}  {tf:>10.4f}  {ma:>12.4f}  {arrow} {abs(delta):.4f}")

    # Per-JD breakdown
    print(f"\n--- Per-JD ranking preview (top {k} candidates) ---")
    for _, group in merged.groupby("jd_id"):
        jd_title = group["jd_title"].iloc[0][:50]
        print(f"\nJD: {jd_title}")
        top = group.sort_values("ma_score", ascending=False).head(k)
        print(f"  {'Resume':>12}  {'Rel':>3}  {'TF-IDF':>7}  {'MA':>7}  {'Matched':<25}  Explanation")
        print("  " + "-" * 95)
        for _, row in top.iterrows():
            matched = ", ".join(row["matched"][:3]) if isinstance(row["matched"], list) else ""
            expl    = str(row["explanation"])[:50] + "…" if len(str(row["explanation"])) > 50 else str(row["explanation"])
            print(
                f"  {row['resume_id'][:12]:>12}  "
                f"{int(row['relevance']):>3}  "
                f"{row['tfidf_score']:>7.4f}  "
                f"{row['ma_score']:>7.4f}  "
                f"{matched:<25}  "
                f"{expl}"
            )

    # Save
    out_path = ROOT / "outputs" / "results" / "baseline_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.drop(columns=["jd_text", "resume_text"], errors="ignore").to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
