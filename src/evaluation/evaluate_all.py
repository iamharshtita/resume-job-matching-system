"""
Unified evaluation - compares all three methods on the same fixed test set.

Runs:
1. TF-IDF Baseline
2. Skill-IDF Baseline
3. Multi-Agent Pipeline (with IDF-weighted matching)

Reads from data/test/eval_pairs.parquet (build it first with scripts/build_eval_dataset.py)
Outputs comparison_results.csv, detailed_scores.csv, avg_comparison_results.csv

Usage:
    python3 src/evaluation/evaluate_all.py
    python3 src/evaluation/evaluate_all.py --k 10
"""
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from baselines.tfidf_baseline import TFIDFBaselineAgent
from agents.orchestrator import SkillMiningOrchestrator
from config import PROCESSED_DIR, RESULTS_DIR

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

    for _, group in df.groupby("jd_id"):
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

EVAL_PAIRS_PATH = ROOT / "data" / "test" / "eval_pairs.parquet"


def _load_eval_pairs():
    if not EVAL_PAIRS_PATH.exists():
        print(f"eval_pairs.parquet not found at {EVAL_PAIRS_PATH}")
        print("run scripts/build_eval_dataset.py first")
        sys.exit(1)
    df = pd.read_parquet(EVAL_PAIRS_PATH)
    print(f"loaded {len(df):,} pairs from eval_pairs.parquet")
    return df.to_dict("records")


def _run_all_methods(pairs, k):
    timings = {}

    t0 = time.time()
    tfidf_scores = run_tfidf_baseline(pairs)
    timings["tfidf"] = time.time() - t0
    print(f"  tfidf done in {timings['tfidf']:.1f}s")

    t0 = time.time()
    skill_idf_scores = run_skill_idf_baseline(pairs)
    timings["skill_idf"] = time.time() - t0
    print(f"  skill-idf done in {timings['skill_idf']:.1f}s")

    t0 = time.time()
    multi_agent_scores = run_multi_agent_pipeline(pairs)
    timings["multi_agent"] = time.time() - t0
    print(f"  multi-agent done in {timings['multi_agent']:.1f}s")

    rows = []
    for p in pairs:
        key = (p["resume_id"], p["jd_id"])
        rows.append({
            "keyword":           p["keyword"],
            "difficulty":        p["difficulty"],
            "resume_id":         p["resume_id"],
            "jd_id":             p["jd_id"],
            "relevance":         p["relevance"],
            "tfidf_score":       tfidf_scores.get(key, 0.0),
            "skill_idf_score":   skill_idf_scores.get(key, 0.0),
            "multi_agent_score": multi_agent_scores.get(key, 0.0),
        })

    df = pd.DataFrame(rows)
    return df, timings


def _print_keyword_table(keyword, metrics, timings, k):
    print(f"\n  {keyword}")
    print(f"  {'Method':<20} {'NDCG@'+str(k):>10} {'P@'+str(k):>10} {'R@'+str(k):>10} {'MAP':>10} {'Time(s)':>10}")
    print(f"  {'-'*65}")
    for label, key in [("TF-IDF", "tfidf"), ("Skill-IDF", "skill_idf"), ("Multi-Agent+IDF", "multi_agent")]:
        m = metrics[key]
        print(f"  {label:<20} {m[f'ndcg@{k}']:>10.4f} {m[f'prec@{k}']:>10.4f} "
              f"{m[f'rec@{k}']:>10.4f} {m['map']:>10.4f} {timings[key]:>10.2f}")


def _print_summary_table(per_keyword_metrics, k):
    methods = [("TF-IDF", "tfidf"), ("Skill-IDF", "skill_idf"), ("Multi-Agent+IDF", "multi_agent")]
    metric_keys = [f"ndcg@{k}", f"prec@{k}", f"rec@{k}", "map"]

    print(f"\n{'Method':<20} {'NDCG@'+str(k):>10} {'P@'+str(k):>10} {'R@'+str(k):>10} {'MAP':>10}")
    print("-" * 55)
    for label, key in methods:
        vals = [per_keyword_metrics[kw][key] for kw in per_keyword_metrics]
        avgs = {mk: round(float(np.mean([v[mk] for v in vals])), 4) for mk in metric_keys}
        print(f"{label:<20} {avgs[f'ndcg@{k}']:>10.4f} {avgs[f'prec@{k}']:>10.4f} "
              f"{avgs[f'rec@{k}']:>10.4f} {avgs['map']:>10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation of all methods")
    parser.add_argument("--k", type=int, default=5, help="top-k for metrics (default 5)")
    args = parser.parse_args()

    print("=" * 70)
    print("UNIFIED EVALUATION - ALL METHODS")
    print("=" * 70)

    pairs = _load_eval_pairs()
    keywords = sorted(set(p["keyword"] for p in pairs))
    print(f"keywords: {', '.join(keywords)}")

    print("\n" + "=" * 70)
    print("RUNNING EVALUATIONS")
    print("=" * 70)

    df, timings = _run_all_methods(pairs, args.k)

    # per keyword metrics
    per_keyword_metrics = {}
    summary_rows = []

    for kw in keywords:
        kw_df = df[df["keyword"] == kw]
        metrics = {
            "tfidf":       evaluate_method(kw_df, "tfidf_score", args.k),
            "skill_idf":   evaluate_method(kw_df, "skill_idf_score", args.k),
            "multi_agent": evaluate_method(kw_df, "multi_agent_score", args.k),
        }
        per_keyword_metrics[kw] = metrics
        _print_keyword_table(kw, metrics, timings, args.k)

        for label, key in [("TF-IDF", "tfidf"), ("Skill-IDF", "skill_idf"), ("Multi-Agent+IDF", "multi_agent")]:
            summary_rows.append({
                "keyword": kw,
                "method": label,
                **metrics[key],
                "time": timings[key],
            })

    print("\n" + "=" * 70)
    print(f"AVERAGE ACROSS {len(per_keyword_metrics)} KEYWORDS")
    print("=" * 70)
    _print_summary_table(per_keyword_metrics, args.k)

    # save outputs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "comparison_results.csv", index=False)
    print(f"\nper-keyword results saved to: {RESULTS_DIR / 'comparison_results.csv'}")

    df.to_csv(RESULTS_DIR / "detailed_scores.csv", index=False)
    print(f"detailed scores saved to: {RESULTS_DIR / 'detailed_scores.csv'}")

    # macro average
    metric_keys = [f"ndcg@{args.k}", f"prec@{args.k}", f"rec@{args.k}", "map"]
    avg_rows = []
    for label, key in [("TF-IDF", "tfidf"), ("Skill-IDF", "skill_idf"), ("Multi-Agent+IDF", "multi_agent")]:
        vals = [per_keyword_metrics[kw][key] for kw in per_keyword_metrics]
        avg_rows.append({
            "method": label,
            "n_keywords": len(per_keyword_metrics),
            **{mk: round(float(np.mean([v[mk] for v in vals])), 4) for mk in metric_keys},
        })
    pd.DataFrame(avg_rows).to_csv(RESULTS_DIR / "avg_comparison_results.csv", index=False)
    print(f"macro-average saved to: {RESULTS_DIR / 'avg_comparison_results.csv'}")

    # graded relevance evaluation (if graded_relevance column exists in eval pairs)
    # this uses skill-overlap based labels instead of binary keyword matching
    ep = pd.read_parquet(EVAL_PAIRS_PATH, columns=["resume_id", "jd_id", "graded_relevance"])
    if "graded_relevance" in ep.columns:
        graded_df = df.merge(ep, on=["resume_id", "jd_id"], how="left")
        if graded_df["graded_relevance"].notna().all():
            print("\n" + "=" * 70)
            print(f"GRADED RELEVANCE NDCG@{args.k} (skill-overlap based labels)")
            print("=" * 70)
            print("  Labels: 0=poor match, 1=partial (same domain), 2=strong (same domain + skill overlap)")

            graded_results = {}
            for col, name in [("tfidf_score", "TF-IDF"), ("skill_idf_score", "Skill-IDF"), ("multi_agent_score", "Multi-Agent+IDF")]:
                ndcgs = []
                for _, group in graded_df.groupby("jd_id"):
                    scores = group[col].values.astype(float)
                    graded = group["graded_relevance"].values.astype(float)
                    try:
                        ndcg = ndcg_score(graded.reshape(1, -1), scores.reshape(1, -1), k=args.k)
                        ndcgs.append(ndcg)
                    except:
                        pass
                graded_results[name] = round(np.nanmean(ndcgs), 4)

            print(f"\n  {'Method':<20} {'Binary NDCG':>12} {'Graded NDCG':>12} {'Difference':>12}")
            print("  " + "-" * 58)
            for name in ["TF-IDF", "Skill-IDF", "Multi-Agent+IDF"]:
                binary = float(avg_rows[[r for r in range(len(avg_rows)) if avg_rows[r]["method"] == name][0]][f"ndcg@{args.k}"])
                graded = graded_results[name]
                diff = graded - binary
                print(f"  {name:<20} {binary:>12.4f} {graded:>12.4f} {diff:>+12.4f}")

            # save graded results
            graded_rows = [{"method": name, f"graded_ndcg@{args.k}": val} for name, val in graded_results.items()]
            pd.DataFrame(graded_rows).to_csv(RESULTS_DIR / "graded_relevance_results.csv", index=False)
            print(f"\n  graded results saved to: {RESULTS_DIR / 'graded_relevance_results.csv'}")


if __name__ == "__main__":
    main()
