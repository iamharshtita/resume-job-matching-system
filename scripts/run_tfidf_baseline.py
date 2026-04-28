"""
TF-IDF baseline scorer. Use this to compare against the multi-agent pipeline.

Usage:
    # Score a single resume vs JD
    python3 scripts/run_tfidf_baseline.py --resume resume.txt --jd job.txt

    # Rank multiple resumes against one JD
    python3 scripts/run_tfidf_baseline.py --rank alice.txt bob.txt --jd job.txt

    # Benchmark on the parsed dataset
    python3 scripts/run_tfidf_baseline.py --benchmark --auto
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

from baselines.tfidf_baseline import TFIDFBaselineAgent

PROCESSED = ROOT / "data" / "processed"


def run_single(resume_path, jd_path, resume_id, jd_id):
    resume_text = Path(resume_path).read_text(encoding="utf-8")
    jd_text = Path(jd_path).read_text(encoding="utf-8")

    agent = TFIDFBaselineAgent()
    agent.fit([resume_text, jd_text])
    score = agent.score_pair(resume_text, jd_text)

    print(f"\nResume : {resume_path}  [{resume_id}]")
    print(f"JD     : {jd_path}  [{jd_id}]")
    print(f"Score  : {score:.4f}")


def run_rank(resume_paths, jd_path, jd_id):
    jd_text = Path(jd_path).read_text(encoding="utf-8")
    candidates = [(Path(p).stem, Path(p).read_text(encoding="utf-8")) for p in resume_paths]

    agent = TFIDFBaselineAgent()
    agent.fit([text for _, text in candidates] + [jd_text])

    results = agent.score_batch([(rid, jd_id, rtext, jd_text) for rid, rtext in candidates])
    results.sort(key=lambda x: x["tfidf_score"], reverse=True)

    print(f"\nRanking {len(candidates)} resumes against: {jd_path}\n")
    print(f"  {'Rank':>4}  {'Resume':>20}  {'Score':>8}")
    print("  " + "-" * 36)
    for rank, r in enumerate(results, 1):
        print(f"  {rank:>4}  {r['resume_id'][:20]:>20}  {r['tfidf_score']:>8.4f}")


def auto_pick_keywords(jds):
    top = jds["primary_keyword"].value_counts().head(3).index.tolist()
    kw, other = top[0], top[1]
    print(f"Auto-selected: '{kw}' (relevant) vs '{other}' (irrelevant)")
    return kw, other


def build_pairs(resumes, jds, keyword, other, n_jobs, n_relevant, n_irrelevant):
    target_jds = jds[jds["primary_keyword"] == keyword]
    target_jds = target_jds[target_jds["required_skills"].apply(len) >= 3].iloc[:n_jobs]
    if len(target_jds) == 0:
        raise ValueError(f"No JDs found for keyword '{keyword}'.")

    rel_r = resumes[resumes["primary_keyword"] == keyword].iloc[:n_relevant]
    irrel_r = resumes[resumes["primary_keyword"] == other].iloc[:n_irrelevant]
    if len(irrel_r) == 0:
        raise ValueError(f"No resumes found for counter keyword '{other}'.")

    pool = pd.concat([rel_r, irrel_r]).reset_index(drop=True)
    pairs = []
    for _, jd in target_jds.iterrows():
        for _, res in pool.iterrows():
            pairs.append({
                "jd_id": str(jd["id"]),
                "jd_title": jd["title"],
                "jd_text": jd["raw_text"],
                "resume_id": str(res["id"]),
                "resume_text": res["raw_text"],
                "relevance": 1 if res["primary_keyword"] == keyword else 0,
            })
    return pairs, target_jds, pool


def compute_metrics(scores, relevance, k):
    order = np.argsort(-scores)
    prec = relevance[order][:k].sum() / k
    try:
        ndcg = ndcg_score(relevance.reshape(1, -1), scores.reshape(1, -1), k=k)
    except Exception:
        ndcg = float("nan")
    try:
        rho, _ = spearmanr(scores, relevance)
    except Exception:
        rho = float("nan")
    return {"ndcg": ndcg, "prec": prec, "spearman": rho}


def evaluate(df, score_col, k):
    ndcgs, precs, spears = [], [], []
    for _, group in df.groupby("jd_id"):
        s = group[score_col].values.astype(float)
        r = group["relevance"].values.astype(float)
        m = compute_metrics(s, r, k)
        ndcgs.append(m["ndcg"])
        precs.append(m["prec"])
        spears.append(m["spearman"])
    return {
        f"ndcg@{k}": round(np.nanmean(ndcgs), 4),
        f"prec@{k}": round(np.nanmean(precs), 4),
        "spearman": round(np.nanmean(spears), 4),
    }


def run_benchmark(keyword, other, n_jobs, n_relevant, n_irrelevant, k, auto):
    resumes = pd.read_parquet(PROCESSED / "resumes_parsed.parquet")
    jds = pd.read_parquet(PROCESSED / "jds_parsed.parquet")

    if auto:
        keyword, other = auto_pick_keywords(jds)

    print(f"TF-IDF Benchmark  —  '{keyword}' vs '{other}'  |  K={k}")
    pairs, _, _ = build_pairs(resumes, jds, keyword, other, n_jobs, n_relevant, n_irrelevant)
    print(f"Total pairs: {len(pairs)}\n")

    agent = TFIDFBaselineAgent()
    agent.fit_from_parquet(PROCESSED / "resumes_parsed.parquet", PROCESSED / "jds_parsed.parquet")

    t0 = time.perf_counter()
    batch = [(p["resume_id"], p["jd_id"], p["resume_text"], p["jd_text"]) for p in pairs]
    res = agent.score_batch(batch)
    print(f"Scored {len(pairs)} pairs in {time.perf_counter()-t0:.2f}s")

    df = pd.DataFrame(res)
    df["relevance"] = [p["relevance"] for p in pairs]
    df["jd_title"] = [p["jd_title"] for p in pairs]

    metrics = evaluate(df, "tfidf_score", k)
    for m, v in metrics.items():
        print(f"  {m:<14} {v:.4f}")

    out = ROOT / "outputs" / "results" / "tfidf_benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


def parse_args():
    p = argparse.ArgumentParser(description="TF-IDF baseline scorer")
    p.add_argument("--resume",       type=str)
    p.add_argument("--jd",           type=str)
    p.add_argument("--resume-id",    type=str, default="resume-001")
    p.add_argument("--jd-id",        type=str, default="jd-001")
    p.add_argument("--rank",         type=str, nargs="+", metavar="RESUME")
    p.add_argument("--benchmark",    action="store_true")
    p.add_argument("--auto",         action="store_true")
    p.add_argument("--keyword",      type=str, default="Python")
    p.add_argument("--other",        type=str, default="Design")
    p.add_argument("--n-jobs",       type=int, default=5)
    p.add_argument("--n-relevant",   type=int, default=5)
    p.add_argument("--n-irrelevant", type=int, default=5)
    p.add_argument("--k",            type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.benchmark:
        run_benchmark(args.keyword, args.other, args.n_jobs,
                      args.n_relevant, args.n_irrelevant, args.k, args.auto)
    elif args.rank:
        if not args.jd:
            print("Error: --rank requires --jd"); raise SystemExit(1)
        run_rank(args.rank, args.jd, args.jd_id)
    elif args.resume:
        if not args.jd:
            print("Error: --resume requires --jd"); raise SystemExit(1)
        run_single(args.resume, args.jd, args.resume_id, args.jd_id)
    else:
        print("Provide --resume/--jd, --rank/--jd, or --benchmark. Use -h for help.")
