"""
Skill-IDF scorer. Weights each matched skill by how rare it is across the JD corpus.
Common skills like "JavaScript" count less; rare ones like "WebRTC" count more.

IDF(skill) = log((N+1) / (df+1)) + 1

Usage:
    # Rank multiple resumes against one JD
    PYTHONPATH=src python3 scripts/run_skill_idf.py --rank alice.txt bob.txt --jd job.txt

    # Benchmark on a keyword-filtered sample
    PYTHONPATH=src python3 scripts/run_skill_idf.py --benchmark --tag js --k 5
"""
import sys
import math
import argparse
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agents.jd_parser import JDParserAgent
from agents.skill_miner import SkillMiningAgent

PROCESSED = ROOT / "data" / "processed"
SAMPLES   = PROCESSED / "samples"


def to_list(x):
    if x is None: return []
    if isinstance(x, list): return x
    if hasattr(x, "tolist"): return x.tolist()
    return list(x)


def compute_idf(jd_skills_list):
    N = len(jd_skills_list)
    df_counts = Counter()
    for skills in jd_skills_list:
        for s in set(str(x).lower() for x in skills):
            df_counts[s] += 1
    return {sk: math.log((N + 1) / (df + 1)) + 1.0 for sk, df in df_counts.items()}


def skill_idf_score(mined, required, idf):
    if not required:
        return 0.0
    resume_set = {str(s).lower() for s in mined}
    total = matched = 0.0
    for s in required:
        w = idf.get(str(s).lower(), 1.0)
        total += w
        if str(s).lower() in resume_set:
            matched += w
    return matched / total if total else 0.0


def run_rank(resume_paths, jd_path, jd_id):
    jd_text = Path(jd_path).read_text(encoding="utf-8")
    jd_parser = JDParserAgent()
    skill_miner = SkillMiningAgent()

    parsed_jd = jd_parser.process({"raw_text": jd_text, "id": jd_id})
    jd_required = parsed_jd["required_skills"]

    # use the full parsed JD corpus for IDF if available, otherwise just this JD
    jds_path = PROCESSED / "jds_parsed.parquet"
    if jds_path.exists():
        jds_df = pd.read_parquet(jds_path)
        idf = compute_idf([to_list(x) for x in jds_df["required_skills"]])
        print(f"  IDF corpus: {len(jds_df)} JDs")
    else:
        idf = compute_idf([jd_required])
        print("  IDF corpus: single JD (uniform weights)")

    candidates = []
    for rp in resume_paths:
        text = Path(rp).read_text(encoding="utf-8")
        mined = skill_miner.process({"raw_skills": [], "raw_text": text, "source": "resume"})
        candidates.append({
            "id": Path(rp).stem,
            "skills": [e["canonical"] for e in mined.get("skills", [])],
        })

    results = []
    for c in candidates:
        s = skill_idf_score(c["skills"], jd_required, idf)
        results.append({"resume_id": c["id"], "score": round(s, 4), "skills": c["skills"]})
    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\nRanking {len(candidates)} resumes against: {jd_path}")
    print(f"JD required skills: {jd_required[:8]}\n")
    print(f"  {'Rank':>4}  {'Resume':>20}  {'Score':>8}  Matched Skills")
    print("  " + "-" * 65)
    jd_set = {str(s).lower() for s in jd_required}
    for rank, r in enumerate(results, 1):
        matched = [s for s in r["skills"] if s.lower() in jd_set]
        print(f"  {rank:>4}  {r['resume_id'][:20]:>20}  {r['score']:>8.4f}  {', '.join(matched[:5])}")


def _metrics(scores, relevance, k):
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
        if r.sum() == 0 or r.sum() == len(r):
            continue
        m = _metrics(s, r, k)
        ndcgs.append(m["ndcg"])
        precs.append(m["prec"])
        spears.append(m["spearman"])
    if not ndcgs:
        return {f"ndcg@{k}": float("nan"), f"prec@{k}": float("nan"), "spearman": float("nan")}
    return {
        f"ndcg@{k}": round(np.nanmean(ndcgs), 4),
        f"prec@{k}": round(np.nanmean(precs), 4),
        "spearman": round(np.nanmean(spears), 4),
    }


def run_benchmark(tag, k):
    r_parsed = pd.read_parquet(SAMPLES / f"resumes_{tag}_parsed.parquet")
    r_mined  = pd.read_parquet(SAMPLES / f"resumes_{tag}_mined.parquet")
    jds      = pd.read_parquet(SAMPLES / f"jds_{tag}_parsed.parquet")

    resumes = r_parsed.merge(r_mined[["id", "mined_skills"]], on="id", how="left")
    resumes["mined_skills"] = resumes["mined_skills"].apply(to_list)
    jds["required_skills"]  = jds["required_skills"].apply(to_list)

    idf = compute_idf(jds["required_skills"].tolist())
    print(f"IDF table: {len(idf)} skills from {len(jds)} JDs\n")

    rows = []
    for _, jd in jds.iterrows():
        for _, res in resumes.iterrows():
            s = skill_idf_score(res["mined_skills"], jd["required_skills"], idf)
            rows.append({
                "resume_id": str(res["id"]),
                "jd_id": str(jd["id"]),
                "skill_idf_score": round(s, 4),
                "relevance": 1 if res.get("primary_keyword") == jd.get("primary_keyword") else 0,
            })
    df = pd.DataFrame(rows)

    metrics = evaluate(df, "skill_idf_score", k)
    for m, v in metrics.items():
        print(f"  {m:<14} {v:.4f}")

    out = ROOT / "outputs" / "results" / f"skill_idf_benchmark_{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Skill-IDF scorer")
    p.add_argument("--rank",      type=str, nargs="+", metavar="RESUME")
    p.add_argument("--jd",        type=str)
    p.add_argument("--jd-id",     type=str, default="jd-001")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--tag",       type=str, default="js")
    p.add_argument("--k",         type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.benchmark:
        run_benchmark(args.tag, args.k)
    elif args.rank:
        if not args.jd:
            print("Error: --rank requires --jd"); raise SystemExit(1)
        run_rank(args.rank, args.jd, args.jd_id)
    else:
        print("Provide --rank/--jd or --benchmark. Use -h for help.")
