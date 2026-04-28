"""
evaluates skill miner quality across keyword categories.

runs three checks:
1. coverage   - how many resumes get at least N skills extracted
2. precision  - do mined skills actually belong to the expected domain
3. recall     - does a Python resume score higher against Python JDs than Java JDs

results saved to outputs/results/skill_miner_evaluation.txt

usage:
    python3 src/evaluation/evaluate_skill_miner.py
    python3 src/evaluation/evaluate_skill_miner.py --keywords Python JavaScript Java --n 50
"""
import sys
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agents.skill_miner import SkillMiningAgent
from config import PROCESSED_DIR


def load_data(keywords, n_per_keyword):
    r = pd.read_parquet(PROCESSED_DIR / "resumes_parsed.parquet")
    j = pd.read_parquet(PROCESSED_DIR / "jds_parsed.parquet")

    resumes, jds = [], []
    for kw in keywords:
        r_kw = r[r["primary_keyword"] == kw].head(n_per_keyword)
        j_kw = j[j["primary_keyword"] == kw].head(5)
        resumes.append(r_kw)
        jds.append(j_kw)

    return pd.concat(resumes).reset_index(drop=True), pd.concat(jds).reset_index(drop=True)


def run_miner(resumes, agent):
    results = []
    for _, row in resumes.iterrows():
        raw_skills = list(row["raw_skills"]) if hasattr(row["raw_skills"], "tolist") else row["raw_skills"]
        mined = agent.process({
            "raw_skills": raw_skills,
            "raw_text":   row["raw_text"],
            "source":     "resume",
        })
        results.append({
            "id":             row["id"],
            "keyword":        row["primary_keyword"],
            "n_explicit":     sum(1 for e in mined["skills"] if e["source"] == "explicit"),
            "n_latent":       sum(1 for e in mined["skills"] if e["source"] == "latent"),
            "n_total":        len(mined["skills"]),
            "mined_skills":   [e["canonical"] for e in mined["skills"]],
            "avg_confidence": round(np.mean([e["confidence"] for e in mined["skills"]]), 3) if mined["skills"] else 0,
        })
    return pd.DataFrame(results)


def coverage_report(df, lines):
    lines.append("COVERAGE")
    lines.append("-" * 40)
    lines.append(f"total resumes evaluated: {len(df):,}")
    lines.append(f"resumes with 0 skills:   {(df['n_total'] == 0).sum():,} ({(df['n_total'] == 0).mean()*100:.1f}%)")
    lines.append(f"resumes with >= 3 skills: {(df['n_total'] >= 3).sum():,} ({(df['n_total'] >= 3).mean()*100:.1f}%)")
    lines.append(f"resumes with >= 5 skills: {(df['n_total'] >= 5).sum():,} ({(df['n_total'] >= 5).mean()*100:.1f}%)")
    lines.append(f"avg skills per resume:    {df['n_total'].mean():.2f}")
    lines.append(f"avg explicit:             {df['n_explicit'].mean():.2f}")
    lines.append(f"avg latent:               {df['n_latent'].mean():.2f}")
    lines.append(f"avg confidence:           {df['avg_confidence'].mean():.3f}")
    lines.append("")

    lines.append("by keyword:")
    for kw, g in df.groupby("keyword"):
        lines.append(f"  {kw:<20} avg_skills={g['n_total'].mean():.1f}  coverage={( g['n_total']>=3).mean()*100:.0f}%  avg_conf={g['avg_confidence'].mean():.3f}")
    lines.append("")


def precision_report(df, jds, keywords, lines):
    # proxy precision: what % of each resume's mined skills belong to its own domain
    # we use the most common skills in each keyword category as "expected" skills
    expected = {}
    for kw in keywords:
        kw_jds = jds[jds["primary_keyword"] == kw]
        skill_counter = Counter()
        for skills in kw_jds["required_skills"]:
            skills = skills.tolist() if hasattr(skills, "tolist") else list(skills)
            skill_counter.update(s.lower() for s in skills)
        expected[kw] = {s for s, _ in skill_counter.most_common(30)}

    lines.append("DOMAIN PRECISION (mined skills matching expected domain skills)")
    lines.append("-" * 40)
    lines.append("expected domain skills = top 30 required skills from JDs of that keyword")
    lines.append("")

    precision_scores = []
    for kw, g in df.groupby("keyword"):
        domain_skills = expected.get(kw, set())
        if not domain_skills:
            continue
        precisions = []
        for _, row in g.iterrows():
            mined = set(s.lower() for s in row["mined_skills"])
            if not mined:
                continue
            p = len(mined & domain_skills) / len(mined)
            precisions.append(p)
        if precisions:
            avg_p = np.mean(precisions)
            precision_scores.append(avg_p)
            lines.append(f"  {kw:<20} precision={avg_p:.3f}  (domain skills available: {len(domain_skills)})")

    if precision_scores:
        lines.append(f"\n  overall avg precision: {np.mean(precision_scores):.3f}")
    lines.append("")


def cross_domain_recall(df, jds, keywords, agent, lines):
    # sanity check: a Python resume should score higher against Python JDs than Java JDs
    # uses jaccard overlap between mined resume skills and JD required skills
    lines.append("CROSS-DOMAIN SANITY CHECK")
    lines.append("-" * 40)
    lines.append("same-domain overlap vs cross-domain overlap (higher same = good)")
    lines.append("")

    jd_skills_by_kw = {}
    for kw in keywords:
        kw_jds = jds[jds["primary_keyword"] == kw]
        all_skills = set()
        for skills in kw_jds["required_skills"]:
            skills = skills.tolist() if hasattr(skills, "tolist") else list(skills)
            all_skills.update(s.lower() for s in skills)
        jd_skills_by_kw[kw] = all_skills

    for kw in keywords:
        kw_resumes = df[df["keyword"] == kw]
        same_overlaps, cross_overlaps = [], []

        for _, row in kw_resumes.iterrows():
            mined = set(s.lower() for s in row["mined_skills"])
            if not mined:
                continue

            same_jd = jd_skills_by_kw.get(kw, set())
            if same_jd:
                same_overlaps.append(len(mined & same_jd) / len(mined | same_jd))

            for other_kw in keywords:
                if other_kw == kw:
                    continue
                other_jd = jd_skills_by_kw.get(other_kw, set())
                if other_jd:
                    cross_overlaps.append(len(mined & other_jd) / len(mined | other_jd))

        if same_overlaps and cross_overlaps:
            same_avg  = np.mean(same_overlaps)
            cross_avg = np.mean(cross_overlaps)
            delta     = same_avg - cross_avg
            verdict   = "PASS" if delta > 0.01 else "FAIL"
            lines.append(f"  {kw:<20} same={same_avg:.3f}  cross={cross_avg:.3f}  delta={delta:+.3f}  [{verdict}]")

    lines.append("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords", nargs="+",
                        default=["Python", "JavaScript", "Java", "DevOps", ".NET"])
    parser.add_argument("--n", type=int, default=50, help="resumes per keyword")
    args = parser.parse_args()

    print(f"loading data ({args.n} resumes per keyword)...")
    resumes, jds = load_data(args.keywords, args.n)
    print(f"resumes: {len(resumes):,} | jds: {len(jds):,}")

    print("running skill miner...")
    agent = SkillMiningAgent()
    df = run_miner(resumes, agent)

    lines = ["SKILL MINER EVALUATION", "=" * 40, ""]
    coverage_report(df, lines)
    precision_report(df, jds, args.keywords, lines)
    cross_domain_recall(df, jds, args.keywords, agent, lines)

    out = ROOT / "outputs" / "results" / "skill_miner_evaluation.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))

    for line in lines:
        print(line)

    print(f"saved to {out}")


if __name__ == "__main__":
    main()