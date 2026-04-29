"""
Build a fixed, frozen evaluation dataset for all evaluation and visualization scripts.

Structure: 10 keywords x 25 JDs x 20 candidates = 5,000 pairs
  Per JD: 10 relevant + 5 hard irrelevant (adjacent domain) + 5 easy irrelevant (different domain)

Quality filters:
  JDs:     required_skills >= 5, raw_text >= 400 chars, title + experience_years not null
  Resumes: raw_text >= 300 chars, raw_skills >= 3, experience_years not null

Run once. All evaluation scripts load from data/test/eval_pairs.parquet.

Usage:
    python3 scripts/build_eval_dataset.py
    python3 scripts/build_eval_dataset.py --force   # rebuild even if file exists
"""
import sys
import json
import random
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import PROCESSED_DIR

SEED         = 42
N_JDS        = 25
N_RELEVANT   = 10
N_HARD_IRREL = 5
N_EASY_IRREL = 5

KEYWORDS = [
    "Python", "JavaScript", "Java", "DevOps", "Data Science",
    "Node.js", "PHP", ".NET", "Data Engineer", "Golang",
]

HARD_NEGATIVE_MAP = {
    "Python":        ["JavaScript", "Java"],
    "JavaScript":    ["PHP", "Node.js"],
    "Java":          [".NET", "Golang"],
    "DevOps":        ["Data Engineer", "Golang"],
    "Data Science":  ["Data Engineer", "Python"],
    "Node.js":       ["JavaScript", "PHP"],
    "PHP":           ["JavaScript", "Node.js"],
    ".NET":          ["Java", "Golang"],
    "Data Engineer": ["DevOps", "Data Science"],
    "Golang":        ["Java", ".NET"],
}

EASY_NEGATIVE_POOL = ["QA", "QA Automation", "iOS", "Business Analyst", "Unity"]

OUT_PATH = ROOT / "data" / "test" / "eval_pairs.parquet"


def quality_filter_jds(df):
    return df[
        (df["required_skills"].apply(len) >= 5) &
        (df["raw_text"].str.len() >= 400) &
        (df["title"].notna()) &
        (df["experience_years"].notna())
    ].copy()


def quality_filter_resumes(df):
    return df[
        (df["raw_text"].str.len() >= 300) &
        (df["raw_skills"].apply(len) >= 3) &
        (df["experience_years"].notna())
    ].copy()


def sample_with_cycle(pool_df, n, seed):
    """Sample n rows from pool, cycling through if pool is smaller than n."""
    if pool_df.empty:
        return pd.DataFrame()
    rng = random.Random(seed)
    ids = pool_df["id"].tolist()
    rng.shuffle(ids)
    sampled_ids = [ids[i % len(ids)] for i in range(n)]
    # build result preserving order, allowing duplicates when cycling
    return pool_df.set_index("id").loc[sampled_ids].reset_index()


def _clean_skills(skills_arr):
    """Normalize skill strings to lowercase tokens for overlap computation."""
    result = set()
    for s in skills_arr:
        s = str(s).lower().strip()
        if len(s) >= 2:
            result.add(s)
            # also add individual words so multi-word skills get partial credit
            for word in s.split():
                if len(word) >= 3:
                    result.add(word)
    return result


def _skill_overlap_ratio(resume_skills, jd_skills):
    """Compute what fraction of JD skill tokens appear in resume skills."""
    rs = _clean_skills(resume_skills)
    js = _clean_skills(jd_skills)
    if not js:
        return 0.0
    return len(rs & js) / len(js)


# threshold for splitting relevant pairs into strong vs partial match
# tuned so roughly half of same-domain pairs get grade 2
STRONG_MATCH_THRESHOLD = 0.05


def make_pair(keyword, jd_row, resume_row, relevance, difficulty):
    # compute skill overlap for graded relevance
    resume_skills = resume_row.get("raw_skills", [])
    jd_skills = list(jd_row.get("required_skills", []))
    if hasattr(resume_skills, "tolist"):
        resume_skills = resume_skills.tolist()
    if hasattr(jd_skills, "tolist"):
        jd_skills = jd_skills.tolist()

    overlap = _skill_overlap_ratio(resume_skills, jd_skills)

    # graded relevance: 0=poor, 1=partial, 2=strong
    # same domain + high overlap = 2, same domain + low overlap = 1, diff domain = 0
    if relevance == 1 and overlap >= STRONG_MATCH_THRESHOLD:
        graded = 2
    elif relevance == 1:
        graded = 1
    else:
        graded = 0

    return {
        "keyword":              keyword,
        "difficulty":           difficulty,
        "jd_id":                jd_row["id"],
        "jd_text":              jd_row["raw_text"],
        "jd_title":             jd_row["title"],
        "jd_exp_years":         jd_row["experience_years"],
        "resume_id":            resume_row["id"],
        "resume_text":          resume_row["raw_text"],
        "resume_position":      resume_row.get("position", ""),
        "resume_exp_years":     resume_row["experience_years"],
        "resume_english_level": resume_row.get("english_level", ""),
        "skill_overlap":        round(overlap, 4),
        "graded_relevance":     graded,
        "relevance":            relevance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild even if eval_pairs.parquet exists")
    args = parser.parse_args()

    if OUT_PATH.exists() and not args.force:
        print(f"eval_pairs.parquet already exists at {OUT_PATH}")
        print("Use --force to rebuild.")
        df = pd.read_parquet(OUT_PATH)
        print(f"Current dataset: {len(df):,} pairs")
        return

    print("Loading parsed data...")
    resumes_all = pd.read_parquet(PROCESSED_DIR / "resumes_parsed.parquet")
    jds_all     = pd.read_parquet(PROCESSED_DIR / "jds_parsed.parquet")
    resumes_all["id"] = resumes_all["id"].astype(str)
    jds_all["id"]     = jds_all["id"].astype(str)
    print(f"  Loaded: {len(resumes_all):,} resumes, {len(jds_all):,} jds")

    # restrict to held-out test split only
    split_path = ROOT / "data" / "test" / "test_split.json"
    if split_path.exists():
        split = json.load(open(split_path))
        test_r_ids = set(rid for ids in split["resume_test_ids"].values() for rid in ids)
        test_j_ids = set(jid for ids in split["jd_test_ids"].values()     for jid in ids)
        resumes_all = resumes_all[resumes_all["id"].isin(test_r_ids)]
        jds_all     = jds_all[jds_all["id"].isin(test_j_ids)]
        print(f"  After test split filter: {len(resumes_all):,} resumes, {len(jds_all):,} jds")
    else:
        print("  WARNING: no test_split.json — using full dataset (training data may leak in)")

    # quality filter
    resumes_q = quality_filter_resumes(resumes_all)
    jds_q     = quality_filter_jds(jds_all)
    print(f"  After quality filter: {len(resumes_q):,} resumes, {len(jds_q):,} jds")

    # easy negative pool — completely different domain
    easy_pool = resumes_q[resumes_q["primary_keyword"].isin(EASY_NEGATIVE_POOL)].copy()
    print(f"  Easy negative pool: {len(easy_pool):,} resumes from {EASY_NEGATIVE_POOL}")

    all_pairs   = []
    summary     = []

    for kw in KEYWORDS:
        kw_jds     = jds_q[jds_q["primary_keyword"] == kw].copy()
        kw_resumes = resumes_q[resumes_q["primary_keyword"] == kw].copy()
        hard_pool  = resumes_q[resumes_q["primary_keyword"].isin(HARD_NEGATIVE_MAP[kw])].copy()
        easy_kw    = easy_pool[easy_pool["primary_keyword"] != kw].copy()
        if easy_kw.empty:
            easy_kw = easy_pool.copy()

        # pick JDs
        n_jds = min(N_JDS, len(kw_jds))
        if n_jds < N_JDS:
            print(f"  WARNING: {kw} only has {n_jds} quality JDs (wanted {N_JDS})")
        jd_sample = kw_jds.sample(n_jds, random_state=SEED)

        kw_pairs = 0
        for jd_idx, (_, jd_row) in enumerate(jd_sample.iterrows()):
            jd_seed = SEED + jd_idx * 3

            rel   = sample_with_cycle(kw_resumes, N_RELEVANT,   jd_seed)
            hard  = sample_with_cycle(hard_pool,  N_HARD_IRREL, jd_seed + 1)
            easy  = sample_with_cycle(easy_kw,    N_EASY_IRREL, jd_seed + 2)

            for _, r in rel.iterrows():
                all_pairs.append(make_pair(kw, jd_row, r, 1, "relevant"))
            for _, r in hard.iterrows():
                all_pairs.append(make_pair(kw, jd_row, r, 0, "hard"))
            for _, r in easy.iterrows():
                all_pairs.append(make_pair(kw, jd_row, r, 0, "easy"))

            kw_pairs += len(rel) + len(hard) + len(easy)

        summary.append({
            "keyword":     kw,
            "jds":         n_jds,
            "relevant":    n_jds * N_RELEVANT,
            "hard_irrel":  n_jds * N_HARD_IRREL,
            "easy_irrel":  n_jds * N_EASY_IRREL,
            "total":       kw_pairs,
        })
        print(f"  {kw:<15} {n_jds} JDs  {kw_pairs} pairs")

    df = pd.DataFrame(all_pairs)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    # summary
    print("\n" + "=" * 65)
    print("EVAL DATASET BUILT")
    print("=" * 65)
    sum_df = pd.DataFrame(summary)
    print(sum_df.to_string(index=False))
    print(f"\nTotal pairs:       {len(df):,}")
    print(f"  Relevant:        {(df['relevance'] == 1).sum():,}")
    print(f"  Hard irrelevant: {(df['difficulty'] == 'hard').sum():,}")
    print(f"  Easy irrelevant: {(df['difficulty'] == 'easy').sum():,}")
    print(f"\nGraded relevance distribution:")
    print(f"  Grade 2 (strong match): {(df['graded_relevance'] == 2).sum():,}")
    print(f"  Grade 1 (partial match): {(df['graded_relevance'] == 1).sum():,}")
    print(f"  Grade 0 (poor match):   {(df['graded_relevance'] == 0).sum():,}")
    print(f"\nData quality:")
    print(f"  avg JD text len:     {df['jd_text'].str.len().mean():.0f} chars")
    print(f"  avg resume text len: {df['resume_text'].str.len().mean():.0f} chars")
    print(f"  avg skill overlap (relevant): {df[df['relevance']==1]['skill_overlap'].mean():.3f}")
    print(f"  avg skill overlap (irrelevant): {df[df['relevance']==0]['skill_overlap'].mean():.3f}")
    print(f"  JDs with exp_years:  {df['jd_exp_years'].notna().sum():,} / {len(df):,}")
    print(f"  Resumes with exp:    {df['resume_exp_years'].notna().sum():,} / {len(df):,}")
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
