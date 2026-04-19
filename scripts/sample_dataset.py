"""
To be removed after testing
Sample resumes and JDs from raw data by keyword filters,
then run parse + skill mine on the filtered slice.

Usage:
    PYTHONPATH=src python3 scripts/sample_dataset.py \
        --resume-keywords javascript \
        --jd-keywords "javascript" "machine learning" "ai" \
        --resume-limit 100 \
        --jd-limit 100 \
        --output-tag js_ml

Outputs (in data/processed/samples/):
    resumes_<tag>_parsed.parquet
    resumes_<tag>_mined.parquet
    jds_<tag>_parsed.parquet
"""
import argparse
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import RESUMES_RAW, JDS_RAW, PROCESSED_DIR
from agents.resume_parser import ResumeParserAgent
from agents.jd_parser import JDParserAgent
from agents.skill_miner import SkillMiningAgent

SAMPLES_DIR = PROCESSED_DIR / "samples"


def filter_resumes(df: pd.DataFrame, keywords: list, limit: int) -> pd.DataFrame:
    if not keywords:
        return df.head(limit)
    mask = pd.Series([False] * len(df), index=df.index)
    for kw in keywords:
        mask |= df["Primary Keyword"].str.lower().str.contains(kw.lower(), na=False, regex=False)
    filtered = df[mask].reset_index(drop=True)
    print(f"  Resumes matching {keywords}: {len(filtered)} → taking first {limit}")
    return filtered.head(limit)


def filter_jds(df: pd.DataFrame, keywords: list, limit: int) -> pd.DataFrame:
    if not keywords:
        return df.head(limit)
    mask = pd.Series([False] * len(df), index=df.index)
    for kw in keywords:
        mask |= df["Position"].str.lower().str.contains(kw.lower(), na=False, regex=False)
    filtered = df[mask].reset_index(drop=True)
    print(f"  JDs matching {keywords}: {len(filtered)} → taking first {limit}")
    return filtered.head(limit)


def parse_resumes(df: pd.DataFrame) -> pd.DataFrame:
    agent = ResumeParserAgent()
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing resumes"):
        try:
            result = agent.process({
                "raw_text": row["CV"],
                "id": row["id"],
                "position": row["Position"],
                "experience_years": row["Experience Years"],
                "english_level": row.get("English Level"),
            })
            result["primary_keyword"] = row.get("Primary Keyword")
            results.append(result)
        except Exception as e:
            print(f"  Skipped {row['id']}: {e}")
    return pd.DataFrame(results)


def parse_jds(df: pd.DataFrame) -> pd.DataFrame:
    agent = JDParserAgent()
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing JDs"):
        try:
            result = agent.process({
                "raw_text": row["Long Description"],
                "id": row["id"],
                "position": row["Position"],
                "company": row.get("Company Name"),
                "exp_years_raw": row.get("Exp Years"),
                "english_level": row.get("English Level"),
                "primary_keyword": row.get("Primary Keyword"),
            })
            results.append(result)
        except Exception as e:
            print(f"  Skipped JD {row['id']}: {e}")
    return pd.DataFrame(results)


def mine_skills(parsed_df: pd.DataFrame) -> pd.DataFrame:
    agent = SkillMiningAgent()
    results = []
    for _, row in tqdm(parsed_df.iterrows(), total=len(parsed_df), desc="Mining skills"):
        try:
            result = agent.process({
                "raw_skills": row.get("raw_skills", []),
                "raw_text": row.get("raw_text", ""),
                "source": "resume",
            })
            results.append({
                "id": row.get("id"),
                "mined_skills": [e["canonical"] for e in result.get("skills", [])],
                "mined_total": len(result.get("skills", [])),
                "mined_explicit": sum(1 for e in result.get("skills", []) if e.get("source") == "explicit"),
                "mined_latent": sum(1 for e in result.get("skills", []) if e.get("source") == "latent"),
            })
        except Exception as e:
            print(f"  Skipped mine {row.get('id')}: {e}")
            results.append({"id": row.get("id"), "mined_skills": [], "mined_total": 0,
                            "mined_explicit": 0, "mined_latent": 0})
    return pd.DataFrame(results)


def print_stats(label, df, skills_col="raw_skills"):
    counts = df[skills_col].apply(lambda x: len(x) if isinstance(x, list) else 0)
    print(f"\n{label} ({len(df)} records)")
    print(f"  Skills coverage : {(counts > 0).sum()} / {len(df)} = {(counts > 0).mean()*100:.1f}%")
    print(f"  Mean skills     : {counts.mean():.2f}")
    print(f"  Sparse (<=2)    : {(counts <= 2).mean()*100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-keywords", nargs="+", default=["javascript"])
    parser.add_argument("--jd-keywords", nargs="+", default=["javascript", "machine learning", "ai"])
    parser.add_argument("--resume-limit", type=int, default=100)
    parser.add_argument("--jd-limit", type=int, default=100)
    parser.add_argument("--output-tag", default="sample")
    args = parser.parse_args()

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    raw_resumes = pd.read_parquet(RESUMES_RAW)
    raw_jds = pd.read_parquet(JDS_RAW)

    print("\nFiltering...")
    r_sample = filter_resumes(raw_resumes, args.resume_keywords, args.resume_limit)
    j_sample = filter_jds(raw_jds, args.jd_keywords, args.jd_limit)

    print("\n--- Parsing resumes ---")
    resumes_parsed = parse_resumes(r_sample)
    r_out = SAMPLES_DIR / f"resumes_{args.output_tag}_parsed.parquet"
    resumes_parsed.to_parquet(r_out, index=False)
    print_stats("Resume parser output", resumes_parsed)

    print("\n--- Mining resume skills ---")
    resumes_mined = mine_skills(resumes_parsed)
    m_out = SAMPLES_DIR / f"resumes_{args.output_tag}_mined.parquet"
    resumes_mined.to_parquet(m_out, index=False)
    print_stats("After skill miner", resumes_mined, skills_col="mined_skills")

    print("\n--- Parsing JDs ---")
    jds_parsed = parse_jds(j_sample)
    j_out = SAMPLES_DIR / f"jds_{args.output_tag}_parsed.parquet"
    jds_parsed.to_parquet(j_out, index=False)

    jd_counts = jds_parsed["required_skills"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    eng_coverage = jds_parsed["english_level"].notna().sum()
    print(f"\nJD parser output ({len(jds_parsed)} records)")
    print(f"  Required skills coverage : {(jd_counts > 0).mean()*100:.1f}%")
    print(f"  Mean required skills     : {jd_counts.mean():.2f}")
    print(f"  English level coverage   : {eng_coverage}/{len(jds_parsed)}")

    # English level coverage in resumes
    eng_r = resumes_parsed["english_level"].notna().sum()
    print(f"\nResume english_level coverage: {eng_r}/{len(resumes_parsed)}")

    print(f"\nOutputs saved to {SAMPLES_DIR}/")
    print(f"  {r_out.name}")
    print(f"  {m_out.name}")
    print(f"  {j_out.name}")


if __name__ == "__main__":
    main()
