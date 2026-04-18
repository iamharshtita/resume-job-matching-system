"""
Runs the full pipeline end-to-end.

Single-pair mode (default):
    python scripts/run_full_pipeline.py

Batch ranking mode (ranks N resumes against one JD from the real dataset):
    python scripts/run_full_pipeline.py --batch --n 20
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from loguru import logger
from agents.orchestrator import SkillMiningOrchestrator

# Sample data
SAMPLE_RESUME = """
Alex Chen
Senior Software Engineer

Experience:
2019 – Present  Senior Backend Engineer at DataStream Inc
2016 – 2019     Software Engineer at WebCo

Skills: Python, FastAPI, PostgreSQL, Docker, Kubernetes, Redis, AWS, Git, SQL, Pytest

Education:
Bachelor's degree in Computer Science, University of Toronto 2016
"""

SAMPLE_JD = """
Senior Python Backend Engineer

Requirements:
- 4+ years experience with Python
- FastAPI or Django REST framework
- PostgreSQL and Redis
- Docker, Kubernetes
- AWS or GCP
- Bachelor's degree in Computer Science

Nice to have:
- Experience with Kafka or RabbitMQ
- Familiarity with Terraform
"""


def run_single():
    orchestrator = SkillMiningOrchestrator()
    result = orchestrator.run(
        resume_text=SAMPLE_RESUME,
        jd_text=SAMPLE_JD,
        resume_id="sample-resume-001",
        jd_id="sample-jd-001",
    )

    print("\n" + "=" * 60)
    print("PARSED RESUME")
    print("=" * 60)
    r = result["parsed_resume"]
    print(f"  Position       : {r['position']}")
    print(f"  Exp years      : {r['experience_years']}")
    print(f"  Raw skills     : {r['raw_skills']}")
    print(f"  Education      : {[e['degree'] for e in r['education']]}")
    print(f"  Experience blks: {len(r['experience'])}")

    print("\n" + "=" * 60)
    print("PARSED JD")
    print("=" * 60)
    j = result["parsed_jd"]
    print(f"  Title          : {j['title']}")
    print(f"  Exp years req  : {j['experience_years']}")
    print(f"  Required skills: {j['required_skills']}")
    print(f"  Preferred skills:{j['preferred_skills']}")
    print(f"  Education req  : {j['education_requirement']}")

    print("\n" + "=" * 60)
    print("MINED SKILLS")
    print("=" * 60)
    rs = result["resume_skills"]
    js = result["jd_skills"]
    print(f"  Resume canonical skills ({len(rs['skills'])}):")
    for e in rs["skills"][:10]:
        print(f"    {e['skill']:20s} → {e['canonical']:20s}  [{e['category']}  conf={e['confidence']}]")
    print(f"  JD canonical skills ({len(js['skills'])}):")
    for e in js["skills"][:10]:
        print(f"    {e['skill']:20s} → {e['canonical']:20s}  [{e['category']}  conf={e['confidence']}]")

    print("\n" + "=" * 60)
    print("MATCH SCORES")
    print("=" * 60)
    m = result["match"]
    print(f"  Skill score    : {m['skill_score']:.4f}  (weight 0.40)")
    print(f"  Experience     : {m['experience_score']:.4f}  (weight 0.30)")
    print(f"  Education      : {m['education_score']:.4f}  (weight 0.20)")
    print(f"  Title          : {m['title_score']:.4f}  (weight 0.10)")
    print(f"  ─────────────────────────")
    print(f"  FINAL SCORE    : {m['final_score']:.4f}")
    print(f"  Jaccard (raw)  : {m['jaccard_score']:.4f}")
    print("=" * 60)


def run_batch(n: int):
    import pandas as pd

    resumes_path = ROOT / "data" / "processed" / "resumes_parsed.parquet"
    jds_path     = ROOT / "data" / "processed" / "jds_parsed.parquet"

    if not resumes_path.exists() or not jds_path.exists():
        logger.error("Parsed data not found. Run scripts/parse_resumes.py and scripts/parse_jds.py first.")
        sys.exit(1)

    resumes_df = pd.read_parquet(resumes_path)
    jds_df     = pd.read_parquet(jds_path)

    # Pick a Python JD with several required skills
    py_jds = jds_df[jds_df["primary_keyword"] == "Python"]
    py_jds = py_jds[py_jds["required_skills"].apply(lambda x: len(x) >= 5)]
    jd_row = py_jds.iloc[0]

    # Pick a mix of Python and non-Python resumes
    py_r  = resumes_df[resumes_df["primary_keyword"] == "Python"].head(n // 2)
    js_r  = resumes_df[resumes_df["primary_keyword"] == "JavaScript"].head(n // 2)
    sample = pd.concat([py_r, js_r]).reset_index(drop=True)

    candidates = [(str(row["id"]), row["raw_text"]) for _, row in sample.iterrows()]

    orchestrator = SkillMiningOrchestrator()

    print(f"\nRanking {len(candidates)} candidates against JD: {jd_row['title']}")
    print(f"Required skills: {list(jd_row['required_skills'])}")
    print()

    ranked = orchestrator.rank_candidates(
        jd_text=jd_row["raw_text"],
        candidates=candidates,
        jd_id=str(jd_row["id"]),
    )

    print(f"\n{'Rank':>4}  {'Resume ID':>12}  {'Final':>6}  {'Matched Skills':<30}  Explanation")
    print("-" * 100)
    for r in ranked:
        matched = ", ".join(r["matched_skills"][:4]) + ("..." if len(r["matched_skills"]) > 4 else "")
        explanation = r["explanation"][:70] + ("..." if len(r["explanation"]) > 70 else "")
        print(
            f"{r['rank']:>4}  "
            f"{r['resume_id'][:12]:>12}  "
            f"{r['final_score']:>6.3f}  "
            f"{matched:<30}  "
            f"{explanation}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true", help="Run batch ranking mode")
    parser.add_argument("--n", type=int, default=20, help="Number of candidates in batch mode")
    args = parser.parse_args()

    if args.batch:
        run_batch(args.n)
    else:
        run_single()
