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

# default demo data - override with --resume and --jd

SAMPLE_RESUME = """
Dmytro Kovalenko
Senior Python Backend Engineer
5 years of experience
English level: Upper

Work Experience:

2021 - Present   Senior Backend Engineer at Growe (Fintech, Kyiv)
  - Designed and maintained REST APIs using Python and FastAPI serving 2M+ users
  - Built data pipelines with Apache Kafka and Celery for real-time event processing
  - Managed PostgreSQL and Redis infrastructure, reduced query latency by 40%
  - Containerized services with Docker and deployed on AWS using Kubernetes
  - Led a team of 3 backend engineers, conducted code reviews

2019 - 2021   Python Developer at Datagroup (Telecom, Kyiv)
  - Developed internal tooling using Django and DRF
  - Wrote unit and integration tests with Pytest, maintained 85% coverage
  - Worked with MySQL and MongoDB for different project needs
  - Used Git, CI/CD pipelines (Jenkins), Linux environments

Skills:
Python, FastAPI, Django, PostgreSQL, Redis, Docker, Kubernetes, AWS,
Apache Kafka, Celery, REST API, Pytest, Git, Linux, SQL, MongoDB

Education:
Bachelor of Science in Computer Science, Kyiv Polytechnic Institute, 2019
"""

SAMPLE_JD = """
Senior Python Backend Engineer

About the role:
We are looking for a Senior Python Backend Engineer to join our platform team
building high-load financial services used by millions of customers.

Requirements:
- 4+ years of experience in Python backend development
- Strong knowledge of FastAPI or Django REST framework
- Experience with PostgreSQL and Redis in production
- Hands-on experience with Docker and Kubernetes
- Familiarity with AWS or GCP cloud services
- Experience with message brokers such as Kafka or RabbitMQ
- Good understanding of REST API design principles
- English level: Upper-Intermediate or higher

Nice to have:
- Experience with Celery for async task processing
- Knowledge of Terraform or other IaC tools
- Experience in fintech or high-load systems
"""


def run_single(resume_text=None, jd_text=None, resume_id="resume-001", jd_id="jd-001", position=None, exp_years=None):
    resume_text = resume_text or SAMPLE_RESUME
    jd_text  = jd_text or SAMPLE_JD
    orchestrator = SkillMiningOrchestrator()
    result = orchestrator.run(
        resume_text=resume_text,
        jd_text=jd_text,
        resume_id=resume_id,
        jd_id=jd_id,
        resume_position=position,
        experience_years=float(exp_years) if exp_years else None,
    )

    print("\nPARSED RESUME")
    r = result["parsed_resume"]
    print(f"Position: {r['position']}")
    print(f"Exp years: {r['experience_years']}")
    print(f"Raw skills: {r['raw_skills']}")
    print(f"Experience blocks: {len(r['experience'])}")

    print("\nPARSED JD")
    j = result["parsed_jd"]
    print(f"Title: {j['title']}")
    print(f"Exp years required: {j['experience_years']}")
    print(f"Required skills: {j['required_skills']}")
    print(f"Preferred skills: {j['preferred_skills']}")

    print("\nMINED SKILLS")
    rs = result["resume_skills"]
    js = result["jd_skills"]
    print(f"Resume ({len(rs['skills'])} skills):")
    for e in rs["skills"][:10]:
        print(f"  {e['skill']} -> {e['canonical']} [{e['category']} conf={e['confidence']}]")
    print(f"JD ({len(js['skills'])} skills):")
    for e in js["skills"][:10]:
        print(f"  {e['skill']} -> {e['canonical']} [{e['category']} conf={e['confidence']}]")

    print("\nMATCH SCORES")
    m = result["match"]
    print(f"Skill score: {m['skill_score']:.4f} (weight 0.40)")
    print(f"Experience: {m['experience_score']:.4f} (weight 0.30)")
    print(f"English level: {m['english_level_score']:.4f} (weight 0.20)")
    print(f"Title: {m['title_score']:.4f} (weight 0.10)")
    print(f"Final score: {m['final_score']:.4f}")
    print(f"Jaccard: {m['jaccard_score']:.4f}")


def run_rank(resume_paths: list, jd_path: str, jd_id: str = "jd-001"):
    jd_text = Path(jd_path).read_text(encoding="utf-8")

    candidates = []
    for rp in resume_paths:
        p = Path(rp)
        candidates.append((p.stem, p.read_text(encoding="utf-8")))

    orchestrator = SkillMiningOrchestrator()
    print(f"\nRanking {len(candidates)} resume(s) against: {jd_path}\n")

    ranked = orchestrator.rank_candidates(
        jd_text=jd_text,
        candidates=candidates,
        jd_id=jd_id,
    )

    print(f"{'Rank':>4}  {'Resume':>20}  {'Score':>6}  {'Matched Skills':<30}  Explanation")
    print("-" * 105)
    for r in ranked:
        matched     = ", ".join(r["matched_skills"][:4]) + ("..." if len(r["matched_skills"]) > 4 else "")
        explanation = r["explanation"][:70] + ("..." if len(r["explanation"]) > 70 else "")
        print(
            f"{r['rank']:>4}  "
            f"{r['resume_id'][:20]:>20}  "
            f"{r['final_score']:>6.3f}  "
            f"{matched:<30}  "
            f"{explanation}"
        )


def run_batch(n: int, keyword: str, counter: str):
    import pandas as pd

    resumes_path = ROOT / "data" / "processed" / "resumes_parsed.parquet"
    jds_path     = ROOT / "data" / "processed" / "jds_parsed.parquet"

    if not resumes_path.exists() or not jds_path.exists():
        logger.error("Parsed data not found. Run scripts/run_preprocessing.py first.")
        sys.exit(1)

    resumes_df = pd.read_parquet(resumes_path)
    jds_df     = pd.read_parquet(jds_path)

    target_jds = jds_df[
        (jds_df["primary_keyword"] == keyword) &
        (jds_df["required_skills"].apply(len) >= 3)
    ]
    if len(target_jds) == 0:
        logger.error(f"No JDs found for keyword '{keyword}'. Check --keyword value.")
        sys.exit(1)
    jd_row = target_jds.iloc[0]

    rel_r   = resumes_df[resumes_df["primary_keyword"] == keyword].head(n // 2)
    irrel_r = resumes_df[resumes_df["primary_keyword"] == counter].head(n // 2)

    if len(irrel_r) == 0:
        logger.error(f"No resumes found for counter keyword '{counter}'. Check --counter value.")
        sys.exit(1)

    sample     = pd.concat([rel_r, irrel_r]).reset_index(drop=True)
    candidates = [(str(row["id"]), row["raw_text"]) for _, row in sample.iterrows()]

    orchestrator = SkillMiningOrchestrator()

    print(f"\nJD: {jd_row['title']} [{keyword}]")
    print(f"Required skills: {list(jd_row['required_skills'])[:8]}")
    print(f"Candidates: {len(rel_r)} {keyword} (relevant) + {len(irrel_r)} {counter} (irrelevant)")
    print()

    ranked = orchestrator.rank_candidates(
        jd_text=jd_row["raw_text"],
        candidates=candidates,
        jd_id=str(jd_row["id"]),
    )

    print(f"{'Rank':>4}  {'Resume':>12}  {'Score':>6}  Explanation")
    print("-" * 80)
    for r in ranked:
        explanation = r["explanation"][:60] + ("..." if len(r["explanation"]) > 60 else "")
        print(f"{r['rank']:>4}  {r['resume_id'][:12]:>12}  {r['final_score']:>6.3f}  {explanation}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resume–JD matching pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo with built-in sample data
  python3 scripts/run_full_pipeline.py

  # Single resume vs JD
  python3 scripts/run_full_pipeline.py --resume resume.txt --jd job.txt

  # Single resume with metadata
  python3 scripts/run_full_pipeline.py --resume resume.txt --jd job.txt --position "Senior Backend Engineer" --exp-years 5

  # Rank multiple resumes against one JD
  python3 scripts/run_full_pipeline.py --rank alice.txt bob.txt carol.txt --jd job.txt

  # Batch ranking from parsed dataset
  python3 scripts/run_full_pipeline.py --batch --n 20
        """,
    )
    parser.add_argument("--resume",    type=str,   help="Path to a single resume text file (.txt)")
    parser.add_argument("--jd",        type=str,   help="Path to job description text file (.txt)")
    parser.add_argument("--resume-id", type=str,   default="resume-001", help="Resume identifier (single mode)")
    parser.add_argument("--jd-id",     type=str,   default="jd-001",     help="JD identifier")
    parser.add_argument("--position",  type=str,   help="Candidate's current/target position title")
    parser.add_argument("--exp-years", type=float, help="Candidate's years of experience")
    parser.add_argument("--rank",      type=str,   nargs="+", metavar="RESUME",
                        help="Rank multiple resume files against --jd  (e.g. --rank r1.txt r2.txt r3.txt)")
    parser.add_argument("--batch",     action="store_true", help="Batch ranking mode (uses parsed dataset)")
    parser.add_argument("--n",         type=int,   default=20,           help="Number of candidates in batch mode")
    parser.add_argument("--keyword",   type=str,   default="Python",     help="Target keyword for batch mode JD selection")
    parser.add_argument("--counter",   type=str,   default="JavaScript", help="Counter keyword for irrelevant resumes in batch mode")
    args = parser.parse_args()

    if args.batch:
        run_batch(args.n, args.keyword, args.counter)
    elif args.rank:
        if not args.jd:
            parser.error("--rank requires --jd to specify the job description file.")
        run_rank(resume_paths=args.rank, jd_path=args.jd, jd_id=args.jd_id)
    else:
        resume_text = jd_text = None
        if args.resume:
            resume_text = Path(args.resume).read_text(encoding="utf-8")
        if args.jd:
            jd_text = Path(args.jd).read_text(encoding="utf-8")
        if bool(args.resume) != bool(args.jd):
            parser.error("Provide both --resume and --jd, or neither (uses built-in sample).")
        run_single(
            resume_text=resume_text,
            jd_text=jd_text,
            resume_id=args.resume_id,
            jd_id=args.jd_id,
            position=args.position,
            exp_years=args.exp_years,
        )
