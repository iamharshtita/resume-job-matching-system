"""
Runs the full pipeline on a sample resume + JD pair.
"""
import json
from loguru import logger
from agents.orchestrator import SkillMiningOrchestrator

SAMPLE_RESUME = """
John Doe
john.doe@email.com | 555-123-4567

Software Engineer with 5 years of experience in Python, machine learning,
and web development. Led team of 3 developers on ML project.

Experience:
- Senior Software Engineer at Tech Corp (2020-2025)
- Junior Developer at StartUp Inc (2018-2020)

Education:
- BS Computer Science, State University (2018)

Skills: Python, TensorFlow, React, AWS, SQL
"""

SAMPLE_JD = """
Machine Learning Engineer

We are seeking an experienced ML Engineer to join our AI team.

Requirements:
- 3+ years experience with Python and ML frameworks
- Experience with TensorFlow or PyTorch
- Strong software engineering background
- BS in Computer Science or related field

Nice-to-have:
- Docker and Kubernetes experience
- Familiarity with AWS or GCP
"""

def main():
    logger.info("Starting full pipeline...")
    orchestrator = SkillMiningOrchestrator()
    result = orchestrator.run(SAMPLE_RESUME, SAMPLE_JD)

    print("\n" + "=" * 50)
    print("PARSED RESUME")
    print("=" * 50)
    print(json.dumps({k: v for k, v in result["parsed_resume"].items() if k != "raw_text"}, indent=2))

    print("\n" + "=" * 50)
    print("PARSED JD")
    print("=" * 50)
    print(json.dumps({k: v for k, v in result["parsed_jd"].items() if k != "raw_text"}, indent=2))

    # TODO: print resume_skills once ResumeSkillMinerAgent is implemented
    # TODO: print jd_skills once JDSkillMinerAgent is implemented
    # TODO: print match_scores once MatcherAgent is implemented
    # TODO: print ranked results once InsightAgent is implemented

    print("\n" + "=" * 50)
    print("MATCHING RESULTS")
    print("=" * 50)
    print(f"Final Score: {result['final_score']}")
    print(f"Explanation: {result['explanation']}")
    print("=" * 50)

    logger.info("Pipeline complete.")

if __name__ == "__main__":
    main()
