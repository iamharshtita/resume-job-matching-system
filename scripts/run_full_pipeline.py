"""
Script to run the full resume-job matching pipeline
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
from agents.orchestrator import SkillMiningOrchestrator


def main():
    """Run the full pipeline on test data."""
    logger.info("Starting full pipeline...")

    # Initialize orchestrator
    orchestrator = SkillMiningOrchestrator()

    # Example usage
    resume_text = """
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

    job_text = """
    Machine Learning Engineer

    We are seeking an experienced ML Engineer to join our AI team.

    Requirements:
    - 3+ years experience with Python and ML frameworks
    - Experience with TensorFlow or PyTorch
    - Strong software engineering background
    - BS in Computer Science or related field

    Responsibilities:
    - Build and deploy ML models
    - Collaborate with data scientists
    - Optimize model performance
    """

    # Process the pair
    result = orchestrator.process_candidate_job_pair(resume_text, job_text)

    # Print results
    print("\n" + "="*50)
    print("MATCHING RESULTS")
    print("="*50)
    print(f"Final Score: {result['final_score']}")
    print(f"\nExplanation:\n{result['explanation']}")
    print("="*50)

    logger.info("Pipeline execution complete!")


if __name__ == "__main__":
    main()
