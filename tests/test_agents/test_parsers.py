"""
Smoke tests for ResumeParserAgent and JDParserAgent.
Run from project root: pytest tests/test_agents/test_parsers.py -v
"""
import pytest
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
SAMPLE_N = 5  # number of real rows to test
from agents.resume_parser import ResumeParserAgent
from agents.jd_parser import JDParserAgent

SAMPLE_RESUME = """
John Doe
john.doe@email.com | +1-555-0100

Software Engineer at Google 2020
Senior Developer at Meta 2018

Skills: Python, FastAPI, PostgreSQL, Docker, Kubernetes

Bachelor's degree in Computer Science, University of Toronto 2016

- Languages; Python, JavaScript, TypeScript
- Frameworks: FastAPI, React
"""

SAMPLE_JD = """
Software Engineer

Requirements:
- 3+ years of experience with Python
- Experience with FastAPI or Django
- Knowledge of PostgreSQL
- Docker and Kubernetes

Nice to have:
- Experience with React
- Familiarity with Redis
"""


@pytest.fixture(scope="module")
def resume_agent():
    return ResumeParserAgent()


@pytest.fixture(scope="module")
def jd_agent():
    return JDParserAgent()


class TestResumeParser:
    def test_returns_dict(self, resume_agent):
        result = resume_agent.process({"raw_text": SAMPLE_RESUME})
        assert isinstance(result, dict)

    def test_extracts_skills(self, resume_agent):
        result = resume_agent.process({"raw_text": SAMPLE_RESUME})
        assert len(result["raw_skills"]) > 0, "No skills extracted"

    def test_extracts_education(self, resume_agent):
        result = resume_agent.process({"raw_text": SAMPLE_RESUME})
        assert len(result["education"]) > 0, "No education blocks extracted"

    def test_extracts_experience(self, resume_agent):
        result = resume_agent.process({"raw_text": SAMPLE_RESUME})
        assert len(result["experience"]) > 0, "No experience blocks extracted"

    def test_missing_raw_text_raises(self, resume_agent):
        with pytest.raises(Exception):
            resume_agent.process({})


class TestJDParser:
    def test_returns_dict(self, jd_agent):
        result = jd_agent.process({"raw_text": SAMPLE_JD})
        assert isinstance(result, dict)

    def test_extracts_required_skills(self, jd_agent):
        result = jd_agent.process({"raw_text": SAMPLE_JD})
        assert len(result["required_skills"]) > 0, "No required skills extracted"

    def test_extracts_preferred_skills(self, jd_agent):
        result = jd_agent.process({"raw_text": SAMPLE_JD})
        assert len(result["preferred_skills"]) > 0, "No preferred skills extracted"

    def test_no_overlap_between_skill_lists(self, jd_agent):
        result = jd_agent.process({"raw_text": SAMPLE_JD})
        overlap = set(result["required_skills"]) & set(result["preferred_skills"])
        assert len(overlap) == 0, f"Skills in both lists: {overlap}"

    def test_exp_years_parsed(self, jd_agent):
        result = jd_agent.process({"raw_text": SAMPLE_JD, "exp_years_raw": "3y"})
        assert result["experience_years"] == 3

    def test_missing_raw_text_raises(self, jd_agent):
        with pytest.raises(Exception):
            jd_agent.process({})


# --- Real data tests ---

@pytest.fixture(scope="module")
def real_resumes():
    path = DATA_DIR / "resumes.parquet"
    if not path.exists():
        pytest.skip("resumes.parquet not found — run setup_env.py first")
    return pd.read_parquet(path).head(SAMPLE_N)


@pytest.fixture(scope="module")
def real_jds():
    path = DATA_DIR / "jds.parquet"
    if not path.exists():
        pytest.skip("jds.parquet not found — run setup_env.py first")
    return pd.read_parquet(path).head(SAMPLE_N)


class TestResumeParserRealData:
    def test_all_rows_return_dict(self, resume_agent, real_resumes):
        for _, row in real_resumes.iterrows():
            result = resume_agent.process({
                "raw_text": row["CV"],
                "id": str(row["id"]),
                "position": row["Position"],
                "experience_years": row["Experience Years"],
            })
            assert isinstance(result, dict), f"Expected dict for id={row['id']}"

    def test_skills_extracted_in_most_rows(self, resume_agent, real_resumes):
        extracted = 0
        for _, row in real_resumes.iterrows():
            result = resume_agent.process({"raw_text": row["CV"]})
            if len(result["raw_skills"]) > 0:
                extracted += 1
        assert extracted >= SAMPLE_N // 2, f"Skills extracted in only {extracted}/{SAMPLE_N} resumes"

    def test_no_crashes_on_real_text(self, resume_agent, real_resumes):
        for _, row in real_resumes.iterrows():
            try:
                resume_agent.process({"raw_text": row["CV"]})
            except Exception as e:
                pytest.fail(f"Crashed on id={row['id']}: {e}")


class TestJDParserRealData:
    def test_all_rows_return_dict(self, jd_agent, real_jds):
        for _, row in real_jds.iterrows():
            result = jd_agent.process({
                "raw_text": row["Long Description"],
                "id": str(row["id"]),
                "position": row["Position"],
                "company": row["Company Name"],
                "exp_years_raw": row["Exp Years"],
            })
            assert isinstance(result, dict), f"Expected dict for id={row['id']}"

    def test_required_skills_extracted_in_most_rows(self, jd_agent, real_jds):
        extracted = 0
        for _, row in real_jds.iterrows():
            result = jd_agent.process({"raw_text": row["Long Description"]})
            if len(result["required_skills"]) > 0:
                extracted += 1
        assert extracted >= SAMPLE_N // 2, f"Required skills in only {extracted}/{SAMPLE_N} JDs"

    def test_no_crashes_on_real_text(self, jd_agent, real_jds):
        for _, row in real_jds.iterrows():
            try:
                jd_agent.process({"raw_text": row["Long Description"]})
            except Exception as e:
                pytest.fail(f"Crashed on id={row['id']}: {e}")
