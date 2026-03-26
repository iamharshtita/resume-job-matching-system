import sys
import re
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import PROCESSED_DATA_DIR
from agents.skill_miner import SkillMiningAgent


def extract_explicit_skills(row):
    if "skills" in row and pd.notna(row["skills"]):
        raw = str(row["skills"])
        parts = re.split(r"[,;/|]", raw)
        return [p.strip() for p in parts if p.strip()]
    return []


def main():
    resumes_file = PROCESSED_DATA_DIR / "resumes_parsed.parquet"
    skills_file = PROCESSED_DATA_DIR / "skills" / "skills_master.csv"
    output_file = PROCESSED_DATA_DIR / "resume_skills.parquet"

    if not resumes_file.exists():
        logger.error(f"Missing file: {resumes_file}")
        return

    if not skills_file.exists():
        logger.error(f"Missing file: {skills_file}")
        return

    resumes_df = pd.read_parquet(resumes_file)
    agent = SkillMiningAgent(config={"skills_file": str(skills_file)})

    records = []

    for _, row in resumes_df.iterrows():
        input_data = {
            "text": row["cleaned_text"],
            "explicit_skills": extract_explicit_skills(row)
        }

        result = agent.process(input_data)

        for item in result.get("explicit_skills", []):
            records.append({
                "resume_id": row["resume_id"],
                "skill": item["skill"],
                "category": item["category"],
                "matched_alias": item["matched_alias"],
                "source": item["source"],
                "confidence": item["confidence"]
            })

        for item in result.get("latent_skills", []):
            records.append({
                "resume_id": row["resume_id"],
                "skill": item["skill"],
                "category": item["category"],
                "matched_alias": item["matched_alias"],
                "source": item["source"],
                "confidence": item["confidence"]
            })

    if not records:
        logger.warning("No skills extracted.")
        return

    skills_df = pd.DataFrame(records).drop_duplicates()
    skills_df.to_parquet(output_file, index=False)

    logger.info(f"Saved {len(skills_df)} extracted skills to {output_file}")


if __name__ == "__main__":
    main()