"""
Runs ResumeParserAgent over a resumes dataset and saves output to
data/processed/resumes_parsed.parquet
"""
import pandas as pd
from tqdm import tqdm
from agents.resume_parser import ResumeParserAgent
from config import RESUMES_RAW, PROCESSED_DIR

def main():
    print(f"Loading resumes...")
    df = pd.read_parquet(RESUMES_RAW)
    sample = df.reset_index(drop=True)
    print(f"Processing {len(sample)} resumes...")

    agent = ResumeParserAgent()
    results = []

    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        try:
            result = agent.process({
                "raw_text": row["CV"],
                "id": row["id"],
                "position": row["Position"],
                "experience_years": row["Experience Years"],
                "english_level": row.get("English Level"),
            })
            result["primary_keyword"] = row["Primary Keyword"]
            results.append(result)
        except Exception as e:
            print(f"  Skipped {row['id']}: {e}")

    out = pd.DataFrame(results)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "resumes_parsed.parquet"
    out.to_parquet(out_path, index=False)

    print(f"\nSaved {len(out)} records to {out_path}")
    print(f"Columns: {list(out.columns)}")
    print(f"\nSkills coverage: {(out['raw_skills'].apply(len) > 0).sum()} / {len(out)} resumes have skills extracted")

if __name__ == "__main__":
    main()
