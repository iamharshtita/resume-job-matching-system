"""
Runs ResumeParserAgent over a resumes dataset and saves output to
data/processed/resumes_parsed.parquet
"""
import pandas as pd
from tqdm import tqdm
from agents.resume_parser import ResumeParserAgent
from config import RESUMES_RAW, PROCESSED_DIR

PRIMARY_KEYWORD_COL = 'Primary Keyword'

def main():
    print(f"Loading resumes...")
    df = pd.read_parquet(RESUMES_RAW)

    # Define target keywords
    technical_roles = [
        '.NET', 'C++', 'DevOps', 'Flutter', 'Golang', 'Java',
        'JavaScript', 'Node.js', 'PHP', 'Python', 'Ruby', 'Scala',
        'SQL', 'iOS', 'Unity'
    ]
    data_analytics = ['Data Analyst', 'Data Engineer', 'Data Science', 'Business Analyst']
    qa_testing = ['QA', 'QA Automation']

    target_keywords = technical_roles + data_analytics + qa_testing

    # Filter to only target keywords
    print(f"Original dataset: {len(df):,} resumes")
    df_filtered = df[df[PRIMARY_KEYWORD_COL].isin(target_keywords)].copy()
    print(f"After keyword filter: {len(df_filtered):,} resumes")

    # Sample 50% of QA records
    qa_mask = df_filtered[PRIMARY_KEYWORD_COL] == 'QA'
    qa_records = df_filtered[qa_mask]
    non_qa_records = df_filtered[~qa_mask]

    # Take 50% sample of QA
    qa_sample = qa_records.sample(frac=0.5, random_state=42)
    print(f"  QA records: {len(qa_records):,} → sampled to {len(qa_sample):,} (50%)")

    # Combine back together
    sample = pd.concat([non_qa_records, qa_sample]).reset_index(drop=True)
    print(f"Final dataset for processing: {len(sample):,} resumes")
    print("  Breakdown by keyword:")
    for keyword in sorted(sample[PRIMARY_KEYWORD_COL].value_counts().index):
        count = (sample[PRIMARY_KEYWORD_COL] == keyword).sum()
        print(f"    {keyword:20s}: {count:>6,}")
    print()

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
            result["primary_keyword"] = row[PRIMARY_KEYWORD_COL]
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
