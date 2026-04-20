"""
Runs JDParserAgent over job descriptions and saves output to
data/processed/jds_parsed.parquet
"""
import pandas as pd
from tqdm import tqdm
from agents.jd_parser import JDParserAgent
from config import JDS_RAW, PROCESSED_DIR

PRIMARY_KEYWORD_COL = 'Primary Keyword'

def main():
    print("Loading JDs...")
    df = pd.read_parquet(JDS_RAW)

    # Define target keywords (same as resume dataset)
    technical_roles = [
        '.NET', 'C++', 'DevOps', 'Flutter', 'Golang', 'Java',
        'JavaScript', 'Node.js', 'PHP', 'Python', 'Ruby', 'Scala',
        'SQL', 'iOS', 'Unity'
    ]
    data_analytics = ['Data Analyst', 'Data Engineer', 'Data Science', 'Business Analyst']
    qa_testing = ['QA', 'QA Automation']

    target_keywords = technical_roles + data_analytics + qa_testing

    # Filter to only target keywords
    print(f"Original dataset: {len(df):,} JDs")
    sample = df[df[PRIMARY_KEYWORD_COL].isin(target_keywords)].copy().reset_index(drop=True)
    print(f"After keyword filter: {len(sample):,} JDs")
    print("  Breakdown by keyword:")
    for keyword in sorted(sample[PRIMARY_KEYWORD_COL].value_counts().index):
        count = (sample[PRIMARY_KEYWORD_COL] == keyword).sum()
        print(f"    {keyword:20s}: {count:>6,}")
    print()

    print(f"Processing {len(sample)} JDs...")
    agent = JDParserAgent()
    results = []

    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        try:
            result = agent.process({
                "raw_text": row["Long Description"],
                "id": row["id"],
                "position": row["Position"],
                "company": row["Company Name"],
                "exp_years_raw": row["Exp Years"],
                "primary_keyword": row[PRIMARY_KEYWORD_COL],
                "english_level": row.get("English Level"),
            })
            results.append(result)
        except Exception as e:
            print(f"  Skipped {row['id']}: {e}")

    out = pd.DataFrame(results)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "jds_parsed.parquet"
    out.to_parquet(out_path, index=False)

    print(f"\nSaved {len(out)} records → {out_path}")
    print(f"Columns: {list(out.columns)}")
    req_coverage = out['required_skills'].apply(len).gt(0).sum()
    pref_coverage = out['preferred_skills'].apply(len).gt(0).sum()
    print(f"\nRequired skills coverage : {req_coverage} / {len(out)} JDs")
    print(f"Preferred skills coverage: {pref_coverage} / {len(out)} JDs")
    print(f"Avg required skills      : {out['required_skills'].apply(len).mean():.1f}")

if __name__ == "__main__":
    main()
