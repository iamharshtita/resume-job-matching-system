"""
Runs JDParserAgent over job descriptions and saves output to
data/processed/jds_parsed.parquet
"""
import pandas as pd
from tqdm import tqdm
from agents.jd_parser import JDParserAgent
from config import JDS_RAW, PROCESSED_DIR

SAMPLE_SIZE = 2000
def main():
    print("Loading JDs...")
    df = pd.read_parquet(JDS_RAW)
    sample = df.head(SAMPLE_SIZE).reset_index(drop=True)
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
                "primary_keyword": row["Primary Keyword"],
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
