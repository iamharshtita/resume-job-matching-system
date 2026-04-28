"""
Pre-compute IDF weights for all skills in the JD corpus.
Run once to generate data/processed/skill_idf.json

IDF(skill) = log((N + 1) / (df + 1)) + 1.0
where N = total JDs, df = JDs containing this skill

Usage:
    python3 src/preprocess/compute_idf_weights.py
"""
import sys
import json
import math
from pathlib import Path
from collections import Counter

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import PROCESSED_DIR

def main():
    print("Loading parsed JDs...")
    jds_path = PROCESSED_DIR / "jds_parsed.parquet"

    if not jds_path.exists():
        print(f"Error: {jds_path} not found. Run parse_jds.py first.")
        sys.exit(1)

    df = pd.read_parquet(jds_path)
    print(f"Loaded {len(df):,} JDs")

    # Count document frequency for each skill
    print("Computing skill document frequencies...")
    N = len(df)
    df_counts = Counter()

    for _, row in df.iterrows():
        required = row.get("required_skills", [])
        preferred = row.get("preferred_skills", [])
        all_skills = list(required) + list(preferred)

        # Count each unique skill once per JD
        for skill in set(str(s).lower().strip() for s in all_skills if s):
            df_counts[skill] += 1

    print(f"Found {len(df_counts):,} unique skills")

    # Compute IDF weights
    print("Computing IDF weights...")
    idf_weights = {}
    for skill, df_count in df_counts.items():
        idf_weights[skill] = math.log((N + 1) / (df_count + 1)) + 1.0

    # Save to JSON
    output_path = PROCESSED_DIR / "skill_idf.json"
    with open(output_path, 'w') as f:
        json.dump(idf_weights, f, indent=2)

    print(f"\nSaved IDF weights to: {output_path}")
    print(f"Total skills: {len(idf_weights):,}")

    idf_values = list(idf_weights.values())
    print(f"min={min(idf_values):.4f}  max={max(idf_values):.4f}  mean={sum(idf_values)/len(idf_values):.4f}")

    print("\ntop 10 most common skills:")
    sorted_skills = sorted(df_counts.items(), key=lambda x: -x[1])
    for skill, freq in sorted_skills[:10]:
        print(f"  {skill}  freq={freq:,}  idf={idf_weights[skill]:.4f}")

if __name__ == "__main__":
    main()
