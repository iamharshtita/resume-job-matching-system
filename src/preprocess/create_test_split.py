"""
creates the 80/20 train/test split from parsed data.
run this once after run_preprocessing.py.
fine-tuning and evaluation both use this split.

usage:
    python3 src/preprocess/create_test_split.py
"""
import sys
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import PROCESSED_DIR

SPLIT_PATH = ROOT / "data" / "test" / "test_split.json"
TEST_RATIO  = 0.2
SEED        = 42


def main():
    print("loading parsed data...")
    resumes = pd.read_parquet(PROCESSED_DIR / "resumes_parsed.parquet")
    jds     = pd.read_parquet(PROCESSED_DIR / "jds_parsed.parquet")
    print(f"resumes: {len(resumes):,} | jds: {len(jds):,}")

    random.seed(SEED)

    r_test_ids, r_train_ids = {}, {}
    j_test_ids, j_train_ids = {}, {}

    for kw in resumes['primary_keyword'].unique():
        ids = resumes[resumes['primary_keyword'] == kw]['id'].astype(str).tolist()
        random.shuffle(ids)
        n_test = max(1, int(len(ids) * TEST_RATIO))
        r_test_ids[kw]  = ids[:n_test]
        r_train_ids[kw] = ids[n_test:]

    for kw in jds['primary_keyword'].unique():
        ids = jds[jds['primary_keyword'] == kw]['id'].astype(str).tolist()
        random.shuffle(ids)
        n_test = max(1, int(len(ids) * TEST_RATIO))
        j_test_ids[kw]  = ids[:n_test]
        j_train_ids[kw] = ids[n_test:]

    split = {
        "resume_test_ids":  r_test_ids,
        "resume_train_ids": r_train_ids,
        "jd_test_ids":      j_test_ids,
        "jd_train_ids":     j_train_ids,
    }

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_PATH, "w") as f:
        json.dump(split, f)

    total_r_test  = sum(len(v) for v in r_test_ids.values())
    total_j_test  = sum(len(v) for v in j_test_ids.values())
    total_r_train = sum(len(v) for v in r_train_ids.values())
    total_j_train = sum(len(v) for v in j_train_ids.values())

    print(f"train: {total_r_train:,} resumes, {total_j_train:,} jds")
    print(f"test:  {total_r_test:,} resumes, {total_j_test:,} jds")
    print(f"saved to {SPLIT_PATH}")


if __name__ == "__main__":
    main()