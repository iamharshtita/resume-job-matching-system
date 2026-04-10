"""
Runs the full preprocessing pipeline:
  1. Parse resumes  → data/processed/resumes_parsed.parquet
  2. Parse JDs      → data/processed/jds_parsed.parquet

Run individual scripts instead if you only need one:
  python scripts/parse_resumes.py
  python scripts/parse_jds.py
"""
from loguru import logger
import parse_resumes
import parse_jds

def main():
    logger.info("Starting preprocessing pipeline...")

    logger.info("Step 1/2: Parsing resumes")
    parse_resumes.main()

    logger.info("Step 2/2: Parsing job descriptions")
    parse_jds.main()

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
