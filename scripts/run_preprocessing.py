"""
Runs the full preprocessing pipeline:
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
