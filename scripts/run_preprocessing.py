"""
Script to run data preprocessing pipeline
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def main():
    """Run the preprocessing pipeline."""
    logger.info("Starting data preprocessing...")

    # TODO: Implement preprocessing steps
    # 1. Load raw resumes and job postings
    # 2. Clean and normalize text
    # 3. Extract metadata
    # 4. Save processed data

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
