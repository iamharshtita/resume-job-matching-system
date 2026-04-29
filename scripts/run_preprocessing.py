"""
one-time data preparation pipeline.
run this once after setup_env.py to prepare everything the pipeline needs.

steps:
    1. parse resumes and JDs from raw data
    2. rebuild skill taxonomy from filtered dataset
    3. compute IDF weights across JD corpus
    4. fine-tune embedding model (optional, ~30 mins)

usage:
    python3 scripts/run_preprocessing.py
    python3 scripts/run_preprocessing.py --skip-finetune
"""
import sys
import argparse
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from preprocess import parse_resumes, parse_jds
from preprocess import rebuild_taxonomy, compute_idf_weights, create_test_split


def save_summary(out_path: Path):
    r = pd.read_parquet(ROOT / "data/processed/resumes_parsed.parquet")
    j = pd.read_parquet(ROOT / "data/processed/jds_parsed.parquet")

    lines = []
    lines.append("PREPROCESSING SUMMARY")
    lines.append("=" * 40)
    lines.append(f"\nRESUMES ({len(r):,} total)")
    lines.append(f"columns: {list(r.columns)}")
    lines.append(f"skills coverage: {(r['raw_skills'].apply(len) > 0).sum():,} / {len(r):,}")
    lines.append(f"avg skills: {r['raw_skills'].apply(len).mean():.2f}")
    lines.append(f"english coverage: {r['english_level'].notna().sum():,} / {len(r):,}")
    lines.append("\nbreakdown by keyword:")
    for kw, cnt in r["primary_keyword"].value_counts().items():
        lines.append(f"  {kw:<25} {cnt:>7,}")

    lines.append(f"\nJOB DESCRIPTIONS ({len(j):,} total)")
    lines.append(f"columns: {list(j.columns)}")
    lines.append(f"required skills coverage: {(j['required_skills'].apply(len) > 0).sum():,} / {len(j):,}")
    lines.append(f"avg required skills: {j['required_skills'].apply(len).mean():.2f}")
    lines.append("\nbreakdown by keyword:")
    for kw, cnt in j["primary_keyword"].value_counts().items():
        lines.append(f"  {kw:<25} {cnt:>7,}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    logger.info(f"summary saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-finetune", action="store_true",
                        help="skip fine-tuning the embedding model")
    parser.add_argument("--finetune-samples", type=int, default=2000,
                        help="samples per keyword for fine-tuning (default 2000)")
    parser.add_argument("--finetune-epochs", type=int, default=2,
                        help="epochs for fine-tuning (default 2)")
    args = parser.parse_args()

    # step 1 - parse
    logger.info("step 1/4: parsing resumes")
    parse_resumes.main()

    logger.info("step 2/4: parsing job descriptions")
    parse_jds.main()

    save_summary(ROOT / "outputs/results/preprocessing_summary.txt")

    # step 2 - taxonomy
    logger.info("step 3/5: rebuilding skill taxonomy")
    rebuild_taxonomy.main()

    # step 3 - idf weights
    logger.info("step 4/5: computing IDF weights")
    compute_idf_weights.main()

    # step 4 - train/test split (always runs, independent of fine-tuning)
    logger.info("step 5/5: creating train/test split")
    create_test_split.main()

    # step 4 - fine-tune (optional)
    if args.skip_finetune:
        logger.info("skipping fine-tuning (--skip-finetune)")
    else:
        logger.info(f"step 5/5: fine-tuning embedding model (samples={args.finetune_samples} epochs={args.finetune_epochs})")
        logger.info("this takes ~30 mins on CPU, skip with --skip-finetune if not needed")
        from preprocess import finetune_embeddings
        sys.argv = [
            "finetune_embeddings.py",
            "--samples", str(args.finetune_samples),
            "--epochs",  str(args.finetune_epochs),
        ]
        finetune_embeddings.main()

    logger.info("all done. run python3 scripts/run_full_pipeline.py to test the pipeline.")


if __name__ == "__main__":
    main()