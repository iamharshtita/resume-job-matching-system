"""
Environment setup script for the resume-job matching pipeline.
Handles: dependency check + raw data download

Usage:
    PYTHONPATH=src python3 scripts/setup_environment.py [--skip-download]

Options:
    --skip-download    Skip downloading raw data (if already exists)
"""
import sys
import time
import argparse
from pathlib import Path

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import RESUMES_RAW, JDS_RAW


def print_banner(text):
    """Print a formatted section banner."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def check_dependencies():
    """Verify required packages are installed."""
    print_banner("STEP 1/2: Checking Dependencies")

    missing = []
    packages = {
        'pandas': 'pandas',
        'tqdm': 'tqdm',
        'huggingface_hub': 'huggingface_hub',
        'pyarrow': 'pyarrow',
        'sentence_transformers': 'sentence-transformers',
        'faiss': 'faiss-cpu',
        'loguru': 'loguru',
        'pydantic': 'pydantic',
        'numpy': 'numpy',
        'scikit-learn': 'scikit-learn',
        'scipy': 'scipy',
    }

    for module, package in packages.items():
        try:
            __import__(module)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [MISSING] {package}")
            missing.append(package)

    if missing:
        print("\nMissing dependencies detected!")
        print(f"Install with: pip install {' '.join(missing)}")
        response = input("\nInstall now? [y/N]: ").strip().lower()
        if response == 'y':
            import subprocess
            print("\nInstalling packages...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
            print("Dependencies installed successfully")
        else:
            print("Cannot proceed without dependencies")
            sys.exit(1)
    else:
        print("\nAll dependencies are installed")


def download_data():
    """Download raw data from HuggingFace."""
    print_banner("STEP 2/2: Downloading Raw Data from HuggingFace")

    if RESUMES_RAW.exists() and JDS_RAW.exists():
        print(f"  Raw data already exists:")
        print(f"    - {RESUMES_RAW}")
        print(f"    - {JDS_RAW}")
        response = input("\n  Re-download? [y/N]: ").strip().lower()
        if response != 'y':
            print("  Skipping download")
            return

    print("\n  Downloading datasets (this may take 10-15 minutes)...")

    from huggingface_hub import hf_hub_download
    from config import HF_RESUMES_DATASET, HF_JDS_DATASET, HF_PARQUET_PATH
    import shutil

    # Download resumes
    print("\n  [1/2] Downloading resumes dataset...")
    start = time.time()
    try:
        path = hf_hub_download(
            repo_id=HF_RESUMES_DATASET,
            filename=HF_PARQUET_PATH,
            repo_type="dataset",
        )
        RESUMES_RAW.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, RESUMES_RAW)
        size_mb = RESUMES_RAW.stat().st_size / (1024**2)
        print(f"  Downloaded: {RESUMES_RAW}")
        print(f"  Size: {size_mb:.1f} MB | Time: {time.time()-start:.1f}s")
    except Exception as e:
        print(f"  Error downloading resumes: {e}")
        sys.exit(1)

    # Download JDs
    print("\n  [2/2] Downloading job descriptions dataset...")
    start = time.time()
    try:
        path = hf_hub_download(
            repo_id=HF_JDS_DATASET,
            filename=HF_PARQUET_PATH,
            repo_type="dataset",
        )
        JDS_RAW.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, JDS_RAW)
        size_mb = JDS_RAW.stat().st_size / (1024**2)
        print(f"  Downloaded: {JDS_RAW}")
        print(f"  Size: {size_mb:.1f} MB | Time: {time.time()-start:.1f}s")
    except Exception as e:
        print(f"  Error downloading JDs: {e}")
        sys.exit(1)

    print("\nRaw data download complete")


def print_summary():
    """Print final summary and next steps."""
    print_banner("SETUP COMPLETE")

    print("\n  Your environment is ready! Here's what was created:")
    print()

    if RESUMES_RAW.exists():
        size = RESUMES_RAW.stat().st_size / (1024**2)
        print(f"  [OK] Raw resumes   : {RESUMES_RAW}")
        print(f"                       ({size:.1f} MB)")

    if JDS_RAW.exists():
        size = JDS_RAW.stat().st_size / (1024**2)
        print(f"  [OK] Raw JDs       : {JDS_RAW}")
        print(f"                       ({size:.1f} MB)")

    print("\n  " + "-" * 66)
    print("  NEXT STEPS:")
    print("  " + "-" * 66)
    print()
    print("  1. Parse and filter resumes:")
    print("     PYTHONPATH=src python3 scripts/parse_resumes.py")
    print()
    print("  2. Parse and filter job descriptions:")
    print("     PYTHONPATH=src python3 scripts/parse_jds.py")
    print()
    print("  3. Test the pipeline:")
    print("     PYTHONPATH=src python3 scripts/run_full_pipeline.py")
    print()


def main():
    """Main execution flow."""
    parser = argparse.ArgumentParser(
        description="Setup environment for resume-job matching pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip downloading raw data (if already exists)')

    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  RESUME-JOB MATCHING PIPELINE - ENVIRONMENT SETUP")
    print("=" * 70)

    start_time = time.time()

    try:
        check_dependencies()

        if not args.skip_download:
            download_data()

        total_time = time.time() - start_time
        print_summary()
        print(f"\n  Total setup time: {total_time/60:.1f} minutes")
        print()

    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nSetup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
