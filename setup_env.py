"""
One-time environment setup script.

Run this once after cloning and activating virtual environment:
    python setup_env.py
"""
import os
import shutil
import subprocess
import sys
import site
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent

DIRS = [
    ROOT / "data" / "raw",
    ROOT / "data" / "processed",
    ROOT / "data" / "taxonomy",
    ROOT / "data" / "logs",
    ROOT / "outputs" / "logs",
    ROOT / "outputs" / "results",
    ROOT / "outputs" / "explanations",
    ROOT / "outputs" / "visualizations",
]

def create_dirs():
    print("Creating project directories...")
    for folder in DIRS:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"  {folder.relative_to(ROOT)}")

def install_deps():
    print("\nInstalling packages from requirements.txt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def download_spacy_model():
    print("\nDownloading spaCy model (en_core_web_sm)...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

def download_data():
    from huggingface_hub import hf_hub_download

    hf_parquet = os.getenv("HF_PARQUET", "data/train-00000-of-00001.parquet")
    datasets = [
        (os.getenv("HF_RESUMES_DATASET", "lang-uk/recruitment-dataset-candidate-profiles-english"),
         ROOT / "data" / "raw" / "resumes.parquet"),
        (os.getenv("HF_JDS_DATASET", "lang-uk/recruitment-dataset-job-descriptions-english"),
         ROOT / "data" / "raw" / "jds.parquet"),
    ]
    for repo_id, dest in datasets:
        print(f"\nDownloading {dest.name} from {repo_id}...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=hf_parquet,
            repo_type="dataset",
            local_dir="/tmp/hf_download",
        )
        shutil.copy(path, dest)
        print(f"  Saved to {dest.relative_to(ROOT)}")

def register_src():
    src_path = ROOT / "src"
    pth_file = Path(site.getsitepackages()[0]) / "resume_job_matching.pth"
    pth_file.write_text(str(src_path) + "\n")
    print(f"\nRegistered src/ in venv site-packages ({pth_file})")

if __name__ == "__main__":
    create_dirs()
    install_deps()
    download_spacy_model()
    register_src()
    download_data()
    print("\nSetup complete.")
