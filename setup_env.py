"""
One-time environment setup script.

Run this python setup_env.py
"""
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
    print("\nDownloading datasets...")
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "download_data.py")])

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
