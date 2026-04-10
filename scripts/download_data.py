"""
Download the datasets as parquet files directly from HuggingFace into data/raw/.
Run: PYTHONPATH=. .venv/bin/python utility_scripts/download_data.py
"""
from huggingface_hub import hf_hub_download
from config import (
    HF_RESUMES_DATASET, HF_JDS_DATASET,
    HF_RESUMES_PARQUET, HF_JDS_PARQUET,
    RESUMES_RAW, JDS_RAW,
)
import shutil

def main():
    print("Downloading resumes.parquet...")
    path = hf_hub_download(
        repo_id=HF_RESUMES_DATASET,
        filename=HF_RESUMES_PARQUET,
        repo_type="dataset",
        local_dir="/tmp/hf_download",
    )
    shutil.copy(path, RESUMES_RAW)
    print(f"Saved to {RESUMES_RAW}")

    print("Downloading jds.parquet...")
    path = hf_hub_download(
        repo_id=HF_JDS_DATASET,
        filename=HF_JDS_PARQUET,
        repo_type="dataset",
        local_dir="/tmp/hf_download",
    )
    shutil.copy(path, JDS_RAW)
    print(f"Saved to {JDS_RAW}")

if __name__ == "__main__":
    main()
