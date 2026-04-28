"""
sets up the project from scratch.
run this once before anything else.

    python3 setup_env.py
    python3 setup_env.py --skip-download
"""
import sys
import platform
import argparse
import subprocess
from pathlib import Path

ROOT     = ROOT = Path(__file__).parent
VENV_DIR = ROOT / "venv"
IS_MAC   = platform.system() == "Darwin"
IS_WIN   = platform.system() == "Windows"

VENV_PY = (
    VENV_DIR / "Scripts" / "python.exe" if IS_WIN
    else VENV_DIR / "bin" / "python3"
)
VENV_ACT = (
    VENV_DIR / "Scripts" / "activate" if IS_WIN
    else VENV_DIR / "bin" / "activate"
)

DIRS = [
    ROOT / "data" / "raw",
    ROOT / "data" / "processed",
    ROOT / "data" / "taxonomy",
    ROOT / "data" / "models",
    ROOT / "outputs" / "results",
    ROOT / "outputs" / "logs",
]

# python 3.11 and 3.12 have pre-built wheels for all packages
# 3.13+ does not yet — so we look for 3.11/3.12 first
PREFERRED_VERSIONS = ["3.11", "3.12"]


def find_compatible_python():
    """look for python 3.11 or 3.12 on the system"""
    candidates = []

    for ver in PREFERRED_VERSIONS:
        candidates += [
            f"python{ver}",
            f"/opt/homebrew/bin/python{ver}",
            f"/usr/local/bin/python{ver}",
            f"/usr/bin/python{ver}",
            f"C:\\Python{ver.replace('.', '')}\\python.exe",
        ]

    for py in candidates:
        try:
            result = subprocess.run(
                [py, "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                version_str = result.stdout.strip() or result.stderr.strip()
                return py, version_str
        except FileNotFoundError:
            continue

    return None, None


def install_python_mac(version="3.11"):
    """try to install python via homebrew on mac"""
    # check if brew exists
    try:
        subprocess.run(["brew", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

    print(f"installing python {version} via homebrew...")
    result = subprocess.run(["brew", "install", f"python@{version}"])
    return result.returncode == 0


def get_current_python_version():
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def create_venv():
    if VENV_DIR.exists():
        print("venv already exists, skipping")
        return

    py_exec, py_version = find_compatible_python()

    if py_exec is None:
        current = get_current_python_version()
        print(f"python 3.11 or 3.12 not found (you have {current})")
        print("these versions are needed because newer pythons lack pre-built")
        print("wheels for torch, pandas, and faiss.")
        print()

        if IS_MAC:
            print("installing python 3.11 via homebrew...")
            if install_python_mac("3.11"):
                py_exec, py_version = find_compatible_python()
            else:
                print("homebrew install failed. install manually:")
                print("  brew install python@3.11")
                sys.exit(1)
        else:
            print("install python 3.11 from https://www.python.org/downloads/")
            print("then re-run this script")
            sys.exit(1)

    if py_exec is None:
        print("could not find python 3.11 or 3.12 after installation attempt")
        sys.exit(1)

    print(f"using {py_version}")
    print("creating venv...")
    subprocess.check_call([py_exec, "-m", "venv", str(VENV_DIR)])
    print("venv created")


def install_deps():
    req = ROOT / "requirements.txt"
    if not req.exists():
        print("requirements.txt not found")
        sys.exit(1)

    print("installing packages (this takes 3-5 minutes)...")
    subprocess.check_call(
        [str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip", "-q"]
    )
    subprocess.check_call(
        [str(VENV_PY), "-m", "pip", "install", "-r", str(req), "-q"]
    )
    print("packages installed")


def create_dirs():
    print("creating project directories...")
    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)
    print("done")


def download_data():
    print("downloading datasets from HuggingFace (10-15 minutes)...")

    script = """
import sys, shutil, time
sys.path.insert(0, '{src}')
from pathlib import Path
from huggingface_hub import hf_hub_download
from config import RESUMES_RAW, JDS_RAW, HF_RESUMES_DATASET, HF_JDS_DATASET, HF_PARQUET_PATH

for label, repo, dest in [
    ('resumes', HF_RESUMES_DATASET, RESUMES_RAW),
    ('job descriptions', HF_JDS_DATASET, JDS_RAW),
]:
    dest = Path(dest)
    if dest.exists():
        print(f'  {{label}} already downloaded, skipping')
        continue
    print(f'  downloading {{label}}...')
    t0 = time.time()
    path = hf_hub_download(repo_id=repo, filename=HF_PARQUET_PATH, repo_type='dataset')
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, dest)
    print(f'  done ({{dest.stat().st_size / 1024**2:.0f}} MB, {{time.time()-t0:.0f}}s)')
print('download complete')
""".format(src=str(ROOT / "src"))

    result = subprocess.run([str(VENV_PY), "-c", script])
    if result.returncode != 0:
        print("download failed. check your internet connection and try again.")
        sys.exit(1)


def next_steps():
    activate_cmd = (
        f"source {VENV_ACT}" if not IS_WIN
        else str(VENV_ACT)
    )
    print()
    print("all done. activate the venv and run the pipeline in order:")
    print()
    print(f"  {activate_cmd}")
    print()
    print("  python3 scripts/run_preprocessing.py")
    print("  python3 scripts/run_full_pipeline.py")
    print()
    print("see README.md for details on each step.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-download", action="store_true",
        help="skip raw data download if already in data/raw/"
    )
    args = parser.parse_args()

    create_venv()
    install_deps()
    create_dirs()

    if args.skip_download:
        print("skipping data download")
    else:
        download_data()

    next_steps()


if __name__ == "__main__":
    main()