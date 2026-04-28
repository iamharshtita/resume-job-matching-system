import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# prevents FAISS segfault on Mac caused by OpenMP threading conflicts
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TAXONOMY_DIR = DATA_DIR / "taxonomy"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RESUMES_RAW = RAW_DIR / "resumes.parquet"
JDS_RAW = RAW_DIR / "jds.parquet"
SKILLS_TAXONOMY_FILE = TAXONOMY_DIR / "skills_master.csv"

LOG_DIR = OUTPUT_DIR / "logs"
RESULTS_DIR = OUTPUT_DIR / "results"
EXPLANATIONS_DIR = OUTPUT_DIR / "explanations"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"

for _d in [LOG_DIR, RESULTS_DIR, EXPLANATIONS_DIR, VISUALIZATIONS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# HuggingFace dataset
HF_RESUMES_DATASET = os.getenv(
    "HF_RESUMES_DATASET",
    "lang-uk/recruitment-dataset-candidate-profiles-english",
)
HF_JDS_DATASET = os.getenv(
    "HF_JDS_DATASET",
    "lang-uk/recruitment-dataset-job-descriptions-english",
)
HF_PARQUET_PATH = os.getenv("HF_PARQUET_PATH", "data/train-00000-of-00001.parquet")

# Models
# use fine-tuned model if available, otherwise fall back to base model
_FINETUNED_MODEL_PATH = DATA_DIR / "models" / "skill_embedding_model"
EMBEDDING_MODEL = (
    str(_FINETUNED_MODEL_PATH)
    if _FINETUNED_MODEL_PATH.exists()
    else os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
)
ENGLISH_LEVELS = {
    "no_english":   0,
    "pre":          1,
    "basic":        2,
    "intermediate": 3,
    "upper":        4,
    "fluent":       5,
}

# Defaults used when no learned weights file is present.
# Run scripts/train_matcher_weights.py to produce outputs/results/learned_weights.json,
# which will be picked up automatically on the next import.
_DEFAULT_WEIGHTS = {
    "skill": 0.4,
    "experience": 0.3,
    "education": 0.2,
    "title": 0.1,
}

_LEARNED_WEIGHTS_FILE = RESULTS_DIR / "learned_weights.json"

def _load_weights() -> dict:
    if _LEARNED_WEIGHTS_FILE.exists():
        import json
        with open(_LEARNED_WEIGHTS_FILE) as f:
            loaded = json.load(f)
        # Validate all four keys are present
        if all(k in loaded for k in ("skill", "experience", "education", "title")):
            return loaded
    return _DEFAULT_WEIGHTS

WEIGHTS = _load_weights()

# Used in the exact-match fallback when skill vectors are unavailable
SKILL_WEIGHTS = {
    "required": 0.7,
    "preferred": 0.3,
}

# Confidence assigned to skills depending on how they were found
EXPLICIT_SKILL_CONFIDENCE = 1.0
LATENT_SKILL_CONFIDENCE = 0.75

# Education levels
# Covers all degree types seen in the lang-uk recruitment dataset
EDUCATION_LEVELS = {
    "high school": 1,
    "secondary": 1,
    "certificate": 2,
    "diploma": 2,
    "associate": 2,
    "incomplete higher": 2,
    "bachelor": 3,
    "undergraduate": 3,
    "bsc": 3,
    "b.sc": 3,
    "b.s.": 3,
    "higher education": 3,
    "specialist": 4,
    "master": 5,
    "msc": 5,
    "m.sc": 5,
    "m.s.": 5,
    "mba": 5,
    "second higher": 5,
    "phd": 6,
    "ph.d": 6,
    "doctorate": 6,
    "doctor": 6,
}

# OpenAI used for LLM to establish ground truth
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOG_DIR / "app.log"
