"""
Configuration management for the resume-job matching system
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent

# Data configs
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TAXONOMY_DIR = DATA_DIR / "taxonomy"

RESUMES_RAW = RAW_DIR / "resumes.parquet"
JDS_RAW = RAW_DIR / "jds.parquet"

# HuggingFace dataset config
HF_RESUMES_DATASET = os.getenv("HF_RESUMES_DATASET", "lang-uk/recruitment-dataset-candidate-profiles-english")
HF_JDS_DATASET = os.getenv("HF_JDS_DATASET", "lang-uk/recruitment-dataset-job-descriptions-english")
HF_RESUMES_PARQUET = os.getenv("HF_PARQUET", "data/train-00000-of-00001.parquet")
HF_JDS_PARQUET = os.getenv("HF_PARQUET", "data/train-00000-of-00001.parquet")

# Skill mapping file path
SKILLS_TAXONOMY_FILE = TAXONOMY_DIR / "skills_master.csv"

# Score parameters
WEIGHTS = {
    "skill": 0.4,
    "experience": 0.3,
    "education": 0.2,
    "title": 0.1,
}
SKILL_WEIGHTS = {
    "required": 0.7,
    "preferred": 0.3,
}
EXPLICIT_SKILL_CONFIDENCE = 1.0
LATENT_SKILL_CONFIDENCE = 0.75

# Output Paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
RESULTS_DIR = OUTPUT_DIR / "results"
EXPLANATIONS_DIR = OUTPUT_DIR / "explanations"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"

# Ensure directories exist
for directory in [LOG_DIR, RESULTS_DIR, EXPLANATIONS_DIR, VISUALIZATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

# Scoring Weights
SKILL_WEIGHT = float(os.getenv("SKILL_WEIGHT", 0.40))
EXPERIENCE_WEIGHT = float(os.getenv("EXPERIENCE_WEIGHT", 0.30))
EDUCATION_WEIGHT = float(os.getenv("EDUCATION_WEIGHT", 0.20))
TITLE_WEIGHT = float(os.getenv("TITLE_WEIGHT", 0.10))

# API Limits
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2000))
TIMEOUT = int(os.getenv("TIMEOUT", 30))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOG_DIR / "app.log"

# Education Levels (for scoring)
EDUCATION_LEVELS = {
    "high school": 1,
    "associate": 2,
    "bachelor": 3,
    "master": 4,
    "phd": 5,
    "doctorate": 5
}

# Risk Thresholds
JOB_HOPPING_THRESHOLD = 3  # jobs in 2 years
CRITICAL_SKILL_GAP_THRESHOLD = 0.5  # 50% of critical skills missing
