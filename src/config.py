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

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVALUATION_DATA_DIR = DATA_DIR / "evaluation"

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
