# Resume-Job Matching System

A multi-agent AI system for intelligent resume-job matching with explainable results.

## Overview

This project implements a multi-agent architecture using the Strands Agents SDK to match job candidates with positions. The system outperforms traditional ATS (Applicant Tracking System) keyword matching by providing:

- **Better Ranking Quality**: Uses semantic understanding beyond keywords
- **Contextual Skill Matching**: Identifies explicit and latent skills
- **Explainability**: Evidence-backed explanations for rankings
- **Fairness**: Consistent evaluation across demographic variations

## Architecture

The system consists of 5 main components:

1. **Resume Parser Agent** - Extracts structured data from raw resumes
2. **Skill Mining Agent** - Identifies skills using O*NET taxonomy and embeddings
3. **Job-Resume Matching Agent** - Computes multi-dimensional fit scores
4. **Ranking & Explanation Agent** - Ranks candidates and generates explanations
5. **Orchestrator** - Coordinates agent interactions

## Project Structure

```
resume-job-matching-system/
├── data/               # Dataset storage
├── src/                # Source code
│   ├── agents/        # Agent implementations
│   ├── preprocess/    # Data preprocessing
│   ├── features/      # Feature extraction
│   ├── retrieval/     # Embeddings and vector store
│   ├── scoring/       # Scoring functions
│   ├── baselines/     # Baseline systems
│   └── evaluation/    # Metrics and evaluation
├── notebooks/         # Jupyter notebooks for experiments
├── configs/           # Configuration files
├── tests/             # Unit tests
├── scripts/           # Executable scripts
└── outputs/           # Results and logs
```

## Getting Started

### Prerequisites

- Python 3.10+
- pip or poetry for dependency management

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/resume-job-matching-system.git
cd resume-job-matching-system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run setup (installs deps, downloads spaCy model, registers src/, downloads datasets)
python setup_env.py

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Quick Start

```bash
# Verify everything works
python scripts/run_full_pipeline.py

# Run preprocessing (parse resumes + JDs)
python scripts/run_preprocessing.py
```

## Usage

### Processing a Single Resume-Job Pair

```python
from src.agents.orchestrator import SkillMiningOrchestrator

orchestrator = SkillMiningOrchestrator()

result = orchestrator.process_candidate_job_pair(
    resume_text="...",
    job_text="..."
)

print(f"Match Score: {result['final_score']}")
print(f"Explanation: {result['explanation']}")
```

### Ranking Multiple Candidates

```python
ranked_results = orchestrator.rank_all_candidates(
    job_id="job_123",
    all_matches=match_results
)

for rank, candidate in enumerate(ranked_results[:10], 1):
    print(f"{rank}. {candidate['name']} - Score: {candidate['final_score']}")
```

## Evaluation

The system is evaluated against two baselines:

1. **TF-IDF Baseline** - Traditional keyword matching
2. **Single-Agent LLM** - Direct LLM scoring

### Metrics

- **Ranking Quality**: NDCG, Spearman correlation, Precision@K
- **Explanation Quality**: Human ratings (usefulness, clarity, faithfulness)
- **Fairness**: Score stability across demographic variations
- **Efficiency**: Runtime, API costs

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black src/

# Lint code
ruff check src/
```

## Team

- **Member 1**: Data & Preprocessing, Resume Parser Agent
- **Member 2**: Skill Mining Agent, O*NET Integration
- **Member 3**: Matching Agent, Scoring System, Baselines
- **Member 4**: Ranking & Explanation Agent, Evaluation

## License

This project is developed for academic purposes as part of CSE 572 - Data Mining course at Arizona State University.

## Acknowledgments

- O*NET Database for skill taxonomy
- Strands Agents SDK for agent orchestration
- sentence-transformers for embeddings
