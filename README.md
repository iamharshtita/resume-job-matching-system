# Resume-Job Matching System

A multi-agent pipeline for intelligent resume–job description matching with explainable, skill-aware ranking.

## Architecture

Six agents work in sequence:

```
Raw Resume Text          Raw JD Text
       │                      │
       ▼                      ▼
 ResumeParserAgent       JDParserAgent
 (skills, experience,    (required/preferred skills,
  english level)          experience years, english level)
       │                      │
       ▼                      ▼
 SkillMiningAgent        SkillMiningAgent
 (canonical skills +     (canonical skills +
  sentence embeddings)    sentence embeddings)
       │                      │
       └──────────┬───────────┘
                  ▼
           MatchingAgent
     (skill score · experience score
      english level score · title score
      → weighted final_score)
                  │
                  ▼
      RankingExplanationAgent
      (rank + plain-English explanation)
```

**Score weights** (configurable in `src/config.py`):

| Component | Weight | Signal |
|-----------|--------|--------|
| Skill match | 0.40 | Semantic cosine similarity between skill embeddings |
| Experience | 0.30 | Resume years / JD required years |
| English level | 0.20 | Ordinal scale (no_english → fluent) |
| Title overlap | 0.10 | Jaccard on position tokens |

**Novel contribution — Skill-IDF weighting**:  
Rare skills in the JD corpus receive higher IDF weight than common ones, integrated into the MatchingAgent's semantic scoring.

**Evaluation results** (Python keyword, 5 JDs × 20 candidates):

| Method | NDCG@5 | Precision@5 | Recall@5 | MAP | Time(s) |
|--------|--------|-------------|----------|-----|---------|
| TF-IDF baseline | 0.687 | 0.600 | 0.600 | 0.548 | 0.01 |
| Skill-IDF baseline | 0.544 | 1.000 | 1.000 | 1.000 | 0.02 |
| **Multi-Agent+IDF** | **0.856** | **0.800** | **0.800** | **0.770** | 8.60 |

**Multi-Agent+IDF achieves 24.6% improvement** over TF-IDF baseline on NDCG@5.

---

## Project Structure

```
resume-job-matching-system/
├── data/
│   ├── raw/                        # Raw Djinni dataset (resumes + JDs)
│   └── processed/
│       ├── resumes_parsed.parquet  # Output of parse_resumes.py
│       ├── jds_parsed.parquet      # Output of parse_jds.py
│       └── samples/                # Keyword-filtered subsets
├── src/
│   ├── agents/
│   │   ├── base_agent.py
│   │   ├── resume_parser.py        # Parses raw CV text
│   │   ├── jd_parser.py            # Parses raw JD text
│   │   ├── skill_miner.py          # Maps skills → canonical + embeddings
│   │   ├── matcher.py              # Multi-dimensional scoring
│   │   ├── ranker.py               # Ranking + explanation generation
│   │   └── orchestrator.py         # End-to-end coordinator
│   ├── baselines/
│   │   └── tfidf_baseline.py       # TF-IDF baseline agent
│   ├── schemas/models.py           # Pydantic data models
│   └── config.py                   # Weights, constants, paths
├── scripts/
│   ├── parse_resumes.py            # Batch parse all resumes (filters to 21 keywords)
│   ├── parse_jds.py                # Batch parse all JDs (filters to 21 keywords)
│   ├── compute_idf_weights.py      # Pre-compute skill IDF weights (run once)
│   ├── run_full_pipeline.py        # Multi-agent pipeline (single pair / batch)
│   ├── evaluate_all.py             # Unified evaluation (all methods, same test set)
│   ├── visualize_results.py        # Generate performance charts & tables
│   ├── fairness_analysis.py        # Bias analysis (experience/category)
│   ├── skill_clustering.py         # t-SNE visualization of skill embeddings
│   └── ablation_study.py           # Component contribution analysis
├── outputs/
│   ├── results/                    # CSV files (comparison_results, detailed_scores)
│   └── visualizations/             # 12 PNG charts for academic presentation
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone and create virtual environment
git clone <repo-url>
cd resume-job-matching-system
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run setup (downloads sentence-transformer model on first use automatically)
```

> **Python 3.9+** required.  
> The skill miner downloads `all-MiniLM-L6-v2` from HuggingFace on first run (~80 MB, cached automatically).

---

## Running the Pipeline

### Step 1 — Parse the dataset (one-time)

Parse all raw resumes and JDs into structured parquet files (filters to 21 technical keywords):

```bash
PYTHONPATH=src python3 scripts/parse_resumes.py
PYTHONPATH=src python3 scripts/parse_jds.py
```

Outputs:
- `data/processed/resumes_parsed.parquet` (113,905 resumes)
- `data/processed/jds_parsed.parquet` (94,558 JDs)

**Step 2 — Pre-compute IDF weights (one-time)**

```bash
PYTHONPATH=src python3 scripts/compute_idf_weights.py
```

Generates `data/processed/skill_idf.json` with 47,612 skill weights.

### Step 3 — Run the full pipeline

**Single resume–JD pair** (uses built-in sample data):

```bash
PYTHONPATH=src python3 scripts/run_full_pipeline.py
```

**Single pair from your own files:**

```bash
PYTHONPATH=src python3 scripts/run_full_pipeline.py \
    --resume path/to/resume.txt \
    --jd path/to/job.txt
```

Optionally pass extra metadata:

```bash
PYTHONPATH=src python3 scripts/run_full_pipeline.py \
    --resume resume.txt \
    --jd job.txt \
    --position "Senior Backend Engineer" \
    --exp-years 5 \
    --resume-id "candidate-42" \
    --jd-id "job-123"
```

> Files must be plain `.txt` — paste the resume or JD text into a text file. Any filename works.

**Rank multiple resumes against one JD:**

```bash
PYTHONPATH=src python3 scripts/run_full_pipeline.py \
    --rank alice.txt bob.txt carol.txt \
    --jd job.txt
```

Pass as many resume files as you want after `--rank`. Results are printed as a ranked table sorted by final score, with matched skills and a plain-English explanation per candidate.

**Batch ranking** — rank N resumes against a JD from the parsed dataset:

```bash
PYTHONPATH=src python3 scripts/run_full_pipeline.py --batch --n 20
```

---

## Using the Orchestrator in Code

### Score a single resume against a JD

```python
import sys
sys.path.insert(0, "src")
from agents.orchestrator import SkillMiningOrchestrator

orchestrator = SkillMiningOrchestrator()

result = orchestrator.run(
    resume_text="Alex Chen\nSkills: Python, FastAPI, PostgreSQL, Docker...",
    jd_text="Senior Python Engineer\nRequirements:\n- 4+ years Python\n- FastAPI...",
    resume_id="candidate-001",
    jd_id="job-001",
    resume_position="Senior Backend Engineer",
    experience_years=5.0,
)

print(f"Final score    : {result['final_score']}")
print(f"Skill score    : {result['skill_score']}")
print(f"Experience     : {result['experience_score']}")
print(f"English level  : {result['english_level_score']}")
print(f"Title match    : {result['title_score']}")
```

### Rank multiple candidates against one JD

```python
from agents.orchestrator import SkillMiningOrchestrator

orchestrator = SkillMiningOrchestrator()

# candidates: list of (resume_id, raw_resume_text)
candidates = [
    ("candidate-001", "Alice ... Skills: React, TypeScript, Node.js ..."),
    ("candidate-002", "Bob   ... Skills: Angular, JavaScript, CSS ..."),
    ("candidate-003", "Carol ... Skills: Vue.js, Python, SQL ..."),
]

jd_text = """
Frontend Engineer
Requirements:
- React or Angular
- TypeScript
- 3+ years experience
"""

ranked = orchestrator.rank_candidates(
    jd_text=jd_text,
    candidates=candidates,
    jd_id="job-frontend-001",
)

for candidate in ranked:
    print(f"Rank {candidate['rank']:>2}  |  Score {candidate['final_score']:.3f}  "
          f"|  {candidate['resume_id']}")
    print(f"         Matched : {candidate['matched_skills']}")
    print(f"         Missing : {candidate['missing_skills']}")
    print(f"         {candidate['explanation']}")
    print()
```

`rank_candidates` returns a list of dicts sorted by `final_score` descending, each containing:

| Field | Type | Description |
|-------|------|-------------|
| `rank` | int | Position in ranking (1 = best) |
| `resume_id` | str | ID passed in |
| `final_score` | float | Weighted composite score (0–1) |
| `matched_skills` | list[str] | JD skills found in resume |
| `missing_skills` | list[str] | JD skills absent from resume |
| `explanation` | str | Plain-English recruiter summary |

---

## Evaluation & Visualization

### Unified evaluation (all methods)

```bash
PYTHONPATH=src python3 scripts/evaluate_all.py --keyword Python --n-jobs 5 --k 5
```

Runs TF-IDF, Skill-IDF, and Multi-Agent+IDF on the same test set. Outputs:
- `outputs/results/comparison_results.csv`
- `outputs/results/detailed_scores.csv`

### Generate visualizations

```bash
PYTHONPATH=src python3 scripts/visualize_results.py
```

Creates 4 core charts (metric comparison, NDCG@5, score distributions, tables) in `outputs/visualizations/`.

### Academic enhancements

**Fairness analysis** — score distribution by experience level and keyword category:
```bash
PYTHONPATH=src python3 scripts/fairness_analysis.py
```

**Skill clustering** — t-SNE visualization showing related skills cluster together:
```bash
PYTHONPATH=src python3 scripts/skill_clustering.py
```

**Ablation study** — measure contribution of each component (IDF, experience, title):
```bash
PYTHONPATH=src python3 scripts/ablation_study.py --n-jobs 3 --k 5
```

All outputs saved to `outputs/visualizations/` (12 charts total).

---

## Dataset

**Djinni dataset** (Ukrainian tech job market, HuggingFace):
- 210,250 developer resumes → filtered to 113,905 (21 technical keywords)
- 141,897 job descriptions → filtered to 94,558 (21 technical keywords)
- Fields: raw text, position, experience years, English level, primary keyword

**Keywords**: .NET, C++, DevOps, Flutter, Golang, Java, JavaScript, Node.js, PHP, Python, Ruby, Scala, SQL, iOS, Unity, Data Analyst, Data Engineer, Data Science, Business Analyst, QA (50% sampled), QA Automation.

Paths configured in `src/config.py`. No LLM API required — uses SentenceTransformers (all-MiniLM-L6-v2, 80MB).

---

## Key Features

✅ **No LLM API required** — Uses SentenceTransformers for embeddings (runs locally)  
✅ **Skill-IDF weighting** — Rare skills weighted higher than common ones  
✅ **Semantic matching** — Cosine similarity over skill embeddings (not exact match)  
✅ **FAISS indexing** — Fast similarity search over 25,963 canonical skills  
✅ **Explainable rankings** — Plain-English explanations per candidate  
✅ **Fairness validated** — No significant bias across experience levels (ANOVA p=0.806)  
✅ **Academic rigor** — 12 visualizations, ablation study, clustering analysis

---

## Team

*CSE 572 — Data Mining, Arizona State University*
