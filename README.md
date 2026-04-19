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

**Novel contribution — Skill-IDF weighting** (`scripts/test_skill_idf.py`):  
Rare skills in the JD corpus receive higher IDF weight than ubiquitous ones.  
Evaluated against a TF-IDF baseline and GPT-4 recruiter scores:

| Metric | TF-IDF | Skill-IDF | Combined |
|--------|--------|-----------|----------|
| NDCG@5 | 0.732 | **0.894** | 0.882 |
| Precision@5 | 0.724 | **1.000** | 0.916 |
| Spearman vs GPT | 0.175 | 0.440 | **0.615** |

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
│   ├── parse_resumes.py            # Batch parse all resumes
│   ├── parse_jds.py                # Batch parse all JDs
│   ├── sample_dataset.py           # Filter dataset by keyword
│   ├── run_full_pipeline.py        # Multi-agent pipeline (single pair / batch)
│   ├── run_tfidf_baseline.py       # TF-IDF baseline scorer
│   └── run_skill_idf.py            # Skill-IDF baseline scorer
├── tests/
│   └── test_agents/
├── outputs/results/                # Evaluation CSVs
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

Parse all raw resumes and JDs into structured parquet files:

```bash
PYTHONPATH=src python3 scripts/parse_resumes.py
PYTHONPATH=src python3 scripts/parse_jds.py
```

Outputs:
- `data/processed/resumes_parsed.parquet`
- `data/processed/jds_parsed.parquet`

### Step 2 — (Optional) Sample a keyword-filtered subset

```bash
PYTHONPATH=src python3 scripts/sample_dataset.py \
    --resume-keywords javascript \
    --jd-keywords javascript \
    --resume-limit 100 \
    --jd-limit 100 \
    --output-tag js
```

Outputs to `data/processed/samples/`.

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

## Evaluation Scripts

### TF-IDF baseline comparison

```bash
PYTHONPATH=src python3 scripts/run_tfidf_baseline.py --auto --k 5
```

`--auto` picks the two most common keywords from the parsed dataset automatically.

### Skill-IDF evaluation

Requires the JS sample to exist (`scripts/sample_dataset.py --output-tag js` first):

```bash
PYTHONPATH=src python3 scripts/run_skill_idf.py --k 5
```

Prints NDCG@5, Precision@5, and Spearman correlation for TF-IDF vs Skill-IDF vs Combined.

---

## Dataset

The system was built and evaluated on the **Djinni dataset** (Ukrainian tech job market):
- 210,250 developer resumes
- 141,897 job descriptions
- Fields: raw CV text, position, experience years, English level, primary keyword

Raw data goes in `data/raw/`. Paths are configured in `src/config.py`.

---

## Team

- **Member 1**: Data & Preprocessing, Resume Parser Agent
- **Member 2**: Skill Mining Agent, O*NET Integration
- **Member 3**: Matching Agent, Scoring System, Baselines
- **Member 4**: Ranking & Explanation Agent, Evaluation

*CSE 572 — Data Mining, Arizona State University*
