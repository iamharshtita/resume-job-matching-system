# Resume-Job Matching System

Multi-agent pipeline for matching resumes to job descriptions. Uses semantic skill mining with FAISS + sentence-transformers, IDF-weighted scoring, and explainable ranking.

Built for CSE 572 (Data Mining) at Arizona State University.

---

## What it does

Most resume screening tools just do keyword matching, which misses candidates who have the right skills but describe them differently. This system uses a chain of specialized agents that parse resumes and JDs, extract and normalize skills semantically, score candidates on multiple dimensions, and produce ranked results with explanations for each match.

---

## Architecture

```
Raw Resume Text              Raw JD Text
       |                          |
       v                          v
 ResumeParserAgent          JDParserAgent
 (skills, experience,       (required/preferred skills,
  english level)             experience years, english level)
       |                          |
       v                          v
 SkillMiningAgent           SkillMiningAgent
 (maps raw skills to        (maps raw skills to
  canonical names via        canonical names via
  FAISS + embeddings)        FAISS + embeddings)
       |                          |
       +------------+-------------+
                    |
                    v
             MatchingAgent
       skill score (0.40)
       experience score (0.30)
       english level score (0.20)
       title overlap score (0.10)
                    |
                    v
        RankingExplanationAgent
        ranked list + explanation per candidate
```

Weights are configurable in `src/config.py`.

---

## Project structure

```
resume-job-matching-system/
├── data/
│   ├── raw/                        # raw Djinni dataset (parquet)
│   ├── processed/                  # parsed parquets + skill_idf.json
│   ├── test/                       # test split + eval_pairs.parquet (5000 pairs)
│   ├── taxonomy/                   # skills_master.csv
│   └── models/                     # fine-tuned embedding model
├── src/
│   ├── agents/                     # pipeline agents
│   │   ├── resume_parser.py
│   │   ├── jd_parser.py
│   │   ├── skill_miner.py
│   │   ├── matcher.py
│   │   ├── ranker.py
│   │   └── orchestrator.py
│   ├── baselines/
│   │   └── tfidf_baseline.py
│   ├── preprocess/
│   │   ├── parse_resumes.py
│   │   ├── parse_jds.py
│   │   ├── rebuild_taxonomy.py
│   │   ├── compute_idf_weights.py
│   │   ├── create_test_split.py
│   │   └── finetune_embeddings.py
│   ├── evaluation/
│   │   ├── evaluate_all.py         # main 3-way comparison
│   │   ├── evaluate_skill_miner.py
│   │   ├── ablation_study.py
│   │   ├── fairness_analysis.py
│   │   ├── stress_test.py          # domain discrimination analysis
│   │   ├── statistical_tests.py    # significance + confidence intervals
│   │   ├── llm_judge.py            # LLM-as-Judge via AWS Bedrock
│   │   ├── visualize_results.py
│   │   └── skill_clustering.py
│   ├── schemas/models.py
│   └── config.py
├── scripts/
│   ├── run_evaluation_pipeline.py  # runs everything end-to-end
│   ├── run_preprocessing.py
│   ├── build_eval_dataset.py
│   ├── run_full_pipeline.py        # demo: match resumes against a JD
│   └── run_tfidf_baseline.py
├── outputs/
│   ├── results/                    # CSVs and text summaries
│   └── visualizations/             # charts (PNG) and markdown tables
├── requirements.txt
└── setup_env.py
```

---

## Requirements

- Python 3.11+ (tested on 3.12 and 3.13)
- ~10 GB disk space for data and models
- AWS credentials with Bedrock access (only needed for the LLM-as-Judge step, everything else runs locally)

---

## Setup

```bash
git clone <repo-url>
cd resume-job-matching-system
python3 setup_env.py
```

This creates the venv, installs dependencies, and downloads the dataset. Takes about 10-15 minutes. If you already have the data:

```bash
python3 setup_env.py --skip-download
```

Then activate the environment:

```bash
source venv/bin/activate
```

All commands below assume the venv is active.

---

## Quick start

### Run the demo

```bash
python3 scripts/run_full_pipeline.py
```

This runs a sample resume against a sample JD and prints the match score, matched/missing skills, and explanation.

### Match your own files

```bash
# single resume vs JD
python3 scripts/run_full_pipeline.py --resume resume.txt --jd job.txt

# rank multiple candidates
python3 scripts/run_full_pipeline.py --rank alice.txt bob.txt carol.txt --jd job.txt
```

---

## Running the full evaluation

The easiest way is the pipeline runner, which executes all 12 steps in order:

```bash
# full run from scratch (about 1-2 hours, includes fine-tuning)
python3 scripts/run_evaluation_pipeline.py

# skip preprocessing if data already exists (~6 min)
python3 scripts/run_evaluation_pipeline.py --skip-preprocessing --skip-ablation

# skip LLM judge if you dont have AWS credentials
python3 scripts/run_evaluation_pipeline.py --skip-llm
```

Or run steps individually:

```bash
# Step 1: preprocessing (parse, taxonomy, IDF, train/test split, fine-tune)
python3 scripts/run_preprocessing.py

# Step 2: build evaluation dataset (5000 pairs with graded relevance)
python3 scripts/build_eval_dataset.py --force

# Step 3: TF-IDF benchmark
python3 scripts/run_tfidf_baseline.py --benchmark --eval-pairs

# Step 4: full 3-way evaluation (TF-IDF vs Skill-IDF vs Multi-Agent)
python3 src/evaluation/evaluate_all.py

# Step 5: ablation study
python3 src/evaluation/ablation_study.py --n-jds 20

# Step 6: fairness/bias analysis
python3 src/evaluation/fairness_analysis.py

# Step 7: skill miner quality check
python3 src/evaluation/evaluate_skill_miner.py

# Step 8: generate charts
python3 src/evaluation/visualize_results.py

# Step 9: skill clustering (t-SNE)
python3 src/evaluation/skill_clustering.py

# Step 10: stress test (domain discrimination)
python3 src/evaluation/stress_test.py

# Step 11: statistical significance + improvement report
python3 src/evaluation/statistical_tests.py

# Step 12: LLM-as-Judge (needs AWS Bedrock)
python3 src/evaluation/llm_judge.py --profile hackathon
```

Steps 5-12 can run in any order after Step 4. Results go to `outputs/results/`, charts to `outputs/visualizations/`.

---

## Results

Evaluated on 5,000 JD-resume pairs (10 keywords x 25 JDs x 20 candidates each), with both binary and graded relevance labels.

### Main comparison

| Method | NDCG@5 | Precision@5 | MAP | 95% CI |
|---|---|---|---|---|
| TF-IDF | 0.7197 | 0.7136 | 0.3066 | [0.692, 0.748] |
| Skill-IDF | 0.7134 | 0.6560 | 0.2972 | [0.685, 0.741] |
| **Multi-Agent+IDF** | **0.8145** | **0.8032** | **0.3676** | **[0.790, 0.839]** |

Multi-Agent improves NDCG@5 by +13.2% over TF-IDF (p = 1.35e-08, Wilcoxon signed-rank test). Confidence intervals do not overlap.

### Graded relevance

Using skill-overlap based labels (0=poor, 1=partial, 2=strong) instead of binary keyword matching:

| Method | Binary NDCG@5 | Graded NDCG@5 | Gap vs TF-IDF |
|---|---|---|---|
| TF-IDF | 0.7197 | 0.6297 | baseline |
| Skill-IDF | 0.7134 | 0.7082 | +12.5% |
| **Multi-Agent+IDF** | **0.8145** | **0.7704** | **+22.3%** |

Under graded labels the advantage grows because Multi-Agent is better at ranking within the same domain, not just across domains.

### Independent validation (LLM-as-Judge)

200 pairs rated by AWS Bedrock Nova Lite (0-2 scale). Spearman correlation with each method's rankings:

| Method | Spearman rho |
|---|---|
| TF-IDF | 0.369 |
| Skill-IDF | 0.311 |
| **Multi-Agent+IDF** | **0.577** |

### Fine-tuning

all-MiniLM-L6-v2 fine-tuned on 43K domain pairs. Base model score: 0.106, fine-tuned: 0.374.

---

## Evaluation methodology

We use IR (information retrieval) metrics since there are no real hiring decisions in the dataset. The ground truth is based on domain relevance:

- **Binary labels**: same keyword category = relevant (1), different = irrelevant (0)
- **Graded labels**: same domain + high skill overlap = 2, same domain + low overlap = 1, different domain = 0

Metrics reflect ranking quality, not hiring accuracy. The claim is: "the system ranks relevant candidates higher than baselines under proxy relevance signals, and this ranking aligns with independent LLM judgment."

---

## Using the orchestrator in code

```python
import sys
sys.path.insert(0, "src")
from agents.orchestrator import SkillMiningOrchestrator

orchestrator = SkillMiningOrchestrator()

# score one pair
result = orchestrator.run(
    resume_text="Skills: Python, FastAPI, PostgreSQL, Docker...",
    jd_text="Requirements:\n- Python\n- FastAPI\n- PostgreSQL",
    resume_id="candidate-001",
    jd_id="job-001",
)
print(f"Final score: {result['final_score']}")

# rank multiple candidates against one JD
candidates = [
    ("candidate-001", "Alice ... Skills: Python, Django ..."),
    ("candidate-002", "Bob   ... Skills: JavaScript, React ..."),
]

ranked = orchestrator.rank_candidates(
    jd_text="Senior Python Engineer...",
    candidates=candidates,
    jd_id="job-001",
)

for r in ranked:
    print(f"Rank {r['rank']} | Score {r['final_score']:.3f} | {r['resume_id']}")
    print(f"Matched: {r['matched_skills']}")
    print(f"Missing: {r['missing_skills']}")
    print(r['explanation'])
```

---

## Output files

After running the full pipeline, you get:

**outputs/results/**
- `comparison_results.csv` - per-keyword metrics for all 3 methods
- `avg_comparison_results.csv` - macro-averaged metrics
- `detailed_scores.csv` - all 5000 pair scores
- `graded_relevance_results.csv` - NDCG under graded labels
- `significance_tests.csv` - Wilcoxon p-values
- `improvement_report.csv` - absolute and relative gains
- `confidence_intervals.csv` - bootstrap 95% CIs
- `stress_test_separation.csv` - AUC and effect sizes
- `llm_judge_scores.csv` - per-pair LLM ratings
- `llm_judge_correlations.csv` - Spearman correlations
- `ablation_results.csv` - component contribution
- `skill_miner_evaluation.txt` - coverage and precision
- `tfidf_benchmark.csv` - TF-IDF per-keyword results

**outputs/visualizations/**
- `metric_comparison.png` - bar chart comparing all metrics
- `ndcg_comparison.png` - NDCG@5 focused chart
- `score_distributions.png` - relevant vs irrelevant histograms
- `ablation_study.png` - component contribution chart
- `fairness_by_experience.png` - score distribution by experience level
- `fairness_by_category.png` - score distribution by keyword category
- `stress_test_auc.png` - AUC comparison
- `stress_test_distributions.png` - score distributions by difficulty
- `skill_clusters.png` - t-SNE skill embedding visualization
- `category_similarity.png` - skill category heatmap
- `llm_judge_distributions.png` - method scores colored by LLM rating

---

## Troubleshooting

**ModuleNotFoundError** - make sure the venv is active: `source venv/bin/activate`

**FAISS fails on Mac** - install CPU version: `pip install faiss-cpu`

**Fine-tuning out of memory** - reduce batch size:
```bash
python3 scripts/run_preprocessing.py --finetune-samples 2000 --finetune-epochs 2 --batch-size 8
```

**LLM judge fails** - check AWS credentials: `aws sts get-caller-identity --profile hackathon`. The LLM step is optional, everything else runs without AWS.

---

## Dataset

Djinni dataset (Ukrainian tech job market, from HuggingFace):
- 210,250 resumes filtered to ~81K after quality filtering
- 141,897 job descriptions filtered to ~95K
- 21 keyword categories: .NET, C++, DevOps, Flutter, Golang, Java, JavaScript, Node.js, PHP, Python, Ruby, Scala, SQL, iOS, Unity, Data Analyst, Data Engineer, Data Science, Business Analyst, QA, QA Automation
- Evaluation set: 5,000 pairs from held-out test split (10 keywords, 250 JDs)

---

## Team

CSE 572 - Data Mining, Arizona State University
