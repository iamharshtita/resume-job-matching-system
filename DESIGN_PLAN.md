# Design Plan: Resume-Job Matching System

## 1. Project Overview

### 1.1 Objective
Build a multi-agent AI system that intelligently matches job candidates with positions, providing explainable results that outperform traditional keyword-based ATS systems.

### 1.2 Key Features
- Semantic skill matching beyond keywords
- Evidence-backed explanations
- Fairness across demographic variations
- Multi-dimensional scoring (skills, experience, education, title match)

### 1.3 Tech Stack
- **Agent Framework**: Strands Agents SDK (or custom orchestration)
- **LLM**: GPT-4o-mini or Llama 3 (local)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **NLP**: spaCy, regex, dateparser
- **Data**: Pandas, NumPy
- **Skills Database**: O*NET taxonomy

---

## 2. Architecture

### 2.1 Multi-Agent System Design

```
┌─────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                          │
│              (Coordinates all agent interactions)            │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Resume     │ │     Job      │ │   Matching   │
│   Parser     │ │   Parser     │ │    Agent     │
│              │ │              │ │              │
│ - Extract    │ │ - Extract    │ │ - Compute    │
│   name, edu  │ │   req. skills│ │   scores     │
│ - Parse work │ │ - Parse      │ │ - Multi-dim  │
│   history    │ │   experience │ │   evaluation │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼
                ┌──────────────┐
                │    Skill     │
                │   Mining     │
                │    Agent     │
                │              │
                │ - O*NET map  │
                │ - Latent     │
                │   skills     │
                │ - Embeddings │
                └──────┬───────┘
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
    ┌──────────────┐      ┌──────────────┐
    │   Ranking &  │      │  Explanation │
    │   Scoring    │      │    Agent     │
    │              │      │              │
    │ - Final rank │      │ - Evidence   │
    │ - Aggregate  │      │ - Reasoning  │
    │   scores     │      │ - Justif.    │
    └──────────────┘      └──────────────┘
```

### 2.2 Agent Responsibilities

#### Resume Parser Agent
- **Input**: Raw resume text (PDF/DOCX → text)
- **Output**: Structured JSON with:
  - Name, contact info
  - Education (degree, institution, dates)
  - Work experience (titles, companies, dates, descriptions)
  - Explicitly mentioned skills
- **Tools**: spaCy NER, regex, dateparser

#### Job Parser Agent
- **Input**: Job description text
- **Output**: Structured JSON with:
  - Job title, company
  - Required vs. preferred skills
  - Experience requirements
  - Education requirements
  - Responsibilities
- **Tools**: Similar to Resume Parser

#### Skill Mining Agent
- **Input**: Parsed resume/job + raw text
- **Output**: Enhanced skill data:
  - O*NET mapped skills (standardized)
  - Latent skills (inferred from context)
  - Skill categories (technical, soft, domain)
  - Proficiency levels (if determinable)
- **Tools**:
  - O*NET database for taxonomy
  - Sentence embeddings for similarity
  - LLM for context understanding

#### Job-Resume Matching Agent
- **Input**: Enhanced resume + enhanced job data
- **Output**: Multi-dimensional scores:
  - Skill match score (0-1)
  - Experience match score (0-1)
  - Education match score (0-1)
  - Title match score (0-1)
  - Composite score with weights
- **Scoring Logic**:
  ```
  Skill Score = (matched_required_skills / total_required_skills) * 0.7 +
                (matched_preferred_skills / total_preferred_skills) * 0.3

  Experience Score = min(candidate_years / required_years, 1.0)

  Education Score = candidate_level / required_level (capped at 1.0)

  Title Score = cosine_similarity(candidate_titles, job_title)

  Final Score = 0.4*Skill + 0.3*Experience + 0.2*Education + 0.1*Title
  ```

#### Ranking & Explanation Agent
- **Input**: All candidate match results for a job
- **Output**:
  - Ranked candidate list
  - Explanation for each candidate
- **Explanation Format**:
  ```
  Candidate X is ranked #N because:
  - Matches 8/10 required skills (Python, ML, AWS...)
  - Has 5 years experience (requirement: 3+)
  - Bachelor's degree matches requirement
  - Previous role "Data Scientist" similar to "ML Engineer"

  Gaps:
  - Missing skill: Kubernetes
  - No cloud certification mentioned
  ```

---

## 3. Data Pipeline

### 3.1 Data Collection
- **Primary Resume Dataset**: Kaggle Resume Dataset (~2,000 resumes)
- **Secondary Dataset**: UpdatedResumeDataSet.csv
- **Job Postings**: LinkedIn postings from Kaggle or custom scraping
- **O*NET Database**: Skills, knowledge, abilities taxonomy

### 3.2 Preprocessing Steps
1. **Text Extraction**: PDF/DOCX → plain text
2. **Cleaning**: Remove artifacts, normalize whitespace
3. **Sectioning**: Identify resume sections (education, experience, skills)
4. **Tokenization**: spaCy pipeline
5. **Embedding Generation**: Store embeddings for fast retrieval

### 3.3 Data Storage Structure
```
data/
├── raw/
│   ├── resumes_primary/          # Original resume files
│   ├── resumes_secondary/        # Additional resumes
│   ├── job_postings/             # Job description files
│   └── onet/                     # O*NET database files
├── interim/
│   └── embeddings/               # Precomputed embeddings
├── processed/
│   ├── resumes_parsed.parquet    # Structured resume data
│   ├── jobs_parsed.parquet       # Structured job data
│   └── skills_mapped.parquet     # O*NET mapped skills
└── evaluation/
    ├── test_pairs.csv            # Human-labeled test pairs
    └── fairness_tests.csv        # Demographic variation tests
```

---

## 4. Implementation Phases

### Phase 1: Data & Infrastructure (Week 1-2)
**Owner**: Member 1
- [ ] Download and organize datasets
- [ ] Set up project structure and Git repository
- [ ] Implement Resume Parser Agent
  - Name extraction
  - Education parsing
  - Experience timeline extraction
- [ ] Create data preprocessing scripts
- [ ] Unit tests for parser

### Phase 2: Skill Mining (Week 2-3)
**Owner**: Member 2
- [ ] Download and integrate O*NET database
- [ ] Implement Skill Mining Agent
  - O*NET taxonomy mapping
  - Embedding-based similarity
  - Latent skill inference
- [ ] Build skill categorization logic
- [ ] Create embedding pipeline (FAISS index)
- [ ] Unit tests for skill mining

### Phase 3: Matching & Scoring (Week 3-4)
**Owner**: Member 3
- [ ] Implement Job Parser Agent (similar to Resume Parser)
- [ ] Implement Matching Agent
  - Multi-dimensional scoring
  - Weighted aggregation
- [ ] Implement baseline systems:
  - TF-IDF baseline
  - Single-agent LLM baseline
- [ ] Unit tests for scoring logic

### Phase 4: Ranking & Explanation (Week 4-5)
**Owner**: Member 4
- [ ] Implement Ranking & Explanation Agent
  - Evidence extraction
  - Natural language explanation generation
- [ ] Implement Orchestrator
  - Agent coordination
  - Error handling
  - Caching layer
- [ ] Integration tests for full pipeline

### Phase 5: Evaluation (Week 5-6)
**Team Effort**
- [ ] Create evaluation dataset (100-200 resume-job pairs)
- [ ] Human annotation for ground truth
- [ ] Implement evaluation metrics:
  - NDCG, Spearman correlation
  - Precision@K
  - Explanation quality ratings
  - Fairness metrics
- [ ] Run ablation studies
- [ ] Comparative analysis with baselines

### Phase 6: Refinement & Documentation (Week 6-7)
**Team Effort**
- [ ] Optimize performance (caching, batching)
- [ ] Error analysis and fixes
- [ ] Final report writing
- [ ] Demo preparation
- [ ] Code cleanup and documentation

---

## 5. Evaluation Plan

### 5.1 Metrics

#### Ranking Quality
- **NDCG@10**: Normalized Discounted Cumulative Gain
- **Spearman Correlation**: Correlation with human rankings
- **Precision@K**: Accuracy of top-K results

#### Explanation Quality (Human Evaluation)
- **Usefulness**: Does the explanation help understand the match?
- **Clarity**: Is the explanation easy to understand?
- **Faithfulness**: Does the explanation match the actual scoring?

#### Fairness
- **Score Stability**: Compare scores when name/gender/ethnicity is varied
- **Demographic Parity**: Equal ranking distribution across groups

#### Efficiency
- **Runtime**: Time per resume-job pair
- **API Cost**: Token usage and cost per match

### 5.2 Baselines

1. **TF-IDF Baseline**
   - Skills as keywords
   - Cosine similarity between TF-IDF vectors
   - No contextual understanding

2. **Single-Agent LLM**
   - Direct prompt to GPT-4o-mini
   - No structured agents
   - Minimal skill taxonomy

### 5.3 Test Cases
- **Standard matches**: Clear fit between resume and job
- **Ambiguous cases**: Candidate slightly underqualified
- **Negative cases**: Clear mismatch
- **Fairness cases**: Same resume with varied demographics

---

## 6. Risk Management

### 6.1 Technical Risks

| Risk | Mitigation |
|------|------------|
| O*NET integration complexity | Start early, use pre-processed subset if needed |
| LLM API costs | Use caching, batch requests, consider local models |
| Parsing errors on diverse formats | Test on varied resume formats, fallback parsing |
| Slow embedding generation | Pre-compute and cache embeddings |

### 6.2 Project Risks

| Risk | Mitigation |
|------|------------|
| Scope creep | Strict MVP definition, defer nice-to-haves |
| Integration delays | Weekly sync meetings, clear interfaces |
| Data quality issues | Multiple datasets, data cleaning pipeline |
| Evaluation complexity | Start evaluation dataset early |

---

## 7. Success Criteria

### Minimum Viable Product (MVP)
- ✅ Parse resumes and extract structured data
- ✅ Map skills to O*NET taxonomy
- ✅ Compute multi-dimensional match scores
- ✅ Rank candidates for a job
- ✅ Generate basic explanations
- ✅ Outperform TF-IDF baseline on NDCG@10

### Stretch Goals
- ⭐ Real-time processing (<5 seconds per pair)
- ⭐ Interactive explanation interface
- ⭐ Skill gap analysis and recommendations
- ⭐ Fairness guarantees (< 5% score variation)

---

## 8. Team Collaboration

### Weekly Schedule
- **Monday**: Standup (progress updates, blockers)
- **Wednesday**: Technical sync (integration, design decisions)
- **Friday**: Code review and testing

### Communication Channels
- **GitHub**: Code, PRs, issues
- **Slack/Discord**: Quick questions, daily updates
- **Google Docs**: Shared documentation, meeting notes

### Code Review Process
1. Create feature branch
2. Implement and test locally
3. Submit PR with description
4. At least 1 team member review
5. Merge after approval

---

## 9. Deliverables

### Code
- ✅ Complete working system on GitHub
- ✅ Comprehensive README
- ✅ Requirements and setup instructions
- ✅ Unit and integration tests
- ✅ Example notebooks

### Report
- Introduction and motivation
- Related work
- System architecture
- Implementation details
- Evaluation results
- Discussion and limitations
- Conclusion and future work

### Presentation
- 15-20 minute presentation
- Live demo
- Results visualization
- Q&A preparation

---

## 10. Next Steps

### Immediate Actions (This Week)
1. **Set up GitHub repository** (see below)
2. **Create project board** with tasks
3. **Download datasets** and store in data/raw/
4. **Set up development environment** (venv, install requirements)
5. **First standup meeting** to assign Phase 1 tasks

### GitHub Setup (Next Section)
Follow the guide below to initialize Git and push to GitHub.
