"""
train_matcher_weights.py
Phase 1: Tests learned weights on hardcoded resume-JD pairs with hand labels.
Phase 2 (TODO): Replace hand labels with GPT-4o-mini labels on full dataset.
1. Runs 3 JDs against 4 resumes each (12 pairs total)
2. Collects [skill_score, exp_score, edu_score, title_score] from the pipeline
3. Trains a linear regression on those 4 features using hand-assigned labels
4. Normalizes learned coefficients into weights (sum to 1)
5. Compares NDCG@4, Precision@4, Spearman for:
    TF-IDF baseline
    Multi-agent fixed weights (from config)
    Multi-agent learned weights (from regression)
6. Saves learned weights to outputs/results/learned_weights.json

Run from project root:
    python scripts/train_matcher_weights.py
"""
import sys
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ndcg_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agents.resume_parser import ResumeParserAgent
from agents.jd_parser import JDParserAgent
from agents.skill_miner import SkillMiningAgent
from agents.matcher import MatchingAgent
from config import WEIGHTS

K = 4

RESUMES = {
    "resume_python_strong": {
        "id": "resume_python_strong",
        "position": "Python Backend Developer",
        "experience_years": 5.0,
        "raw_text": """
Python Backend Developer — 5 years experience

Skills: Python, Django, PostgreSQL, Docker, AWS, REST APIs, Git, Redis, Celery

Experience:
2020 - Present: Backend Developer at TechCorp
2018 - 2020: Junior Python Developer at StartupXYZ

Education: Bachelor of Computer Science, University of Toronto, 2018

Built scalable REST APIs with Django and PostgreSQL.
Deployed services on AWS using Docker and CI/CD pipelines.
""",
    },
    "resume_python_mid": {
        "id": "resume_python_mid",
        "position": "Python Developer",
        "experience_years": 3.0,
        "raw_text": """
Python Developer — 3 years experience

Skills: Python, Flask, MySQL, REST API, Git, Linux

Experience:
2021 - Present: Python Developer at FinTechCo
2019 - 2021: Junior Developer at WebAgency

Education: Bachelor of Computer Science, 2019

Developed Flask APIs and MySQL database integrations.
""",
    },
    "resume_react": {
        "id": "resume_react",
        "position": "React Frontend Developer",
        "experience_years": 4.0,
        "raw_text": """
React Frontend Developer — 4 years experience

Skills: React, TypeScript, JavaScript, Node.js, Redux, CSS, HTML, Jest, Webpack

Experience:
2021 - Present: Frontend Developer at UIStudio
2019 - 2021: JavaScript Developer at WebCo

Education: Bachelor of Software Engineering, 2019

Built React applications with TypeScript and Redux state management.
""",
    },
    "resume_datascience": {
        "id": "resume_datascience",
        "position": "Data Scientist",
        "experience_years": 4.0,
        "raw_text": """
Data Scientist — 4 years experience

Skills: Python, Machine Learning, TensorFlow, PyTorch, Pandas, NumPy, Scikit-learn, SQL

Experience:
2020 - Present: Data Scientist at DataLab
2018 - 2020: ML Engineer at AIStartup

Education: Master of Data Science, Stanford University, 2018

Developed ML models using TensorFlow and PyTorch for classification tasks.
""",
    },
}

JDS = {
    "jd_python": {
        "id": "jd_python",
        "position": "Python Backend Developer",
        "exp_years_raw": "4y",
        "raw_text": """
Python Backend Developer

Requirements:
- Python
- Django
- PostgreSQL
- Docker
- AWS
- REST APIs
- Git

Nice to have:
- Redis
- Celery
- Kubernetes

Bachelor's degree in Computer Science or related field.
At least 4 years of Python backend development experience.
""",
    },
    "jd_react": {
        "id": "jd_react",
        "position": "React Frontend Developer",
        "exp_years_raw": "3y",
        "raw_text": """
React Frontend Developer

Requirements:
- React
- TypeScript
- JavaScript
- Redux
- HTML
- CSS

Nice to have:
- Node.js
- Jest
- Webpack

Bachelor's degree in Computer Science.
3+ years of React development experience.
""",
    },
    "jd_datascience": {
        "id": "jd_datascience",
        "position": "Data Scientist",
        "exp_years_raw": "3y",
        "raw_text": """
Data Scientist

Requirements:
- Python
- Machine Learning
- TensorFlow
- Pandas
- NumPy
- SQL

Nice to have:
- PyTorch
- Scikit-learn
- Deep Learning

Master's degree in Data Science or related field.
3+ years of machine learning experience.
""",
    },
}

# Each (resume_id, jd_id) pair is assigned a relevance score:
#   1.0 = strong match    (skills, experience, title all align)
#   0.5 = partial match   (some skill overlap, different domain)
#   0.2 = weak match      (minimal overlap)
#   0.0 = no match        (completely different domain)

GROUND_TRUTH = {
    # Python JD: python_strong is best, python_mid is decent, rest are poor
    ("resume_python_strong", "jd_python"):      1.0,
    ("resume_python_mid",    "jd_python"):      0.6,
    ("resume_react",         "jd_python"):      0.1,
    ("resume_datascience",   "jd_python"):      0.3,

    # React JD: react dev is best, others are poor
    ("resume_react",         "jd_react"):       1.0,
    ("resume_python_strong", "jd_react"):       0.1,
    ("resume_python_mid",    "jd_react"):       0.1,
    ("resume_datascience",   "jd_react"):       0.1,

    # Data Science JD: datascience is best, python devs have some overlap
    ("resume_datascience",   "jd_datascience"): 1.0,
    ("resume_python_strong", "jd_datascience"): 0.3,
    ("resume_python_mid",    "jd_datascience"): 0.2,
    ("resume_react",         "jd_datascience"): 0.0,
}


def run_pipeline(resume_parser, jd_parser, skill_miner, matcher):
    """
    Run all 12 pairs through the full pipeline.
    Returns a list of dicts with pair IDs, four component scores, and ground truth.
    """
    print("Parsing resumes...")
    parsed_resumes = {}
    for rid, res in RESUMES.items():
        parsed = resume_parser.process({
            "id": res["id"],
            "raw_text": res["raw_text"],
            "position": res["position"],
            "experience_years": res["experience_years"],
        })
        mined = skill_miner.process({
            "raw_skills": parsed["raw_skills"],
            "raw_text": parsed["raw_text"],
            "source": "resume",
        })
        parsed_resumes[rid] = {"parsed": parsed, "mined": mined}
        print(f"  {rid}: {len(mined['skills'])} skills")

    print("\nParsing JDs...")
    parsed_jds = {}
    for jid, jd in JDS.items():
        parsed = jd_parser.process({
            "id": jd["id"],
            "raw_text": jd["raw_text"],
            "position": jd["position"],
            "exp_years_raw": jd["exp_years_raw"],
        })
        mined = skill_miner.process({
            "raw_skills": parsed["required_skills"] + parsed["preferred_skills"],
            "raw_text": parsed["raw_text"],
            "source": "jd",
        })
        parsed_jds[jid] = {"parsed": parsed, "mined": mined}
        print(f"  {jid}: {len(mined['skills'])} skills")

    print("\nRunning matcher on all pairs...")
    results = []
    for (rid, jid), label in GROUND_TRUTH.items():
        res_data = parsed_resumes[rid]
        jd_data  = parsed_jds[jid]

        score = matcher.process({
            "resume_id":      rid,
            "jd_id":          jid,
            "resume_skills":  res_data["mined"],
            "jd_skills":      jd_data["mined"],
            "resume_exp_years":   res_data["parsed"].get("experience_years"),
            "jd_exp_years":       jd_data["parsed"].get("experience_years"),
            "resume_education":   res_data["parsed"].get("education", []),
            "jd_education_req":   jd_data["parsed"].get("education_requirement"),
            "resume_position":    res_data["parsed"].get("position"),
            "jd_title":           jd_data["parsed"].get("title"),
        })

        results.append({
            "resume_id":        rid,
            "jd_id":            jid,
            "label":            label,
            "skill_score":      score["skill_score"],
            "exp_score":        score["experience_score"],
            "edu_score":        score["education_score"],
            "title_score":      score["title_score"],
            "fixed_final":      score["final_score"],
        })

    return results


def learn_weights(results):
    """
    Fit a linear regression on [skill, exp, edu, title] scores → label.
    Normalize positive coefficients to sum to 1.
    Returns a weight dict in the same format as config.WEIGHTS.
    """
    X = np.array([[r["skill_score"], r["exp_score"], r["edu_score"], r["title_score"]]
                  for r in results])
    y = np.array([r["label"] for r in results])

    model = LinearRegression(fit_intercept=False, positive=True)
    model.fit(X, y)

    coefs = model.coef_
    total = coefs.sum()
    if total > 0:
        coefs = coefs / total

    return {
        "skill":      round(float(coefs[0]), 4),
        "experience": round(float(coefs[1]), 4),
        "education":  round(float(coefs[2]), 4),
        "title":      round(float(coefs[3]), 4),
    }


def apply_learned_weights(results, learned_weights):
    """Compute final score for each pair using learned weights."""
    for r in results:
        r["learned_final"] = round(
            learned_weights["skill"]      * r["skill_score"]
            + learned_weights["experience"] * r["exp_score"]
            + learned_weights["education"]  * r["edu_score"]
            + learned_weights["title"]      * r["title_score"],
            4,
        )
    return results


def tfidf_scores(results):
    """Compute TF-IDF cosine similarity for each pair as a third baseline."""
    all_texts = [RESUMES[r["resume_id"]]["raw_text"] for r in results] + \
                [JDS[r["jd_id"]]["raw_text"] for r in results]
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    vec.fit(all_texts)

    for r in results:
        r_vec = vec.transform([RESUMES[r["resume_id"]]["raw_text"]])
        j_vec = vec.transform([JDS[r["jd_id"]]["raw_text"]])
        r["tfidf_score"] = float(cosine_similarity(r_vec, j_vec)[0][0])
    return results


def compute_metrics(scores, labels, k):
    order       = np.argsort(-np.array(scores))
    rel_sorted  = np.array(labels)[order]
    prec_at_k   = (rel_sorted[:k] > 0.5).sum() / k
    try:
        ndcg = ndcg_score(
            np.array(labels).reshape(1, -1),
            np.array(scores).reshape(1, -1),
            k=k,
        )
    except Exception:
        ndcg = float("nan")
    try:
        rho, _ = spearmanr(scores, labels)
    except Exception:
        rho = float("nan")
    return {"ndcg": ndcg, "precision": prec_at_k, "spearman": rho}


def evaluate_all(results, k):
    """Average metrics across all three JDs."""
    systems = {
        "TF-IDF":         "tfidf_score",
        "Fixed weights":  "fixed_final",
        "Learned weights":"learned_final",
    }
    jd_ids = list(set(r["jd_id"] for r in results))
    output = {}
    for name, col in systems.items():
        ndcgs, precs, spears = [], [], []
        for jd_id in jd_ids:
            group   = [r for r in results if r["jd_id"] == jd_id]
            scores  = [r[col] for r in group]
            labels  = [r["label"] for r in group]
            m = compute_metrics(scores, labels, k)
            ndcgs.append(m["ndcg"])
            precs.append(m["precision"])
            spears.append(m["spearman"])
        output[name] = {
            f"ndcg@{k}":      round(np.nanmean(ndcgs),  4),
            f"precision@{k}": round(np.nanmean(precs),  4),
            "spearman":       round(np.nanmean(spears), 4),
        }
    return output


def main():
    print("=" * 65)
    print("Learned Weights Experiment — Phase 1 (10 hardcoded pairs)")
    print("=" * 65)

    resume_parser = ResumeParserAgent()
    jd_parser = JDParserAgent()
    skill_miner = SkillMiningAgent()
    matcher = MatchingAgent()

    results = run_pipeline(resume_parser, jd_parser, skill_miner, matcher)

    print("\nLearning weights from pipeline scores...")
    learned = learn_weights(results)
    print(f"  Fixed weights:   skill={WEIGHTS['skill']}, exp={WEIGHTS['experience']}, "
          f"edu={WEIGHTS['education']}, title={WEIGHTS['title']}")
    print(f"  Learned weights: skill={learned['skill']}, exp={learned['experience']}, "
          f"edu={learned['education']}, title={learned['title']}")

    results = apply_learned_weights(results, learned)
    results = tfidf_scores(results)

    metrics = evaluate_all(results, K)

    print(f"\n{'System':<20}  {'NDCG@'+str(K):>8}  {'P@'+str(K):>6}  {'Spearman':>9}")
    print("-" * 50)
    for system, m in metrics.items():
        print(f"{system:<20}  {m[f'ndcg@{K}']:>8.4f}  "
              f"{m[f'precision@{K}']:>6.4f}  {m['spearman']:>9.4f}")

    print("\n--- Per-pair breakdown ---")
    print(f"{'Resume':<30}  {'JD':<20}  {'Label':>5}  {'Fixed':>6}  {'Learned':>7}  {'TF-IDF':>7}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: (x["jd_id"], -x["label"])):
        print(f"{r['resume_id']:<30}  {r['jd_id']:<20}  "
              f"{r['label']:>5.1f}  {r['fixed_final']:>6.4f}  "
              f"{r['learned_final']:>7.4f}  {r['tfidf_score']:>7.4f}")

    out_path = ROOT / "outputs" / "results" / "learned_weights.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(learned, f, indent=2)
    print(f"\nLearned weights saved to {out_path}")


if __name__ == "__main__":
    main()
