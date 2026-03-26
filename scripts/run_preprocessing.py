"""
Combined preprocessing script:
1. Resume preprocessing
2. O*NET/base skills preprocessing
3. New skill candidate mining from resumes
"""
import sys
import re
import pandas as pd
from pathlib import Path
from collections import Counter
from loguru import logger
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s\+\#\.\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_resumes():
    logger.info("Start resume preprocessing...")

    resume_path_primary = RAW_DATA_DIR / "resumes" / "PrimaryResDataSet.csv"
    resume_path_secondary = RAW_DATA_DIR / "resumes" / "SecondaryResDataSet.csv"
    resume_sources = [resume_path_primary, resume_path_secondary]
    datafiles = []
    for resume_path in resume_sources:
        if not resume_path.exists():
            logger.error(f"Could not find dataset at {resume_path}")
            continue
        df = pd.read_csv(resume_path)
        df.columns = df.columns.str.strip()
        if "Primary" in resume_path.name:
            if "Resume_str" not in df.columns:
                logger.error(f"Resume_str missing in {resume_path.name}")
                continue

            df["raw_text"] = df["Resume_str"].fillna("")
            if "Category" not in df.columns:
                df["Category"] = None
        elif "Secondary" in resume_path.name:
            cols = [
                c for c in [
                    "career_objective",
                    "skills",
                    "degree_names",
                    "major_field_of_studies",
                    "professional_company_names",
                    "positions",
                    "responsibilities",
                    "certification_skills",
                    "languages"
                ] if c in df.columns
            ]
            if not cols:
                logger.error(f"No usable columns found in {resume_path.name}")
                continue
            df["raw_text"] = df[cols].fillna("").agg(" ".join, axis=1)
            df["Category"] = None
        else:
            logger.error(f"Unknown dataset format: {resume_path.name}")
            continue
        df["cleaned_text"] = df["raw_text"].apply(clean_text)
        df = df[df["cleaned_text"] != ""].copy()
        df = df.drop_duplicates(subset=["cleaned_text"]).copy()

        df = df.reset_index(drop=True)
        df["resume_id"] = [f"{resume_path.stem}_{i:05d}" for i in range(len(df))]
        df["data_source"] = resume_path.name
        df = df[["resume_id", "cleaned_text", "Category", "data_source"]]
        datafiles.append(df)
        logger.info(f"Processed {len(df)} resumes from {resume_path.name}")

    if datafiles:
        combined_df = pd.concat(datafiles, ignore_index=True)
        output_file = PROCESSED_DATA_DIR / "resumes_parsed.parquet"
        combined_df.to_parquet(output_file, index=False)
        logger.info(f"Saved resumes to {output_file}")
        return combined_df

    logger.error("No data files were processed.")
    return pd.DataFrame()

def preprocess_base_skills():
    logger.info("Creating base skills_master.csv...")
    skill_op_dir = PROCESSED_DATA_DIR / "skills"
    skill_op_dir.mkdir(parents=True, exist_ok=True)
    output_file = skill_op_dir / "skills_master.csv"

    base_skills = [
        ("python", "python|py", "programming"),
        ("java", "java", "programming"),
        ("javascript", "javascript|js", "programming"),
        ("sql", "sql|mysql|postgresql", "database"),
        ("machine learning", "machine learning|ml", "ai"),
        ("deep learning", "deep learning|dl", "ai"),
        ("data science", "data science", "ai"),
        ("natural language processing", "natural language processing|nlp", "ai"),
        ("pandas", "pandas", "library"),
        ("numpy", "numpy", "library"),
        ("scikit-learn", "scikit-learn|sklearn", "library"),
        ("tensorflow", "tensorflow", "library"),
        ("pytorch", "pytorch", "library"),
        ("react", "react|reactjs", "frontend"),
        ("html", "html|html5", "frontend"),
        ("css", "css|css3", "frontend"),
        ("node.js", "node.js|nodejs|node", "backend"),
        ("django", "django", "backend"),
        ("flask", "flask", "backend"),
        ("aws", "aws|amazon web services", "cloud"),
        ("azure", "azure", "cloud"),
        ("docker", "docker", "devops"),
        ("kubernetes", "kubernetes|k8s", "devops"),
        ("git", "git", "tools"),
        ("linux", "linux", "tools"),
        ("jira", "jira|atlassian jira", "tools"),
        ("excel", "microsoft excel|excel", "office"),
        ("tableau", "tableau", "analytics"),
        ("power bi", "power bi|powerbi", "analytics"),
        ("project management", "project management", "project_management"),
        ("agile", "agile", "project_management"),
        ("scrum", "scrum", "project_management"),
        ("accounting", "accounting", "finance"),
        ("financial analysis", "financial analysis", "finance"),
        ("human resources", "human resources|hr", "hr"),
        ("recruitment", "recruitment|talent acquisition", "hr"),
        ("onboarding", "onboarding", "hr"),
        ("customer service", "customer service", "customer_support"),
        ("technical support", "technical support", "customer_support"),
        ("troubleshooting", "troubleshooting", "customer_support"),
        ("autocad", "autocad", "engineering"),
        ("patient care", "patient care", "healthcare"),
        ("clinical research", "clinical research", "healthcare"),
        ("seo", "seo|search engine optimization", "marketing"),
        ("digital marketing", "digital marketing", "marketing"),
        ("google analytics", "google analytics", "marketing"),
        ("logistics", "logistics", "supply_chain"),
        ("communication", "communication", "soft_skills"),
        ("leadership", "leadership", "soft_skills"),
        ("teamwork", "teamwork", "soft_skills"),
        ("problem solving", "problem solving", "soft_skills"),
    ]

    skills_df = pd.DataFrame(base_skills, columns=["canonical_skill", "aliases", "category"])
    skills_df.to_csv(output_file, index=False)

    logger.info(f"Saved base skills to {output_file}")
    return skills_df

def load_known_terms(skills_df: pd.DataFrame) -> set:
    known_terms = set()
    for _, row in skills_df.iterrows():
        canonical = str(row["canonical_skill"]).strip().lower()
        aliases = str(row["aliases"]).strip().lower()
        if canonical:
            known_terms.add(canonical)
        for alias in aliases.split("|"):
            alias = alias.strip()
            if alias:
                known_terms.add(alias)
    return known_terms

def extract_ngrams(text: str) -> list:
    tokens = text.split()
    terms = []
    for i in range(len(tokens)):
        terms.append(tokens[i])
        if i < len(tokens) - 1:
            terms.append(tokens[i] + " " + tokens[i + 1])
    return terms

def is_good_candidate(term: str) -> bool:
    bad_terms = {
        "experience", "experienced", "responsible", "responsibilities",
        "worked", "working", "ability", "knowledge", "skills", "skill",
        "project", "projects", "team", "work", "role", "roles",
        "company", "organization", "using", "used", "development",
        "design", "support", "management", "analysis", "system", "systems",
        "process", "processes", "task", "tasks", "objective", "position",
        "degree", "education", "software", "application", "applications"
    }

    if not term or len(term) < 2:
        return False

    if term in bad_terms:
        return False

    if re.fullmatch(r"\d+", term):
        return False

    return True

def expand_skills_from_resumes(resumes_df: pd.DataFrame, skills_df: pd.DataFrame):
    logger.info("Mining new skill candidates from resumes...")
    known_terms = load_known_terms(skills_df)
    counter = Counter()
    for text in resumes_df["cleaned_text"]:
        text = clean_text(text)
        terms = extract_ngrams(text)

        for term in terms:
            if term in known_terms:
                continue
            if not is_good_candidate(term):
                continue
            counter[term] += 1

    candidate_df = pd.DataFrame(counter.items(), columns=["canonical_skill", "frequency"])
    candidate_df = candidate_df.sort_values("frequency", ascending=False)
    candidate_df = candidate_df[candidate_df["frequency"] >= 3].copy()

    if candidate_df.empty:
        logger.warning("No new candidate skills found.")
        return

    candidate_df["aliases"] = candidate_df["canonical_skill"]
    candidate_df["category"] = "other"

    output_file = PROCESSED_DATA_DIR / "skills/new_skill_candidates.csv"
    candidate_df.to_csv(output_file, index=False)

    logger.info(f"Saved new skill candidates to {output_file}")

def main():
    logger.info("Start full preprocessing pipeline...")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    resumes_df = preprocess_resumes()
    if resumes_df.empty:
        logger.error("Stopping because resume preprocessing failed.")
        return
    skills_df = preprocess_base_skills()
    expand_skills_from_resumes(resumes_df, skills_df)
    logger.info("Preprocessing complete.")

if __name__ == "__main__":
    main()