"""
run this after parse_resumes.py and parse_jds.py are done
usage:
    python3 src/preprocess/rebuild_taxonomy.py
"""
import sys
import re
from pathlib import Path
from collections import Counter

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import PROCESSED_DIR, SKILLS_TAXONOMY_FILE


# same cleaning logic as skill_miner
def clean_skill(skill: str) -> str:
    s = re.sub(r'[\x00-\x1f\x7f]', ' ', str(skill))
    s = re.sub(r'^[\s"\'`!@#\$%\^&\*\(\)\[\]{}<>\\|/]+', '', s)
    s = re.sub(r'[\s"\'`!@#\$%\^&\*\(\)\[\]{}<>\\|/]+$', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()


def is_valid_skill(skill: str) -> bool:
    if len(skill) < 2 or len(skill) > 50:
        return False
    # skip things that look like sentences or have too many words
    if len(skill.split()) > 4:
        return False
    # must have at least one letter
    if not any(c.isalpha() for c in skill):
        return False
    return True


def main():
    print("loading parsed resumes and JDs...")
    resumes = pd.read_parquet(PROCESSED_DIR / "resumes_parsed.parquet")
    jds     = pd.read_parquet(PROCESSED_DIR / "jds_parsed.parquet")
    print(f"resumes: {len(resumes):,} | jds: {len(jds):,}")

    # load existing taxonomy to preserve category labels if it exists
    if SKILLS_TAXONOMY_FILE.exists():
        old_tax = pd.read_csv(SKILLS_TAXONOMY_FILE)
        old_categories = dict(zip(old_tax['skill'].str.lower(), old_tax['category']))
        print(f"loaded {len(old_categories):,} existing category labels")
    else:
        old_tax = pd.DataFrame(columns=["skill", "category"])
        old_categories = {}
        print("no existing taxonomy found, building from scratch")

    # count skill frequencies from resumes
    resume_counts: Counter = Counter()
    for skills in resumes['raw_skills']:
        skills = skills.tolist() if hasattr(skills, 'tolist') else list(skills)
        for s in skills:
            cleaned = clean_skill(s)
            if cleaned and is_valid_skill(cleaned):
                resume_counts[cleaned] += 1

    # count skill frequencies from JDs (required + preferred)
    jd_counts: Counter = Counter()
    for _, row in jds.iterrows():
        req  = row.get('required_skills', [])
        pref = row.get('preferred_skills', [])
        req  = req.tolist()  if hasattr(req,  'tolist') else list(req)
        pref = pref.tolist() if hasattr(pref, 'tolist') else list(pref)
        for s in req + pref:
            cleaned = clean_skill(s)
            if cleaned and is_valid_skill(cleaned):
                jd_counts[cleaned] += 1

    print(f"unique skills from resumes: {len(resume_counts):,}")
    print(f"unique skills from JDs: {len(jd_counts):,}")

    # combine all skills
    all_skills = set(resume_counts.keys()) | set(jd_counts.keys())
    print(f"total unique skills combined: {len(all_skills):,}")

    rows = []
    for skill in all_skills:
        r_freq = resume_counts.get(skill, 0)
        j_freq = jd_counts.get(skill, 0)
        total  = r_freq + j_freq

        # skip very rare skills (likely noise)
        if total < 3:
            continue

        # use existing category if available, default to technical
        category = old_categories.get(skill, "technical")

        rows.append({
            "skill": skill,
            "total_freq": total,
            "resume_freq": r_freq,
            "jd_freq": j_freq,
            "category": category,
            "in_resume": r_freq > 0,
            "in_jd": j_freq > 0,
        })

    df = pd.DataFrame(rows).sort_values("total_freq", ascending=False).reset_index(drop=True)
    print(f"skills after noise filter (freq >= 3): {len(df):,}")
    print(f"category breakdown:")
    for cat, cnt in df['category'].value_counts().items():
        print(f"  {cat}: {cnt:,}")

    # backup old taxonomy (if it existed)
    if len(old_tax) > 0:
        backup = SKILLS_TAXONOMY_FILE.parent / "skills_master_backup.csv"
        old_tax.to_csv(backup, index=False)
        print(f"backup saved to {backup}")

    # save new taxonomy
    SKILLS_TAXONOMY_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SKILLS_TAXONOMY_FILE, index=False)
    print(f"new taxonomy saved to {SKILLS_TAXONOMY_FILE}")

    # write summary to outputs
    out_dir = ROOT / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("TAXONOMY REBUILD SUMMARY")
    lines.append("=" * 40)
    lines.append(f"old taxonomy: {len(old_tax):,} skills")
    lines.append(f"new taxonomy: {len(df):,} skills")
    lines.append(f"removed: {len(old_tax) - len(df):,} (noise from non-technical roles)")
    lines.append(f"added: {len(set(df['skill']) - set(old_tax['skill'].str.lower())):,} (new from filtered dataset)")
    lines.append(f"min frequency: {df['total_freq'].min()}")
    lines.append(f"max frequency: {df['total_freq'].max()}")
    lines.append("\ncategory breakdown:")
    for cat, cnt in df['category'].value_counts().items():
        lines.append(f"  {cat}: {cnt:,}")
    lines.append("\ntop 20 skills:")
    for _, row in df.head(20).iterrows():
        lines.append(f"  {row['skill']:<30} total={row['total_freq']:>6,}  resume={row['resume_freq']:>6,}  jd={row['jd_freq']:>6,}  [{row['category']}]")

    summary_path = out_dir / "taxonomy_rebuild_summary.txt"
    summary_path.write_text("\n".join(lines))
    print(f"summary saved to {summary_path}")


if __name__ == "__main__":
    main()
