"""
Skill Mining Agent
Maps extracted skills to O*NET taxonomy and identifies latent skills
"""
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd
import re
from loguru import logger

from .base_agent import BaseAgent


class SkillMiningAgent(BaseAgent):
    """
    Agent responsible for mining skills from resume and job descriptions.
    Uses O*NET taxonomy and embeddings for skill matching.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("SkillMiningAgent", config)
        config = config or {}
        skills_file = config.get("skills_file", "data/processed/skills/skills_master.csv")
        self.skills_file = Path(skills_file)
        self.skill_entries = self._load_skills()

    # Mine skills from parsed resume/job data.
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_input(input_data, ["text", "explicit_skills"]):
            return {"error": "Missing required fields"}
        text = input_data["text"] or ""
        explicit_skills = input_data["explicit_skills"] or []
        mapped_explicit = self._map_to_taxonomy(explicit_skills)
        latent_skills = self._infer_latent_skills(text)
        # merge for category
        merged_skills = mapped_explicit + latent_skills
        result = {
            "explicit_skills": mapped_explicit,
            "latent_skills": latent_skills,
            "skill_categories": self._categorize_skills(merged_skills),
        }
        logger.info(
            f"Mined {len(result['explicit_skills'])} explicit and "
            f"{len(result['latent_skills'])} latent skills"
        )
        return result

    #Load skills into searchable formats
    def _load_skills(self) -> List[Dict[str, Any]]:
        if not self.skills_file.exists():
            logger.warning(f"Skills file not found: {self.skills_file}")
            return []
        df = pd.read_csv(self.skills_file)
        df.columns = df.columns.str.strip()
        required_cols = {"canonical_skill", "aliases", "category"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in skills file: {missing}")
        entries = []
        for _, row in df.iterrows():
            canonical_skill = str(row["canonical_skill"]).strip().lower()
            aliases = str(row["aliases"]).strip().lower()
            category = str(row["category"]).strip().lower()
            alias_list = [a.strip() for a in aliases.split("|") if a.strip()]
            if canonical_skill and canonical_skill not in alias_list:
                alias_list.insert(0, canonical_skill)
            entries.append({
                "canonical_skill": canonical_skill,
                "aliases": alias_list,
                "category": category,
            })
        logger.info(f"Loaded {len(entries)} skills from {self.skills_file}")
        return entries

    # Map skills to taxonomy
    def _map_to_taxonomy(self, skills: List[str]) -> List[Dict[str, Any]]:
        mapped = []
        seen = set()
        for skill in skills:
            skill_norm = self._normalize_text(skill)

            for entry in self.skill_entries:
                if skill_norm in entry["aliases"]:
                    key = entry["canonical_skill"]
                    if key not in seen:
                        seen.add(key)
                        mapped.append({
                            "skill": entry["canonical_skill"],
                            "matched_alias": skill_norm,
                            "category": entry["category"],
                            "source": "explicit",
                            "confidence": 1.0,
                        })
                    break
        return mapped

    # Deduce latent skills
    def _infer_latent_skills(self, text: str) -> List[Dict[str, Any]]:
        text_norm = self._normalize_text(text)
        inferred = []
        seen = set()

        for entry in self.skill_entries:
            matched_alias = None

            for alias in entry["aliases"]:
                pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
                if re.search(pattern, text_norm, flags=re.IGNORECASE):
                    matched_alias = alias
                    break

            if matched_alias:
                key = entry["canonical_skill"]
                if key not in seen:
                    seen.add(key)
                    inferred.append({
                        "skill": entry["canonical_skill"],
                        "matched_alias": matched_alias,
                        "category": entry["category"],
                        "source": "latent",
                        "confidence": 0.85,
                    })

        return inferred

    # Group skills by taxonomy
    def _categorize_skills(self, skills: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        categorized: Dict[str, List[str]] = {}
        for item in skills:
            category = item.get("category", "other")
            skill = item.get("skill")
            if not skill:
                continue
            if category not in categorized:
                categorized[category] = []
            if skill not in categorized[category]:
                categorized[category].append(skill)
        return categorized

    #Text normalization
    def _normalize_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text
