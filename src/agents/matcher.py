"""
Matching Agent — computes a multi-dimensional fit score between a
mined resume and a mined job description.
Skill score — semantic overlap (primary):
    For each JD skill vector, find the highest cosine similarity to any
    resume skill vector, weighted by that resume skill's confidence.
    Similarities below _SEMANTIC_THRESHOLD are treated as zero.
    IDF-weighted: Rare skills count more than common skills.
Falls back to exact canonical matching when skill vectors are absent.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Set, Optional

from .base_agent import BaseAgent
from config import WEIGHTS, SKILL_WEIGHTS, ENGLISH_LEVELS, PROCESSED_DIR
from schemas.models import MatchScore

# Cosine similarity below this is treated as noise
_SEMANTIC_THRESHOLD = 0.40


class MatchingAgent(BaseAgent):

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("MatchingAgent", config)
        self._idf_weights: Optional[Dict[str, float]] = None
        self._load_idf_weights()

    def _load_idf_weights(self):
        """Load pre-computed IDF weights from skill_idf.json"""
        idf_path = PROCESSED_DIR / "skill_idf.json"
        if idf_path.exists():
            with open(idf_path, 'r') as f:
                self._idf_weights = json.load(f)
        else:
            self._idf_weights = {}

    """
    Input:
        resume_id
        jd_id   str
        resume_skills dict  — MinedSkills
        jd_skills  dict  — MinedSkills
        resume_exp_years
        jd_exp_years
        resume_english_level
        jd_english_level
        resume_position
        jd_title
    """
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        required = ["resume_id", "jd_id", "resume_skills", "jd_skills"]
        if not self.validate_input(input_data, required):
            return {"error": "Missing required fields"}

        skill_score = self._skill_score(
            input_data["resume_skills"],
            input_data["jd_skills"],
        )
        experience_score = self._experience_score(
            input_data.get("resume_exp_years"),
            input_data.get("jd_exp_years"),
        )
        english_level_score = self._english_level_score(
            input_data.get("resume_english_level"),
            input_data.get("jd_english_level"),
        )
        title_score = self._title_score(
            input_data.get("resume_position", ""),
            input_data.get("jd_title", ""),
        )

        final_score = round(
            WEIGHTS["skill"]      * skill_score
            + WEIGHTS["experience"] * experience_score
            + WEIGHTS["education"]  * english_level_score
            + WEIGHTS["title"]      * title_score,
            4,
        )

        self.log_metrics({
            "resume_id": input_data["resume_id"],
            "jd_id": input_data["jd_id"],
            "skill": round(skill_score, 3),
            "experience": round(experience_score, 3),
            "english_level": round(english_level_score, 3),
            "title": round(title_score, 3),
            "final": final_score,
        })

        result = MatchScore(
            resume_id=input_data["resume_id"],
            jd_id=input_data["jd_id"],
            jaccard_score=self._jaccard(input_data["resume_skills"], input_data["jd_skills"]),
            skill_score=skill_score,
            experience_score=experience_score,
            english_level_score=english_level_score,
            title_score=title_score,
            final_score=final_score,
        )
        return result.model_dump()

    def _skill_score(self, resume_skills: dict, jd_skills: dict) -> float:
        r_vecs = resume_skills.get("skill_vectors")
        j_vecs = jd_skills.get("skill_vectors")

        if r_vecs and j_vecs:
            semantic = self._semantic_overlap(
                np.array(r_vecs, dtype=np.float32),
                np.array(j_vecs, dtype=np.float32),
                resume_skills,
            )
            # blend semantic similarity with IDF-weighted string matching
            # semantic handles meaning, skill-idf handles rarity — different signals
            idf = self._skill_idf_score(resume_skills, jd_skills)
            return float(np.clip(0.7 * semantic + 0.3 * idf, 0.0, 1.0))

        # fallback when vectors are missing
        resume_map = self._skill_confidence_map(resume_skills)
        required   = self._canonical_set(jd_skills, source_filter="required")
        preferred  = self._canonical_set(jd_skills, source_filter="preferred")

        req_score  = self._weighted_overlap(resume_map, required)
        pref_score = self._weighted_overlap(resume_map, preferred)

        if not required and not preferred:
            return 0.0
        if required and preferred:
            return SKILL_WEIGHTS["required"] * req_score + SKILL_WEIGHTS["preferred"] * pref_score
        return req_score if required else pref_score

    def _semantic_overlap(
        self,
        r_vecs: "np.ndarray",
        j_vecs: "np.ndarray",
        resume_skills: dict,
    ) -> float:
        if r_vecs.shape[0] == 0 or j_vecs.shape[0] == 0:
            return 0.0

        confs = np.array(
            [e.get("confidence", 1.0) for e in resume_skills.get("skills", [])],
            dtype=np.float32,
        )
        if len(confs) != r_vecs.shape[0]:
            confs = np.ones(r_vecs.shape[0], dtype=np.float32)

        sim      = r_vecs @ j_vecs.T
        sim      = np.where(sim >= _SEMANTIC_THRESHOLD, sim, 0.0)
        weighted = sim * confs[:, None]
        best     = weighted.max(axis=0)

        return float(np.clip(best.mean(), 0.0, 1.0))

    def _skill_idf_score(self, resume_skills: dict, jd_skills: dict) -> float:
        # string-level IDF matching — rare JD skills that the resume has count more
        if not self._idf_weights:
            return 0.0

        resume_set    = {e.get("canonical", "").lower() for e in resume_skills.get("skills", [])}
        jd_canonicals = [e.get("canonical", "").lower() for e in jd_skills.get("skills", [])]

        if not jd_canonicals:
            return 0.0

        total_idf = matched_idf = 0.0
        for skill in jd_canonicals:
            w = self._idf_weights.get(skill, 1.0)
            total_idf += w
            if skill in resume_set:
                matched_idf += w

        return matched_idf / total_idf if total_idf > 0 else 0.0

    def _skill_confidence_map(self, mined_skills: dict) -> Dict[str, float]:
        """Return {canonical: confidence} for all skills in a MinedSkills dict."""
        result = {}
        for entry in mined_skills.get("skills", []):
            canonical = entry.get("canonical", "")
            confidence = entry.get("confidence", 1.0)
            if canonical:
                result[canonical] = max(result.get(canonical, 0.0), confidence)
        return result

    def _canonical_set(self, mined_skills: dict, source_filter: str = None) -> Set[str]:
        if source_filter == "required":
            return set(mined_skills.get("grouped", {}).get("technical", []) +
                       mined_skills.get("grouped", {}).get("domain", []))
        if source_filter == "preferred":
            skills = mined_skills.get("skills", [])
            return {e["canonical"] for e in skills if e.get("source") == "explicit"
                    and e.get("confidence", 0) < 1.0}
        return {e["canonical"] for e in mined_skills.get("skills", [])}

    def _weighted_overlap(self, resume_map: Dict[str, float], jd_set: Set[str]) -> float:
        """Exact-match fallback: confidence-weighted overlap, normalised by JD skill count."""
        if not jd_set:
            return 0.0
        score = sum(resume_map.get(skill, 0.0) for skill in jd_set)
        return min(score / len(jd_set), 1.0)

    def _jaccard(self, resume_skills: dict, jd_skills: dict) -> float:
        """Simple Jaccard similarity between canonical skill sets (used for comparison)."""
        r = {e["canonical"] for e in resume_skills.get("skills", [])}
        j = {e["canonical"] for e in jd_skills.get("skills", [])}
        if not r and not j:
            return 0.0
        return len(r & j) / len(r | j)

    def _experience_score(self, resume_years: float, jd_years: float) -> float:
        """
        Ratio of candidate experience to required experience.
        """
        try:
            if resume_years is None or jd_years is None:
                return 0.5
            r = float(resume_years)
            j = float(jd_years)
            if r != r or j != j:
                return 0.5
            if j == 0:
                return 1.0
            return min(r / j, 1.0)
        except (TypeError, ValueError):
            return 0.5

    def _english_level_score(self, resume_level: str, jd_level: str) -> float:
        """Score English level fit: resume level / JD required level, capped at 1.0.
        Returns 0.5 when either side is missing (neutral)."""
        r = ENGLISH_LEVELS.get(str(resume_level).lower().strip()) if resume_level else None
        j = ENGLISH_LEVELS.get(str(jd_level).lower().strip()) if jd_level else None
        if r is None or j is None:
            return 0.5
        if j == 0:
            return 1.0
        return min(r / j, 1.0)

    def _title_score(self, resume_position: str, jd_title: str) -> float:
        """
        Token overlap between resume position and JD title.
        Uses Jaccard
        """
        if not resume_position or not jd_title:
            return 0.0

        r_tokens = set(resume_position.lower().split())
        j_tokens = set(jd_title.lower().split())

        # Remove generic keywords
        noise = {'engineer', 'developer', 'senior', 'junior', 'lead', 'staff',
                 'the', 'and', 'for', 'with', 'a', 'an', 'of', 'in', 'at'}
        r_tokens -= noise
        j_tokens -= noise

        if not r_tokens or not j_tokens:
            return 0.0

        return len(r_tokens & j_tokens) / len(r_tokens | j_tokens)
