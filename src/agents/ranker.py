"""
Ranking & Explanation Agent
Takes a sorted list of MatchScore dicts and produces RankedResult objects
containing rank, matched skills, missing skills, and a plain-English explanation.
Matched/missing skill detection uses the same cosine similarity threshold (0.40)
as the Matching Agent so that explanations are consistent with scores.
"""
import numpy as np
from typing import Dict, Any, List, Tuple

from .base_agent import BaseAgent
from schemas.models import RankedResult

SEMANTIC_THRESHOLD = 0.40

class RankingExplanationAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("RankingExplanationAgent", config)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input keys:
            match_results list[dict]  — MatchScore dicts, pre-sorted by final_score desc
            resume_skills_map  dict   — {resume_id: MinedSkills dict}
            jd_skills dict  — MinedSkills dict for the JD
            jd_exp_years
            jd_education_req

        Returns:
            {"ranked": list[RankedResult dicts]}
        """
        if not self.validate_input(input_data, ["match_results", "resume_skills_map", "jd_skills"]):
            return {"error": "Missing required fields"}

        match_results     = input_data["match_results"]
        resume_skills_map = input_data["resume_skills_map"]
        jd_skills         = input_data["jd_skills"]
        jd_exp_years      = input_data.get("jd_exp_years")
        jd_education_req  = input_data.get("jd_education_req")

        jd_canonicals = [e["canonical"] for e in jd_skills.get("skills", [])]
        j_vecs_raw = jd_skills.get("skill_vectors", [])
        j_vecs = np.array(j_vecs_raw, dtype=np.float32) if j_vecs_raw else None

        ranked = []
        for rank, match in enumerate(match_results, start=1):
            resume_id  = match["resume_id"]
            resume_skills = resume_skills_map.get(resume_id, {})

            r_vecs_raw = resume_skills.get("skill_vectors", [])
            r_vecs = np.array(r_vecs_raw, dtype=np.float32) if r_vecs_raw else None

            matched, missing = self._find_matched_missing(
                jd_canonicals, j_vecs, resume_skills, r_vecs
            )

            explanation = self._build_explanation(
                match=match,
                matched=matched,
                missing=missing,
                jd_required_count=len(jd_canonicals),
                jd_exp_years=jd_exp_years,
                jd_education_req=jd_education_req,
            )

            ranked.append(RankedResult(
                rank=rank,
                resume_id=resume_id,
                final_score=match["final_score"],
                matched_skills=matched,
                missing_skills=missing,
                explanation=explanation,
            ).model_dump())

        self.log_metrics({
            "candidates_ranked": len(ranked),
            "avg_final_score": round(
                sum(r["final_score"] for r in ranked) / max(len(ranked), 1), 3
            ),
        })

        return {"ranked": ranked}

    def _find_matched_missing(
        self,
        jd_canonicals: List[str],
        j_vecs: "np.ndarray | None",
        resume_skills: dict,
        r_vecs: "np.ndarray | None",
    ) -> Tuple[List[str], List[str]]:
        """
        For each JD skill (cosine similarity >= SEMANTIC_THRESHOLD).
        Falls back to exact canonical name matching when vectors are absent.
        Returns (matched, missing) as sorted lists.
        """
        if j_vecs is not None and r_vecs is not None and r_vecs.shape[0] > 0:
            sim = r_vecs @ j_vecs.T       # [n_resume × n_jd]
            best_per_jd = sim.max(axis=0) # [n_jd] — best resume match per JD skill

            matched = sorted(
                jd_canonicals[k] for k in range(len(jd_canonicals))
                if best_per_jd[k] >= SEMANTIC_THRESHOLD
            )
            missing = sorted(
                jd_canonicals[k] for k in range(len(jd_canonicals))
                if best_per_jd[k] < SEMANTIC_THRESHOLD
            )
        else:
            resume_set = {e["canonical"] for e in resume_skills.get("skills", [])}
            jd_set = set(jd_canonicals)
            matched    = sorted(jd_set & resume_set)
            missing    = sorted(jd_set - resume_set)

        return matched, missing

    def _build_explanation(
        self,
        match: dict,
        matched: List[str],
        missing: List[str],
        jd_required_count: int,
        jd_exp_years,
        jd_education_req,
    ) -> str:
        """Plain-English explanation"""
        parts = []

        # Skills
        n_matched = len(matched)
        n_total   = jd_required_count or 1

        if n_matched == 0:
            parts.append("No required skills matched.")
        elif n_matched == n_total:
            skill_list = ", ".join(matched[:5]) + ("..." if n_matched > 5 else "")
            parts.append(f"All {n_matched} required skill{'s' if n_matched > 1 else ''} matched ({skill_list}).")
        else:
            skill_list = ", ".join(matched[:4]) + ("..." if n_matched > 4 else "")
            parts.append(f"Matched {n_matched}/{n_total} required skills ({skill_list}).")
            if missing:
                miss_list = ", ".join(missing[:3]) + ("..." if len(missing) > 3 else "")
                parts.append(f"Missing: {miss_list}.")

        # Experience
        exp_score = match.get("experience_score", 0.5)
        if jd_exp_years and jd_exp_years > 0:
            if exp_score >= 1.0:
                parts.append(f"Meets or exceeds the {jd_exp_years}-year experience requirement.")
            elif exp_score >= 0.75:
                estimated = round(exp_score * jd_exp_years, 1)
                parts.append(f"Slightly below the {jd_exp_years}-year requirement ({estimated} yrs estimated).")
            elif exp_score > 0:
                estimated = round(exp_score * jd_exp_years, 1)
                parts.append(f"Significantly under the {jd_exp_years}-year requirement ({estimated} yrs estimated).")
            else:
                parts.append("Experience information unavailable.")
        elif exp_score == 0.5:
            parts.append("Experience requirement not specified.")

        # Education
        edu_score = match.get("education_score", 0.5)
        if jd_education_req:
            if edu_score >= 1.0:
                parts.append(f"Education meets requirement ({jd_education_req}).")
            elif edu_score > 0:
                parts.append(f"Education below requirement ({jd_education_req}).")
            else:
                parts.append("Education not found in resume.")

        # Title
        title_score = match.get("title_score", 0.0)
        if title_score >= 0.5:
            parts.append("Strong title alignment.")
        elif title_score >= 0.2:
            parts.append("Partial title alignment.")

        return " ".join(parts)
