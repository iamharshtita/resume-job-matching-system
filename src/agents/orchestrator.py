"""
Orchestrator — coordinates all agents in the full pipeline.

Single-pair:
    raw resume text + raw JD text
    ResumeParserAgent
    DParserAgent
    SkillMiningAgent  (resume side)
    SkillMiningAgent  (JD side, shared instance)
    MatchingAgent
    Final scoring

Batch(rank_candidates):
    one JD + list of (resume_id, raw_text)
    same pipeline per resume, JD skills mined once
    returns list of MatchScore dicts sorted by final_score desc
"""
import re
from typing import Dict, Any, List, Tuple
from loguru import logger

# Tokens that are not skill names (copied from resume_parser._SKILL_STOPWORDS subset)
_PHRASE_STOPWORDS = {
    "experience", "knowledge", "understanding", "ability", "strong", "excellent",
    "proficiency", "familiarity", "working", "with", "in", "of", "and", "or",
    "the", "a", "an", "for", "to", "using", "use", "used", "based", "good",
    "skills", "skill", "required", "preferred", "plus", "years", "etc",
    "concepts", "development", "optional", "solid", "basic", "advanced",
}


def _split_jd_skill_phrase(phrase: str) -> List[str]:
    """
    Break JD requirement text into individual skill tokens.
    e.g. "Python. Django experience" to ["Python", "Django"]
         "Docker and Kubernetes" to["Docker", "Kubernetes"]
         "Azure" to ["Azure"]
    """
    # Split on punctuation, conjunctions, and whitespace
    tokens = re.split(r"[.,;/\\|+]|\s+(?:and|or|with|&)\s+|\s{2,}", phrase)
    result = []
    for tok in tokens:
        tok = tok.strip(" .,;:-+")
        if not tok or len(tok) < 2:
            continue
        if tok.lower() in _PHRASE_STOPWORDS:
            continue
        # Drop pure-stopword multi-word tokens (e.g. "experience with")
        words = tok.split()
        meaningful = [w for w in words if w.lower() not in _PHRASE_STOPWORDS]
        if not meaningful:
            continue
        # left out part
        result.append(" ".join(meaningful) if len(meaningful) < len(words) else tok)
    return result if result else [phrase]


def _flatten_jd_skills(raw_skills: List[str]) -> List[str]:
    """Expand compound JD skill phrases into individual tokens."""
    out = []
    seen = set()
    for phrase in raw_skills:
        for tok in _split_jd_skill_phrase(phrase):
            key = tok.lower()
            if key not in seen:
                seen.add(key)
                out.append(tok)
    return out

from .base_agent import BaseAgent
from .resume_parser import ResumeParserAgent
from .jd_parser import JDParserAgent
from .skill_miner import SkillMiningAgent
from .matcher import MatchingAgent
from .ranker import RankingExplanationAgent


class SkillMiningOrchestrator(BaseAgent):

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Orchestrator", config)
        self.resume_parser = ResumeParserAgent()
        self.jd_parser     = JDParserAgent()
        self.skill_miner   = SkillMiningAgent()
        self.matcher       = MatchingAgent()
        self.ranker        = RankingExplanationAgent()

    # Single pair processing
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:
        resume_text
        jd_text
        resume_id
        jd_id
        resume_position
        experience_years
        """
        return self.run(
            resume_text=input_data["resume_text"],
            jd_text=input_data["jd_text"],
            resume_id=input_data.get("resume_id", "unknown"),
            jd_id=input_data.get("jd_id", "unknown"),
            resume_position=input_data.get("resume_position"),
            experience_years=input_data.get("experience_years"),
        )

    def run(
        self,
        resume_text: str,
        jd_text: str,
        resume_id: str = "unknown",
        jd_id: str = "unknown",
        resume_position: str = None,
        experience_years: float = None,
    ) -> Dict[str, Any]:

        logger.info(f"Pipeline start — resume={resume_id}  jd={jd_id}")

        # Step 1 — Parse
        parsed_resume = self.resume_parser.process({
            "raw_text": resume_text,
            "id": resume_id,
            "position": resume_position,
            "experience_years": experience_years,
        })
        parsed_jd = self.jd_parser.process({
            "raw_text": jd_text,
            "id": jd_id,
        })

        # Step 2 — Mine skills
        resume_skills = self.skill_miner.process({
            "raw_skills": parsed_resume["raw_skills"],
            "raw_text":   resume_text,
            "source":     "resume",
        })
        jd_skills = self.skill_miner.process({
            "raw_skills": _flatten_jd_skills(
                parsed_jd["required_skills"] + parsed_jd["preferred_skills"]
            ),
            "raw_text":   jd_text,
            "source":     "jd",
        })

        # Step 3 — Match
        match_result = self.matcher.process({
            "resume_id":        resume_id,
            "jd_id":            jd_id,
            "resume_skills":    resume_skills,
            "jd_skills":        jd_skills,
            "resume_exp_years":      parsed_resume.get("experience_years") or experience_years,
            "jd_exp_years":          parsed_jd.get("experience_years"),
            "resume_english_level":  parsed_resume.get("english_level"),
            "jd_english_level":      parsed_jd.get("english_level"),
            "resume_position":       parsed_resume.get("position") or resume_position or "",
            "jd_title":              parsed_jd.get("title", ""),
        })

        logger.info(
            f"Pipeline done — final_score={match_result.get('final_score')} "
            f"(skill={match_result.get('skill_score')}  "
            f"exp={match_result.get('experience_score')}  "
            f"english={match_result.get('english_level_score')}  "
            f"title={match_result.get('title_score')})"
        )

        return {
            "parsed_resume":       parsed_resume,
            "parsed_jd":           parsed_jd,
            "resume_skills":       resume_skills,
            "jd_skills":           jd_skills,
            "match":               match_result,
            "final_score":         match_result.get("final_score", 0.0),
            "skill_score":         match_result.get("skill_score", 0.0),
            "experience_score":    match_result.get("experience_score", 0.0),
            "english_level_score": match_result.get("english_level_score", 0.0),
            "title_score":         match_result.get("title_score", 0.0),
        }

    # Batch evaluation
    def rank_candidates(
        self,
        jd_text: str,
        candidates: List[Tuple[str, str]],   # list of (resume_id, resume_text)
        jd_id: str = "unknown",
    ) -> List[Dict[str, Any]]:
        """
        Mine JD skills once, then score every candidate against the same JD.
        """
        logger.info(f"Batch ranking: {len(candidates)} candidates vs jd={jd_id}")

        parsed_jd = self.jd_parser.process({"raw_text": jd_text, "id": jd_id})
        jd_skills = self.skill_miner.process({
            "raw_skills": _flatten_jd_skills(
                parsed_jd["required_skills"] + parsed_jd["preferred_skills"]
            ),
            "raw_text":   jd_text,
            "source":     "jd",
        })

        match_results      = []
        resume_skills_map  = {}

        for resume_id, resume_text in candidates:
            try:
                parsed_resume = self.resume_parser.process({
                    "raw_text": resume_text,
                    "id": resume_id,
                })
                resume_skills = self.skill_miner.process({
                    "raw_skills": parsed_resume["raw_skills"],
                    "raw_text":   resume_text,
                    "source":     "resume",
                })
                resume_skills_map[resume_id] = resume_skills

                match = self.matcher.process({
                    "resume_id":        resume_id,
                    "jd_id":            jd_id,
                    "resume_skills":    resume_skills,
                    "jd_skills":        jd_skills,
                    "resume_exp_years": parsed_resume.get("experience_years"),
                    "jd_exp_years":     parsed_jd.get("experience_years"),
                    "resume_english_level": parsed_resume.get("english_level"),
                    "jd_english_level":     parsed_jd.get("english_level"),
                    "resume_position":  parsed_resume.get("position", ""),
                    "jd_title":         parsed_jd.get("title", ""),
                })
                match_results.append(match)
            except Exception as e:
                logger.warning(f"Skipped resume {resume_id}: {e}")

        match_results.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

        ranked_output = self.ranker.process({
            "match_results":      match_results,
            "resume_skills_map":  resume_skills_map,
            "jd_skills":          jd_skills,
            "jd_exp_years":       parsed_jd.get("experience_years"),
            "jd_english_level":   parsed_jd.get("english_level"),
        })

        logger.info(f"Ranking complete — {len(match_results)} candidates scored")
        return ranked_output["ranked"]
