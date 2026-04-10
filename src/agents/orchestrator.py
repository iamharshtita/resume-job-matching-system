"""
Orchestrator — coordinates all agents in the pipeline.
"""
from typing import Dict, Any, List
from loguru import logger

from .base_agent import BaseAgent
from .resume_parser import ResumeParserAgent
from .jd_parser import JDParserAgent


class SkillMiningOrchestrator(BaseAgent):

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Orchestrator", config)
        self.resume_parser = ResumeParserAgent()
        self.jd_parser = JDParserAgent()
        # TODO: self.resume_skill_miner = ResumeSkillMinerAgent()
        # TODO: self.jd_skill_miner = JDSkillMinerAgent()
        # TODO: self.matcher = MatcherAgent()
        # TODO: self.ranker = InsightAgent()

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(
            resume_text=input_data["resume_text"],
            jd_text=input_data["jd_text"],
        )

    def run(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        logger.info("Starting pipeline...")

        parsed_resume = self.resume_parser.process({"raw_text": resume_text})
        logger.info("Resume parsed.")

        parsed_jd = self.jd_parser.process({"raw_text": jd_text})
        logger.info("JD parsed.")

        # TODO: resume_skills = self.resume_skill_miner.process({"parsed_resume": parsed_resume})
        # TODO: jd_skills = self.jd_skill_miner.process({"parsed_jd": parsed_jd})
        # TODO: match_scores = self.matcher.process({...})
        # TODO: result = self.ranker.process({"match_scores": [match_scores]})

        return {
            "parsed_resume": parsed_resume,
            "parsed_jd": parsed_jd,
            "final_score": 0.0,
            "explanation": "Not yet implemented",
        }

    def rank_all_candidates(
        self,
        _job_id: str,
        all_matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        # TODO: Implement ranking logic
        return sorted(all_matches, key=lambda x: x.get("final_score", 0), reverse=True)
