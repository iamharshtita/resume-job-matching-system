"""
Agents schema output
"""
from pydantic import BaseModel
from typing import Optional

# ResumeParser Output
class ExperienceBlock(BaseModel):
    title: Optional[str]
    company: Optional[str]
    duration_months: Optional[int]
    description: str

class EducationBlock(BaseModel):
    degree: str
    field: Optional[str]
    institution: Optional[str]
    year: Optional[int]

class ParsedResume(BaseModel):
    id: Optional[str]
    position: Optional[str]
    experience_years: Optional[float]
    raw_skills: list[str]
    experience: list[ExperienceBlock]
    education: list[EducationBlock]
    raw_text: str

# JDParser Output
class ParsedJD(BaseModel):
    id: Optional[str]
    title: str
    company: Optional[str]
    primary_keyword: Optional[str]
    required_skills: list[str]
    preferred_skills: list[str]
    experience_years: Optional[int]
    education_requirement: Optional[str]
    raw_text: str

# SkillMiner Output
class SkillEntry(BaseModel):
    skill: str
    canonical: str
    category: str
    confidence: float
    source: str

class MinedSkills(BaseModel):
    skills: list[SkillEntry]
    grouped: dict[str, list[str]]

# MatcherOutput
class MatchScore(BaseModel):
    resume_id: str
    jd_id: str
    jaccard_score: float
    semantic_score: float
    skill_score: float
    experience_score: float
    education_score: float
    title_score: float
    final_score: float

# RankerOutput
class RankedResult(BaseModel):
    rank: int
    resume_id: str
    final_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    explanation: str
