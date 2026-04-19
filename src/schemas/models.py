from typing import Optional
from pydantic import BaseModel

class ExperienceBlock(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    duration_months: Optional[int] = None
    description: Optional[str] = None

class EducationBlock(BaseModel):
    degree: str
    field: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[int] = None

class ParsedResume(BaseModel):
    id: Optional[str] = None
    position: Optional[str] = None
    experience_years: Optional[float] = None
    english_level: Optional[str] = None
    raw_skills: list[str] = []
    experience: list[ExperienceBlock] = []
    education: list[EducationBlock] = []
    raw_text: str = ""

class ParsedJD(BaseModel):
    id: Optional[str] = None
    title: str = ""
    company: Optional[str] = None
    primary_keyword: Optional[str] = None
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    experience_years: Optional[int] = None
    education_requirement: Optional[str] = None
    english_level: Optional[str] = None
    raw_text: str = ""

class SkillEntry(BaseModel):
    skill: str
    canonical: str
    category: str
    confidence: float
    source: str

class MinedSkills(BaseModel):
    skills: list[SkillEntry] = []
    grouped: dict[str, list[str]] = {}
    skill_vectors: list[list[float]] = []

class MatchScore(BaseModel):
    resume_id: str
    jd_id: str
    jaccard_score: float
    skill_score: float
    experience_score: float
    english_level_score: float
    title_score: float
    final_score: float

class RankedResult(BaseModel):
    rank: int
    resume_id: str
    final_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    explanation: str
