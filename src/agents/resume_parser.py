"""
Resume Parser Agent
It is used to extract skills,experience and education 
"""
import re
import datetime
from typing import Optional
import spacy
from loguru import logger
from .base_agent import BaseAgent
from schemas.models import ParsedResume, ExperienceBlock, EducationBlock

# Extract intrinsic skills
_SKILL_HEADER = re.compile(
    r"\b(tools?|stack|tech(?:nologies)?|skills?|languages?|frameworks?|platforms?)"
    r"[^:\n]{0,30}:\s*(.+)",
    re.IGNORECASE,
)

# extract degree
_DEGREE = re.compile(
    r"\b(bachelor(?:'?s)?|master(?:'?s)?|b\.s\.|m\.s\.|ph\.?d\.?|mba|bsc|msc"
    r"|specialist|diploma|certificate)\b",
    re.IGNORECASE,
)

# Education context extraction
_EDU_CONTEXT = re.compile(
    r"\b(university|college|school|institute|degree|studied|graduated|faculty"
    r"|course|program|major|minor|academia)\b",
    re.IGNORECASE,
)

# Year pattern
_YEAR = re.compile(r"\b(19[89]\d|20[012]\d)\b")

# Tokens in tech stacks
_TECH_TOKEN = re.compile(r"^[A-Za-z#\.\+][A-Za-z0-9#\.\+\-]{0,28}$")

def _looks_like_tech(token: str) -> bool:
    t = token.strip(" .,;:-")
    if len(t) < 2 or len(t) > 30:
        return False
    if not _TECH_TOKEN.match(t):
        return False
    if not any(c.isalpha() for c in t):
        return False
    has_digit = any(c.isdigit() for c in t)
    has_special = any(c in "#.+" for c in t)
    is_upper = t.isupper() and len(t) >= 2
    is_mixed = t[0].isupper() and any(c.islower() for c in t[1:])
    return has_digit or has_special or is_upper or is_mixed


def _extract_skills(text: str) -> list[str]:
    """
    bullet lines
    inline lists
    """
    skills: set[str] = set()

    for line in text.splitlines():
        stripped = line.strip()

        # Bullet list
        if stripped.startswith(("-", "*", "•")):
            content = stripped.lstrip("-*• ").strip()
            # Check for ; or ,
            if "," in content or ";" in content:
                for token in re.split(r"[,;]", content):
                    token = token.strip(" .\r\n")
                    if _looks_like_tech(token):
                        skills.add(token)

        # Skill detail extraction
        m = _SKILL_HEADER.search(stripped)
        if m:
            for token in re.split(r"[,;/]", m.group(2)):
                token = token.strip(" .\r\n")
                if token:
                    skills.add(token)

    return sorted(skills)


def _extract_education(text: str) -> list[EducationBlock]:
    """
    Find lines with a degree keyword. Accept only if the same line also
    contains an education word or 4 digit year
    """
    blocks: list[EducationBlock] = []
    seen: set[str] = set()

    for line in text.splitlines():
        degree_match = _DEGREE.search(line)
        if not degree_match:
            continue

        has_context = bool(_EDU_CONTEXT.search(line))
        has_year = bool(_YEAR.search(line))
        if not has_context and not has_year:
            continue

        key = line.strip()[:80]
        if key in seen:
            continue
        seen.add(key)

        year_match = _YEAR.search(line)
        blocks.append(EducationBlock(
            degree=degree_match.group(0).capitalize(),
            field=None,
            institution=None,
            year=int(year_match.group(0)) if year_match else None,
        ))

    return blocks


def _extract_experience_blocks(text: str) -> list[ExperienceBlock]:
    """
    Scan lines for 'Title at/@ Company' patterns.
    """
    _JOB_LINE = re.compile(
        r"^(.{3,50}?)\s+(?:at|@|–|—)\s+(.{2,60})$",
        re.IGNORECASE,
    )
    blocks: list[ExperienceBlock] = []
    seen: set[str] = set()

    for line in text.splitlines():
        stripped = line.strip()
        m = _JOB_LINE.match(stripped)
        if not m:
            continue

        title = m.group(1).strip().lstrip("-*•· ").strip()
        company = re.sub(r"\s*\([\d\s\-–—]+\)\s*$", "", m.group(2)).strip()

        # Skip if either part is a long prose phrase
        if title.count(" ") > 4 or company.count(" ") > 4:
            continue

        key = f"{title}|{company}"
        if key in seen:
            continue
        seen.add(key)

        year_match = _YEAR.search(stripped)
        duration_months = None
        if year_match:
            year = int(year_match.group(0))
            duration_months = max(1, (datetime.date.today().year - year) * 12)

        blocks.append(ExperienceBlock(
            title=title,
            company=company,
            duration_months=duration_months,
            description=stripped[:300],
        ))

    return blocks


# Agent ResumeParser
class ResumeParserAgent(BaseAgent):
    """
    Parses raw CV text into a structured ParsedResume.
    """
    def __init__(self):
        super().__init__("ResumeParserAgent")
        logger.info("Loading spaCy model")
        self._nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model ready.")

    def process(self, input_data: dict) -> dict:
        self.validate_input(input_data, ["raw_text"])

        text: str = input_data["raw_text"] or ""

        skills = _extract_skills(text)
        education = _extract_education(text)
        experience = _extract_experience_blocks(text)

        self.log_metrics({
            "skills_found": len(skills),
            "experience_blocks": len(experience),
            "education_blocks": len(education),
            "text_length": len(text),
        })

        result = ParsedResume(
            id=input_data.get("id"),
            name=None,
            email=None,
            phone=None,
            position=input_data.get("position"),
            experience_years=input_data.get("experience_years"),
            raw_skills=skills,
            experience=experience,
            education=education,
            raw_text=text,
        )

        return result.model_dump()