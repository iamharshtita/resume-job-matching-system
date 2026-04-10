"""
JD Parser Agent
Parses a raw JD into structured fields.
"""
import re
from typing import Optional
from .base_agent import BaseAgent
from schemas.models import ParsedJD

# Section headers
_REQUIRED_HEADERS = re.compile(
    r"^\*{0,2}\s*(requirements?|qualifications?|what we (expect|need|look for)"
    r"|must[ -]have|you (will|should) have|we (require|expect))\s*[:\*]*\s*$",
    re.IGNORECASE,
)

_PREFERRED_HEADERS = re.compile(
    r"^\*{0,2}\s*(nice[ -]to[ -]have|preferred|bonus|w(?:ould|ill) be (a )?plus"
    r"|advantageous|optional|good to have|desirable|would be nice"
    r"|plus(?:es)?|additional|nice\s+to\s+have)\s*[:\*]*\s*$",
    re.IGNORECASE,
)

_OTHER_HEADERS = re.compile(
    r"^\*{0,2}\s*(responsibilities|about (the )?(company|project|role|us|team)"
    r"|what you('ll| will) do|your role|benefits?|we offer|perks?|why (join|us))\s*[:\*]*\s*$",
    re.IGNORECASE,
)

# Words that look like section headers
_SECTION_WORDS = {
    "responsibilities", "requirements", "qualifications", "benefits",
    "responsibilities:", "requirements:", "qualifications:", "benefits:",
    "experience", "education", "about", "overview", "summary",
    "skills", "projects", "certifications", "languages",
}

# Inline lists skill headers
_SKILL_HEADER = re.compile(
    r"\b(tools?|stack|tech(?:nologies)?|skills?|languages?|frameworks?|platforms?|requirements?)"
    r"[^:\n]{0,30}:\s*(.+)",
    re.IGNORECASE,
)

# Experience years.
_EXP_YEARS = re.compile(r"(\d+)", )

# Education requirement
_EDU_DEGREE = re.compile(
    r"\b(b\.?sc?\.?|m\.?sc?\.?|ph\.?d\.?|mba|bachelor(?:'?s)?|master(?:'?s)?"
    r"|degree|diploma|certificate|university|college|computer\s*science"
    r"|related\s+field)\b",
    re.IGNORECASE,
)

# Tech stack token
_TECH_TOKEN = re.compile(r"^[A-Za-z#\.\+][A-Za-z0-9#\.\+\-]{0,28}$")

_PROSE_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "have", "has",
    "been", "also", "more", "very", "some", "such", "into", "over",
    "other", "work", "team", "role", "year", "time", "high", "good",
    "able", "well", "will", "can", "our", "all", "any", "are", "was",
    "not", "but", "use", "used", "using", "based", "include", "including",
    "experience", "knowledge", "understanding", "ability", "strong",
    "excellent", "proficiency", "familiarity", "working",
}

def _looks_like_tech(token: str) -> bool:
    t = token.strip(" .,;:-\r\n")
    if len(t) < 2 or len(t) > 30:
        return False
    if t.lower() in _PROSE_STOPWORDS:
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

_PROSE_SKILL = re.compile(
    r"\b(?:knowledge of|experience (?:with|in)|proficiency (?:in|with)"
    r"|familiar(?:ity)? with|working with|expertise in|skilled in)\s+"
    r"([A-Za-z][A-Za-z0-9#\.\+\-]*(?:\s+[A-Za-z][A-Za-z0-9#\.\+\-]*){0,2})",
    re.IGNORECASE,
)

def _extract_skills_from_lines(lines: list[str]) -> list[str]:
    """
    Four-pass skill extraction from a list of lines.

    Pass 1 — bullet lines with comma/semicolon separated tokens.
    Pass 2 — inline lists after a skill-header keyword (Stack: X, Y).
    Pass 3 — single short lines in requirements section (one skill per line,
              no bullet marker — common in Djinni JDs e.g. "Kyber Network").
    Pass 4 — prose patterns: "knowledge of X", "experience with X".
    """
    skills: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # bullet lines
        if stripped.startswith(("-", "*", "•", "·")):
            content = stripped.lstrip("-*•· ").strip()
            if "," in content or ";" in content:
                for token in re.split(r"[,;]", content):
                    token = token.strip(" .\r\n")
                    if _looks_like_tech(token):
                        skills.add(token)
            else:
                if _looks_like_tech(content):
                    skills.add(content)
            

        # inline skill list after a header
        m = _SKILL_HEADER.search(stripped)
        if m:
            for token in re.split(r"[,;/]", m.group(2)):
                token = token.strip(" .\r\n")
                if token and len(token) > 1:
                    skills.add(token)

        # short non-bullet line (skip lines already handled as bullets)
        words = stripped.split()
        if (
            not stripped.startswith(("-", "*", "•", "·"))
            and 1 <= len(words) <= 3
            and stripped.lower().rstrip(":") not in _SECTION_WORDS
        ):
            if _looks_like_tech(stripped):
                skills.add(stripped)
        for m in _PROSE_SKILL.finditer(stripped):
            token = m.group(1).strip(" .,;:-")
            if token and 2 <= len(token) <= 40 and token[0].isupper():
                skills.add(token)

    return sorted(skills)


def _extract_education_requirement(text: str) -> Optional[str]:
    """Return the first line that mentions an education requirement."""
    for line in text.splitlines():
        if _EDU_DEGREE.search(line):
            stripped = line.strip().lstrip("-*•· ").strip()
            if stripped and len(stripped) < 200:
                return stripped
    return None


def _parse_exp_years(exp_str: str):
    """
    Convert Exp Years string to an integer.
    Examples: "2y" → 2, "3-5y" → 3, "no exp" → 0, "10y+" → 10
    """
    if not exp_str or str(exp_str).lower() in ("nan", "none", "no exp", ""):
        return None
    nums = _EXP_YEARS.findall(str(exp_str))
    if nums:
        return int(nums[0])
    return None


def _split_sections(text: str) -> dict[str, list[str]]:
    """
    Split JD into named sections by detecting header.
    Returns dict: {"required": [], "preferred": [], "other": []}
    """
    sections = {"required": [], "preferred": [], "other": []}
    current = "other"
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _REQUIRED_HEADERS.match(stripped):
            current = "required"
        elif _PREFERRED_HEADERS.match(stripped):
            current = "preferred"
        elif _OTHER_HEADERS.match(stripped):
            current = "other"
        else:
            sections[current].append(line)
    return sections


class JDParserAgent(BaseAgent):
    """
    Parses raw job description into structured fields.
    """

    def __init__(self):
        super().__init__("JDParserAgent")

    def process(self, input_data: dict) -> dict:
        self.validate_input(input_data, ["raw_text"])

        text: str = input_data["raw_text"] or ""

        # Split sections
        sections = _split_sections(text)

        # No required section detected, treat the whole text as required
        if not sections["required"]:
            sections["required"] = text.splitlines()

        required_skills = _extract_skills_from_lines(sections["required"])
        preferred_skills = _extract_skills_from_lines(sections["preferred"])

        # overlap
        preferred_skills = [s for s in preferred_skills if s not in required_skills]

        exp_years = _parse_exp_years(input_data.get("exp_years_raw"))
        education_requirement = _extract_education_requirement(text)

        self.log_metrics({
            "required_skills": len(required_skills),
            "preferred_skills": len(preferred_skills),
            "exp_years": exp_years,
            "text_length": len(text),
        })

        result = ParsedJD(
            id=input_data.get("id"),
            title=input_data.get("position", ""),
            company=input_data.get("company"),
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            experience_years=exp_years,
            education_requirement=education_requirement,
            primary_keyword=input_data.get("primary_keyword"),
            raw_text=text,
        )

        return result.model_dump()