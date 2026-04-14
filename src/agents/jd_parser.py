"""
JD Parser Agent
Parses a raw JD into structured fields.
"""
import re
from typing import Optional
from .base_agent import BaseAgent
from schemas.models import ParsedJD

# Section headers
REQUIRED_HEADERS = re.compile(
    r"^\*{0,2}\s*(requirements?|qualifications?|what we (expect|need|look for)"
    r"|must[ -]have|you (will|should) have|we (require|expect))\s*[:\*]*\s*$",
    re.IGNORECASE,
)

PREFERRED_HEADERS = re.compile(
    r"^\*{0,2}\s*(nice[ -]to[ -]have|preferred|bonus|w(?:ould|ill) be (a )?plus"
    r"|advantageous|optional|good to have|desirable|would be nice"
    r"|plus(?:es)?|additional|nice\s+to\s+have)\s*[:\*]*\s*$",
    re.IGNORECASE,
)

OTHER_HEADERS = re.compile(
    r"^\*{0,2}\s*(responsibilities|about (the )?(company|project|role|us|team)"
    r"|what you('ll| will) do|your role|benefits?|we offer|perks?|why (join|us))\s*[:\*]*\s*$",
    re.IGNORECASE,
)

# Words that look like section headers
SECTION_WORDS = {
    "responsibilities", "requirements", "qualifications", "benefits",
    "responsibilities:", "requirements:", "qualifications:", "benefits:",
    "experience", "education", "about", "overview", "summary",
    "skills", "projects", "certifications", "languages",
}

# Inline lists skill headers
SKILL_HEADER = re.compile(
    r"\b(tools?|stack|tech(?:nologies)?|skills?|languages?|frameworks?|platforms?|requirements?)"
    r"[^:\n]{0,30}:\s*(.+)",
    re.IGNORECASE,
)

# Experience years.
EXP_YEARS = re.compile(r"(\d+)", )

# Education requirement
EDU_DEGREE = re.compile(
    r"\b(b\.?sc?\.?|m\.?sc?\.?|ph\.?d\.?|mba|bachelor(?:'?s)?|master(?:'?s)?"
    r"|degree|diploma|certificate|university|college|computer\s*science"
    r"|related\s+field)\b",
    re.IGNORECASE,
)

# Tech stack token
TECH_TOKEN = re.compile(r"^[A-Za-z#\.\+][A-Za-z0-9#\.\+\-]{0,28}$")

PROSE_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "have", "has",
    "been", "also", "more", "very", "some", "such", "into", "over",
    "other", "work", "team", "role", "year", "time", "high", "good",
    "able", "well", "will", "can", "our", "all", "any", "are", "was",
    "not", "but", "use", "used", "using", "based", "include", "including",
    "experience", "knowledge", "understanding", "ability", "strong",
    "excellent", "proficiency", "familiarity", "working",
}

def looks_like_tech(token: str) -> bool:
    t = token.strip(" .,;:-\r\n")
    if len(t) < 2 or len(t) > 30:
        return False
    if t.lower() in PROSE_STOPWORDS:
        return False
    if not TECH_TOKEN.match(t):
        return False
    if not any(c.isalpha() for c in t):
        return False
    has_digit = any(c.isdigit() for c in t)
    has_special = any(c in "#.+" for c in t)
    is_upper = t.isupper() and len(t) >= 2
    is_mixed = t[0].isupper() and any(c.islower() for c in t[1:])
    return has_digit or has_special or is_upper or is_mixed

PROSE_SKILL = re.compile(
    r"\b(?:knowledge of|experience (?:with|in)|proficiency (?:in|with)"
    r"|familiar(?:ity)? with|working with|expertise in|skilled in)\s+"
    r"([A-Za-z][A-Za-z0-9#\.\+\-]*(?:\s+[A-Za-z][A-Za-z0-9#\.\+\-]*){0,2})",
    re.IGNORECASE,
)

def extract_skills_from_lines(lines: list[str]) -> list[str]:
    """
    Four step skill extraction from a list of lines.
    bullet lines with comma/semicolon separated tokens.
    inline lists after a skill-header keyword (Stack: X, Y).
    single short lines in requirements section
    prose patterns: "knowledge of X", "experience with X".
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
                    if looks_like_tech(token):
                        skills.add(token)
            else:
                if looks_like_tech(content):
                    skills.add(content)
            

        # inline skill list after a header
        m = SKILL_HEADER.search(stripped)
        if m:
            for token in re.split(r"[,;/]", m.group(2)):
                token = token.strip(" .\r\n")
                if token and len(token) > 1:
                    skills.add(token)

        # non-bullet line
        words = stripped.split()
        if (
            not stripped.startswith(("-", "*", "•", "·"))
            and 1 <= len(words) <= 3
            and stripped.lower().rstrip(":") not in SECTION_WORDS
        ):
            if looks_like_tech(stripped):
                skills.add(stripped)
        for m in PROSE_SKILL.finditer(stripped):
            token = m.group(1).strip(" .,;:-")
            if token and 2 <= len(token) <= 40 and token[0].isupper():
                skills.add(token)

    return sorted(skills)


def extract_education_requirement(text: str) -> Optional[str]:
    """Return the first line that mentions an education requirement."""
    for line in text.splitlines():
        if EDU_DEGREE.search(line):
            stripped = line.strip().lstrip("-*•· ").strip()
            if stripped and len(stripped) < 200:
                return stripped
    return None


def parse_exp_years(exp_str: str):
    """
    Convert Exp Years string to an integer.
    "2y" → 2, "3-5y" → 3, "no exp" → 0, "10y+" → 10
    """
    if not exp_str or str(exp_str).lower() in ("nan", "none", "no exp", ""):
        return None
    nums = EXP_YEARS.findall(str(exp_str))
    if nums:
        return int(nums[0])
    return None


def split_sections(text: str) -> dict[str, list[str]]:
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
        if REQUIRED_HEADERS.match(stripped):
            current = "required"
        elif PREFERRED_HEADERS.match(stripped):
            current = "preferred"
        elif OTHER_HEADERS.match(stripped):
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
        sections = split_sections(text)

        # No required section detected, treat the whole text as required
        if not sections["required"]:
            sections["required"] = text.splitlines()

        required_skills = extract_skills_from_lines(sections["required"])
        preferred_skills = extract_skills_from_lines(sections["preferred"])

        # overlap
        preferred_skills = [s for s in preferred_skills if s not in required_skills]

        exp_years = parse_exp_years(input_data.get("exp_years_raw"))
        education_requirement = extract_education_requirement(text)

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