"""
Resume Parser Agent
Extracts skills, experience and education from raw CV text.
"""
import re
import datetime
from typing import Optional
import spacy
from loguru import logger
from .base_agent import BaseAgent
from schemas.models import ParsedResume, ExperienceBlock, EducationBlock

# Skill section header on its own line
_SKILL_SECTION_HDR = re.compile(
    r"^[*_\-]{0,2}\s*(skills?|tech(?:nologies?|nical\s*skills?)?|"
    r"stack|tools?|languages?|frameworks?|platforms?|expertise|"
    r"competenc(?:ies|e)|my\s+skills?|key\s+skills?|core\s+skills?|"
    r"technical\s+expertise|what\s+can\s+i\s+technically|"
    r"technologies?\s*(?:used)?|technical\s+stack)\s*[:\*_]{0,2}\s*$",
    re.IGNORECASE,
)

# Signals that a new (non-skill) section has started
_END_SECTION_HDR = re.compile(
    r"^[*_\-]{0,2}\s*(experience|education|work\s+experience|employment|"
    r"personal|contact|certification|project|achievement|"
    r"summary|objective|profile|reference|interest|award|"
    r"volunteer|publication|patent|study|work|about|"
    r"who\s+am\s+i|what\s+can\s+i\s+bring)\s*[:\*_]{0,2}\s*$",
    re.IGNORECASE,
)

# Inline skill header: "Stack: Python, React, ..."
_INLINE_SKILL_HDR = re.compile(
    r"\b(tools?|stack|tech(?:nologies)?|skills?|languages?|frameworks?|platforms?)"
    r"[^:\n]{0,30}:\s*(.+)",
    re.IGNORECASE,
)

_DEGREE = re.compile(
    r"\b(bachelor(?:'?s)?|master(?:'?s)?|b\.s\.|m\.s\.|ph\.?d\.?|mba|bsc|msc"
    r"|specialist|diploma|certificate|degree)\b",
    re.IGNORECASE,
)

_EDU_CONTEXT = re.compile(
    r"\b(university|college|school|institute|degree|studied|graduated|faculty"
    r"|course|program|major|minor|academia|studying)\b",
    re.IGNORECASE,
)

_YEAR = re.compile(r"\b(19[89]\d|20[012]\d)\b")

_YEAR_RANGE = re.compile(
    r"\b((?:19|20)\d{2})\s*[-–—]\s*((?:19|20)\d{2}|present|now|current)\b",
    re.IGNORECASE,
)

_TECH_TOKEN = re.compile(r"^[A-Za-z#\.\+][A-Za-z0-9#\.\+\-]{0,28}$")

_SKILL_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "have", "has",
    "been", "also", "more", "very", "some", "such", "into", "over",
    "other", "work", "team", "role", "year", "time", "high", "good",
    "able", "well", "will", "can", "our", "all", "any", "are", "was",
    "not", "but", "use", "used", "using", "based", "experience",
    "knowledge", "understanding", "ability", "strong", "excellent",
    "proficiency", "familiarity", "working", "plus", "years", "etc",
    "level", "basic", "advanced", "intermediate", "senior", "junior",
}


def _clean_skill_token(token: str) -> Optional[str]:
    """Clean and validate a skill token. Returns None if it doesn't look like a skill."""
    t = re.sub(r'\([^)]*\)', '', token)  # remove parentheticals like (Vue.js)
    t = t.strip(" .,;:-\r\n*_•")
    if not t or len(t) < 2 or len(t) > 50:
        return None
    if t.lower() in _SKILL_STOPWORDS:
        return None
    if not any(c.isalpha() for c in t):
        return None
    if len(t.split()) > 5:
        return None
    return t


def _looks_like_tech(token: str) -> bool:
    """Stricter check for tokens outside explicit skill sections (single-word only)."""
    t = token.strip(" .,;:-")
    if len(t) < 2 or len(t) > 30:
        return False
    if t.lower() in _SKILL_STOPWORDS:
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


def _extract_skills_from_section_lines(lines: list[str]) -> set[str]:
    """
    Lenient skill extraction from within an explicit skill section.
    Accepts multi-word skills and lowercase tokens.
    """
    skills: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("-", "*", "•", "·")):
            content = stripped.lstrip("-*•· ").strip()
            if "," in content or ";" in content:
                for token in re.split(r"[,;]", content):
                    tok = _clean_skill_token(token)
                    if tok:
                        skills.add(tok)
            else:
                tok = _clean_skill_token(content)
                if tok:
                    skills.add(tok)
        else:
            if "," in stripped or ";" in stripped:
                for token in re.split(r"[,;]", stripped):
                    tok = _clean_skill_token(token)
                    if tok:
                        skills.add(tok)
            else:
                tok = _clean_skill_token(stripped)
                if tok and len(stripped.split()) <= 4:
                    skills.add(tok)
    return skills


def _extract_skills(text: str) -> list[str]:
    skills: set[str] = set()
    lines = text.splitlines()
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()

        # Pass 1: Detect explicit skill section header on its own line
        if _SKILL_SECTION_HDR.match(stripped):
            section_lines = []
            blank_count = 0
            j = i + 1
            while j < len(lines):
                next_stripped = lines[j].strip()
                if _END_SECTION_HDR.match(next_stripped):
                    break
                if not next_stripped:
                    blank_count += 1
                    if blank_count >= 3:
                        break
                else:
                    blank_count = 0
                    section_lines.append(lines[j])
                j += 1
            if section_lines:
                skills.update(_extract_skills_from_section_lines(section_lines))
            i = j
            continue

        # Pass 2: Inline skill header (e.g. "Stack: Python, React, Node.js")
        m = _INLINE_SKILL_HDR.search(stripped)
        if m:
            for token in re.split(r"[,;/]", m.group(2)):
                tok = _clean_skill_token(token)
                if tok:
                    skills.add(tok)

        # Pass 3: Bullet lines with comma/semicolon-separated tech tokens
        if stripped.startswith(("-", "*", "•")):
            content = stripped.lstrip("-*• ").strip()
            if "," in content or ";" in content:
                for token in re.split(r"[,;]", content):
                    token = token.strip(" .\r\n")
                    if _looks_like_tech(token):
                        skills.add(token)

        i += 1

    return sorted(skills)


def _extract_education(text: str) -> list[EducationBlock]:
    blocks: list[EducationBlock] = []
    seen: set[str] = set()

    for line in text.splitlines():
        degree_match = _DEGREE.search(line)
        if not degree_match:
            continue

        has_context = bool(_EDU_CONTEXT.search(line))
        has_year = bool(_YEAR.search(line))
        # Accept if there's context, or if line is short enough to be an edu statement
        if not has_context and not has_year:
            if len(line.strip().split()) > 15:
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
    blocks: list[ExperienceBlock] = []
    seen: set[str] = set()
    lines = text.splitlines()

    # Pass 1: Year-range line followed by job title on next line
    # Format: "2011 - Present\nJob Title\n..."
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        yr_match = _YEAR_RANGE.search(stripped)
        if yr_match:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                title_line = lines[j].strip().lstrip("-*•· ").strip()
                if title_line and len(title_line) < 80 and title_line.count(" ") <= 6:
                    start_yr = int(yr_match.group(1))
                    end_str = yr_match.group(2).lower().strip()
                    if end_str in ("present", "now", "current"):
                        end_yr = datetime.date.today().year
                    else:
                        try:
                            end_yr = int(end_str)
                        except ValueError:
                            end_yr = start_yr + 1
                    duration = max(1, (end_yr - start_yr) * 12)
                    key = title_line[:60]
                    if key not in seen:
                        seen.add(key)
                        blocks.append(ExperienceBlock(
                            title=title_line,
                            company=None,
                            duration_months=duration,
                            description=stripped[:300],
                        ))
        i += 1

    # Pass 2: "Title at/@ Company" on a single line
    _JOB_LINE = re.compile(r"^(.{3,50}?)\s+(?:at|@|–|—)\s+(.{2,60})$", re.IGNORECASE)
    for line in lines:
        stripped = line.strip()
        m = _JOB_LINE.match(stripped)
        if not m:
            continue
        title = m.group(1).strip().lstrip("-*•· ").strip()
        company = re.sub(r"\s*\([\d\s\-–—]+\)\s*$", "", m.group(2)).strip()
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


class ResumeParserAgent(BaseAgent):
    """Parses raw CV text into a structured ParsedResume."""

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
            position=input_data.get("position"),
            experience_years=input_data.get("experience_years"),
            raw_skills=skills,
            experience=experience,
            education=education,
            raw_text=text,
        )
        return result.model_dump()
