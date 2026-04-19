import re
import datetime
from typing import Optional


from .base_agent import BaseAgent
from schemas.models import ParsedResume, ExperienceBlock, EducationBlock


# Detects a skills section header
SKILL_SECTION_HDR = re.compile(
    r"^[*_\-]{0,2}\s*(skills?|tech(?:nologies?|nical\s*skills?)?|"
    r"stack|tools?|languages?|frameworks?|platforms?|expertise|"
    r"competenc(?:ies|e)|my\s+skills?|key\s+skills?|core\s+skills?|"
    r"technical\s+expertise|what\s+can\s+i\s+technically|"
    r"technologies?\s*(?:used)?|technical\s+stack)\s*[:\*_]{0,2}\s*$",
    re.IGNORECASE,
)

# Detects the start of a non-skill section so we know when to stop collecting skills
END_SECTION_HDR = re.compile(
    r"^[*_\-]{0,2}\s*(experience|education|work\s+experience|employment|"
    r"personal|contact|certification|project|achievement|"
    r"summary|objective|profile|reference|interest|award|"
    r"volunteer|publication|patent|study|work|about|"
    r"who\s+am\s+i|what\s+can\s+i\s+bring)\s*[:\*_]{0,2}\s*$",
    re.IGNORECASE,
)

# Detects inline skill lists like "Stack: Python, React, Node.js"
INLINE_SKILL_HDR = re.compile(
    r"\b(tools?|stack|tech(?:nologies)?|skills?|languages?|frameworks?|platforms?)"
    r"[^:\n]{0,30}:\s*(.+)",
    re.IGNORECASE,
)

# Degree keywords — used to identify education lines
DEGREE = re.compile(
    r"\b(bachelor(?:'?s)?|master(?:'?s)?|b\.s\.|m\.s\.|ph\.?d\.?|mba|bsc|msc"
    r"|specialist|diploma|certificate|degree"
    r"|higher\s+education|incomplete\s+higher|second\s+higher"
    r"|bachelor\s+of|master\s+of|doctor\s+of)\b",
    re.IGNORECASE,
)

# University/institute keywords
UNI_LINE = re.compile(
    r"\b(university|universit[eéè]|universidad|univerza"
    r"|institute|institut|polytechnic|academy|akademi"
    r"|college|facult(?:y|é)|school\s+of)\b",
    re.IGNORECASE,
)

# education related lines
EDU_CONTEXT = re.compile(
    r"\b(university|college|school|institute|degree|studied|graduated|faculty"
    r"|course|program|major|minor|academia|studying)\b",
    re.IGNORECASE,
)

# Matches a 4-digit year (1980–2029)
YEAR = re.compile(r"\b(19[89]\d|20[012]\d)\b")

# Matches a year range like "2018 - 2021" or "2019 – Present"
YEAR_RANGE = re.compile(
    r"\b((?:19|20)\d{2})\s*[-–—]\s*((?:19|20)\d{2}|present|now|current)\b",
    re.IGNORECASE,
)

# A ech name (used outside skill sections)
TECH_TOKEN = re.compile(r"^[A-Za-z#\.\+][A-Za-z0-9#\.\+\-]{0,28}$")

# Generic words
SKILL_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "have", "has",
    "been", "also", "more", "very", "some", "such", "into", "over",
    "other", "work", "team", "role", "year", "time", "high", "good",
    "able", "well", "will", "can", "our", "all", "any", "are", "was",
    "not", "but", "use", "used", "using", "based", "experience",
    "knowledge", "understanding", "ability", "strong", "excellent",
    "proficiency", "familiarity", "working", "plus", "years", "etc",
    "level", "basic", "advanced", "intermediate", "senior", "junior",
}

# Final filter applied after extraction — removes geography, natural languages,
# and seniority labels that slipped through
SKILL_NOISE = {
    "ukraine", "kyiv", "odesa", "lviv", "kharkiv", "chernivtsi", "dnipro", "zaporizhzhia",
    "poland", "germany", "france", "usa", "canada", "uk", "remote", "worldwide",
    "english", "ukrainian", "russian", "polish", "german", "french", "spanish",
    "present", "currently", "now", "today", "full-time", "part-time", "freelance",
    "senior", "junior", "middle", "lead", "intern", "staff",
}


def clean_skill_token(token: str) -> Optional[str]:
    """Strip punctuation and validate a skill string.
    Returns None if the token is too short, too long, or a stopword."""
    t = re.sub(r'\([^)]*\)', '', token)   # drop parentheticals e.g. "(Vue.js)"
    t = t.strip(" .,;:-\r\n*_•")
    if not t or len(t) < 2 or len(t) > 50:
        return None
    if t.lower() in SKILL_STOPWORDS:
        return None
    if not any(c.isalpha() for c in t):
        return None
    if len(t.split()) > 5:
        return None
    return t


def looks_like_tech(token: str) -> bool:
    """Stricter check used outside explicit skill sections."""
    t = token.strip(" .,;:-")
    if len(t) < 2 or len(t) > 30:
        return False
    if t.lower() in SKILL_STOPWORDS:
        return False
    if not TECH_TOKEN.match(t):
        return False
    if not any(c.isalpha() for c in t):
        return False
    has_digit   = any(c.isdigit() for c in t)
    has_special = any(c in "#.+" for c in t)
    is_upper    = t.isupper() and len(t) >= 2
    is_mixed    = t[0].isupper() and any(c.islower() for c in t[1:])
    return has_digit or has_special or is_upper or is_mixed


def extract_skills_from_section_lines(lines: list) -> set:
    """Extraction once we are inside a skill section."""
    skills = set()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("-", "*", "•", "·")):
            content = stripped.lstrip("-*•· ").strip()
            if "," in content or ";" in content:
                for token in re.split(r"[,;]", content):
                    tok = clean_skill_token(token)
                    if tok:
                        skills.add(tok)
            else:
                tok = clean_skill_token(content)
                if tok:
                    skills.add(tok)
        else:
            if "," in stripped or ";" in stripped:
                for token in re.split(r"[,;]", stripped):
                    tok = clean_skill_token(token)
                    if tok:
                        skills.add(tok)
            else:
                tok = clean_skill_token(stripped)
                if tok and len(stripped.split()) <= 4:
                    skills.add(tok)
    return skills


def extract_skills(text: str) -> list:
    """Main skill extractor. Three passes over the CV text:
    finds an explicit skill section header, collects everything
    until the next non-skill section starts.
    picks up inline skill lists like "Stack: Python, React, Node.js".
    grabs tech-looking tokens from bullet lines with commas.
    """
    skills = set()
    lines = text.splitlines()
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()

        if SKILL_SECTION_HDR.match(stripped):
            section_lines = []
            blank_count = 0
            j = i + 1
            while j < len(lines):
                next_stripped = lines[j].strip()
                if END_SECTION_HDR.match(next_stripped):
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
                skills.update(extract_skills_from_section_lines(section_lines))
            i = j
            continue

        m = INLINE_SKILL_HDR.search(stripped)
        if m:
            for token in re.split(r"[,;/]", m.group(2)):
                tok = clean_skill_token(token)
                if tok:
                    skills.add(tok)

        if stripped.startswith(("-", "*", "•")):
            content = stripped.lstrip("-*• ").strip()
            if "," in content or ";" in content:
                for token in re.split(r"[,;]", content):
                    token = token.strip(" .\r\n")
                    if looks_like_tech(token):
                        skills.add(token)

        i += 1

    return sorted(skills)


def extract_education(text: str) -> list:
    """Two-pass education extractor.
    finds lines that contain a degree keyword alongside a year or
    educational word.
    finds a university name on its own line and looks within a 3-line
    window for a nearby degree string.
    """
    blocks = []
    seen = set()
    lines = text.splitlines()

    def add(degree, year_str, key):
        norm_key = degree.lower().split()[0] + "|" + str(year_str or "")
        if norm_key in seen:
            return
        seen.add(norm_key)
        seen.add(key[:80])
        blocks.append(EducationBlock(
            degree=degree.capitalize(),
            year=int(year_str) if year_str else None,
        ))

    for line in lines:
        dm = DEGREE.search(line)
        if not dm:
            continue
        has_context = bool(EDU_CONTEXT.search(line))
        has_year = bool(YEAR.search(line))
        if not has_context and not has_year and len(line.strip().split()) > 15:
            continue
        yr = YEAR.search(line)
        add(dm.group(0), yr.group(0) if yr else None, line.strip())

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not UNI_LINE.search(stripped):
            continue
        window = lines[max(0, i - 2): i + 4]
        for nearby in window:
            if nearby is line:
                continue
            dm = DEGREE.search(nearby)
            if not dm:
                continue
            yr = YEAR.search(nearby) or YEAR.search(line)
            key = stripped + "|" + nearby.strip()
            add(dm.group(0), yr.group(0) if yr else None, key)
            break

    return blocks


def extract_experience_blocks(text: str) -> list:
    """Two-pass experience extractor.
    looks for a year range on one line ("2018 – Present") with a
    job title on the next line.
    looks for "Title at Company".
    """
    blocks = []
    seen = set()
    lines = text.splitlines()

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        yr_match = YEAR_RANGE.search(stripped)
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
                            duration_months=duration,
                            description=stripped[:300],
                        ))
        i += 1

    job_line = re.compile(r"^(.{3,50}?)\s+(?:at|@|–|—)\s+(.{2,60})$", re.IGNORECASE)
    for line in lines:
        stripped = line.strip()
        m = job_line.match(stripped)
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
        year_match = YEAR.search(stripped)
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

"""Resume Parser agent trigger"""
class ResumeParserAgent(BaseAgent):

    def __init__(self):
        super().__init__("ResumeParserAgent")

    def process(self, input_data: dict) -> dict:
        self.validate_input(input_data, ["raw_text"])
        text = input_data.get("raw_text") or ""

        skills = [s for s in extract_skills(text) if s.lower() not in SKILL_NOISE]
        education = extract_education(text)
        experience = extract_experience_blocks(text)

        self.log_metrics({
            "skills_found": len(skills),
            "experience_blocks": len(experience),
            "education_blocks": len(education),
            "text_length": len(text),
        })

        return ParsedResume(
            id=input_data.get("id"),
            position=input_data.get("position"),
            experience_years=input_data.get("experience_years"),
            english_level=input_data.get("english_level"),
            raw_skills=skills,
            experience=experience,
            education=education,
            raw_text=text,
        ).model_dump()
