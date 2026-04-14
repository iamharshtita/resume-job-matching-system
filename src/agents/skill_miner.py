"""
Skill Mining Agent extracts latent skills from unstructured CV prose.
"""
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from loguru import logger
from sentence_transformers import SentenceTransformer
import faiss

from .base_agent import BaseAgent
from config import EMBEDDING_MODEL, SKILLS_TAXONOMY_FILE, EXPLICIT_SKILL_CONFIDENCE, LATENT_SKILL_CONFIDENCE
from schemas.models import SkillEntry, MinedSkills

# Similarity thresholds
# for explicit skills
NORM_THRESHOLD = 0.82
# implicit skills
LATENT_THRESHOLD = 0.76
# Number if skills extracted allowed
MAX_LATENT = 20

# Known tech keywords
KNOWN_TECH_LOWER = re.compile(
    r'\b(python|java|javascript|typescript|react|angular|vue|node\.?js|django|flask|'
    r'spring|docker|kubernetes|aws|azure|gcp|sql|mysql|postgresql|mongodb|redis|'
    r'git|linux|html|css|php|ruby|scala|golang|rust|c\+\+|c#|\.net|swift|kotlin|'
    r'tensorflow|pytorch|pandas|numpy|spark|hadoop|kafka|graphql|rest|api|'
    r'figma|unity|flutter|android|ios|tableau|powerbi|jenkins|terraform|ansible)\b',
    re.IGNORECASE,
)

# Regex to pull capitalised multi-word phrases that look like tech names
TECH_PHRASE = re.compile(
    r'\b([A-Z][A-Za-z0-9#\.\+\-]{1,29}(?:\s+[A-Z][A-Za-z0-9#\.\+\-]{1,29}){0,2})\b'
)

# known abbreviations
ALIASES: Dict[str, str] = {
    'js': 'javascript',
    'ts': 'typescript',
    'py': 'python',
    'reactjs': 'react',
    'react.js': 'react',
    'react js': 'react',
    'nodejs': 'node.js',
    'node js': 'node.js',
    'vuejs': 'vue.js',
    'vue js': 'vue.js',
    'angularjs': 'angular',
    'k8s': 'kubernetes',
    'tf': 'tensorflow',
    'pg': 'postgresql',
    'postgres': 'postgresql',
    'mongo': 'mongodb',
    'gcp': 'google cloud platform',
    'ml': 'machine learning',
    'dl': 'deep learning',
    'nlp': 'natural language processing',
    'cv': 'computer vision',
    'oop': 'object-oriented programming',
    'cicd': 'ci/cd',
    'ror': 'ruby on rails',
    'ef': 'entity framework',
}

# Stopwords to avoid
STOPWORDS = {
    'the','and','for','with','from','this','that','have','been','also','more',
    'very','some','such','into','over','other','work','team','role','year',
    'time','high','good','able','well','will','can','our','all','any','are',
    'was','not','but','use','used','using','based','experience','knowledge',
    'understanding','ability','strong','excellent','proficiency','familiarity',
    'familiar','working','make','build','built','create','develop','implement',
    'design','support','manage','lead','join','help','provide','ensure',
    'maintain','learn','learned','learning','developing','developed',
    'first','second','third','last','next','new','old','current','previous',
    'sometimes','generally','usually','often','always','never','mostly',
    'coordinate','analyze','promote','actively','provided',
    'so','my','me','we','he','she','it','do','go','be','at','in','on','up',
    'engineer','developer','architect','analyst','manager','specialist',
    'consultant','designer','administrator','lead','senior','junior','middle',
    'intern','staff','principal','director','head','chief','officer',
    'backend','frontend','fullstack','full-stack','full stack',
    'ukraine','kyiv','odesa','chernivtsi','lviv','kharkiv',
    'poland','germany','europe','usa','canada','remote','english','ukrainian',
    'january','february','march','april','may','june','july','august',
    'september','october','november','december',
    'monday','tuesday','wednesday','thursday','friday','saturday','sunday',
    'present','today','now','currently',
    'quality','preparation','keeping','rendering','approval','processing',
    'development','execution','planning','reporting','documentation',
    'implementation','integration','testing','deployment','maintenance',
}

#Skill mining agent
class SkillMiningAgent(BaseAgent):

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("SkillMiningAgent", config)
        self._model: Optional[SentenceTransformer] = None
        self._taxonomy: Optional[pd.DataFrame] = None
        self._index: Optional[faiss.IndexFlatIP] = None
        self._skill_names: Optional[List[str]] = None
        self._skill_categories: Optional[Dict[str, str]] = None
        self._loaded = False

    def _load(self):
        """Lazy-load the embedding model and build the FAISS index."""
        if self._loaded:
            return

        logger.info("Loading embedding model...")
        self._model = SentenceTransformer(EMBEDDING_MODEL)

        logger.info("Loading skills taxonomy...")
        df = pd.read_csv(SKILLS_TAXONOMY_FILE)
        df['skill'] = df['skill'].str.replace(r'[\u200b\u200c\u200d\ufeff]', '', regex=True).str.strip()
        df = df[df['skill'].str.len() > 0].drop_duplicates('skill')
        self._skill_names = df['skill'].tolist()
        self._skill_categories = dict(zip(df['skill'], df['category']))

        logger.info(f"Building FAISS index over {len(self._skill_names)} skills...")
        embeddings = self._model.encode(
            self._skill_names,
            batch_size=512,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

        self._loaded = True
        logger.info("SkillMiningAgent ready.")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_input(input_data, ["raw_skills"]):
            return {"error": "Missing required fields"}

        self._load()

        raw_skills = input_data.get("raw_skills", [])
        raw_text = input_data.get("raw_text", "")
        source = input_data.get("source", "resume")

        explicit_entries = self._normalize_skills(raw_skills, confidence=EXPLICIT_SKILL_CONFIDENCE)

        # Run latent extraction if needed.
        latent_entries = []
        if len(explicit_entries) < 10 and raw_text:
            already_found = {e.canonical for e in explicit_entries}
            candidates = self._extract_candidates_from_text(raw_text)
            latent_entries = [
                e for e in self._normalize_skills(
                    candidates,
                    confidence=LATENT_SKILL_CONFIDENCE,
                    threshold=LATENT_THRESHOLD,
                    exclude=already_found,
                )
                if e.confidence >= 0.65
            ][:MAX_LATENT]

        all_entries = explicit_entries + latent_entries

        grouped: Dict[str, List[str]] = {"technical": [], "soft": [], "domain": []}
        seen = set()
        for entry in all_entries:
            if entry.canonical not in seen:
                seen.add(entry.canonical)
                grouped.setdefault(entry.category, []).append(entry.canonical)

        self.log_metrics({
            "explicit_skills": len(explicit_entries),
            "latent_skills": len(latent_entries),
            "total_skills": len(all_entries),
            "source": source,
        })

        # Encode canonical names
        skill_vectors: List[List[float]] = []
        if all_entries:
            canonical_names = [e.canonical for e in all_entries]
            vecs = self._model.encode(
                canonical_names,
                batch_size=256,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).astype(np.float32)
            skill_vectors = vecs.tolist()

        return MinedSkills(
            skills=all_entries,
            grouped=grouped,
            skill_vectors=skill_vectors,
        ).model_dump()

    def _normalize_skills(
        self,
        skills: List[str],
        confidence: float = 1.0,
        threshold: float = NORM_THRESHOLD,
        exclude: set = None,
    ) -> List[SkillEntry]:
        if not skills:
            return []
        exclude = exclude or set()

        embeddings = self._model.encode(
            skills,
            batch_size=256,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

        entries = []
        seen = set()

        for skill, emb in zip(skills, embeddings):
            skill_clean = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', skill).strip()
            if not skill_clean:
                continue

            # resolve known abbreviations without needing embeddings
            alias = ALIASES.get(skill_clean.lower())
            if alias:
                if alias not in exclude and alias not in seen:
                    seen.add(alias)
                    entries.append(SkillEntry(
                        skill=skill_clean,
                        canonical=alias,
                        category=self._skill_categories.get(alias, "technical"),
                        confidence=round(confidence, 3),
                        source="explicit" if confidence == EXPLICIT_SKILL_CONFIDENCE else "latent",
                    ))
                continue

            scores, indices = self._index.search(emb.reshape(1, -1), 1)
            score = float(scores[0][0])
            canonical = self._skill_names[int(indices[0][0])] if score >= threshold else skill_clean.lower()

            if canonical in exclude or canonical in seen:
                continue
            seen.add(canonical)

            entries.append(SkillEntry(
                skill=skill_clean,
                canonical=canonical,
                category=self._skill_categories.get(canonical, "technical"),
                confidence=round(min(confidence * score, 1.0), 3),
                source="explicit" if confidence == EXPLICIT_SKILL_CONFIDENCE else "latent",
            ))

        return entries

    def _extract_candidates_from_text(self, text: str) -> List[str]:
        candidates = []

        for m in KNOWN_TECH_LOWER.finditer(text):
            candidates.append(m.group(0).lower())

        for m in TECH_PHRASE.finditer(text):
            token = m.group(1).strip()
            lower = token.lower()
            if (1 <= len(token.split()) <= 3
                and len(lower) >= 3
                and lower not in STOPWORDS
                and not (token.islower() and not any(c in token for c in '.#+0123456789'))):
                candidates.append(lower)

        seen = set()
        unique = []
        for c in candidates:
            if c in seen or c in STOPWORDS:
                continue
            if re.search(r'[.,!?;:]\s', c):
                continue
            # e.g. "Senior Backend Engineer" where last word "engineer" is a stopword
            words = c.split()
            if words[-1] in STOPWORDS or words[0] in STOPWORDS:
                continue
            seen.add(c)
            unique.append(c)

        return unique[:50]
