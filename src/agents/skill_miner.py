"""
Skill Mining Agent - extracts and normalises skills from resume/JD text.
Uses sentence-transformers + FAISS to map raw skill strings to canonical names.
"""
import re
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from sentence_transformers import SentenceTransformer
import faiss

from .base_agent import BaseAgent
from config import EMBEDDING_MODEL, SKILLS_TAXONOMY_FILE, EXPLICIT_SKILL_CONFIDENCE, LATENT_SKILL_CONFIDENCE
from schemas.models import SkillEntry, MinedSkills

# similarity thresholds
NORM_THRESHOLD = 0.82
LATENT_THRESHOLD = 0.76
MAX_LATENT = 20
# run latent mining if explicit skill count is below this
LATENT_TRIGGER = 15

# common abbreviations - expanded from original
ALIASES: Dict[str, str] = {
    'js': 'javascript',
    'ts': 'typescript',
    'py': 'python',
    'rb': 'ruby',
    'reactjs': 'react',
    'react.js': 'react',
    'react js': 'react',
    'nextjs': 'next.js',
    'next js': 'next.js',
    'nodejs': 'node.js',
    'node js': 'node.js',
    'vuejs': 'vue.js',
    'vue js': 'vue.js',
    'angularjs': 'angular',
    'k8s': 'kubernetes',
    'kube': 'kubernetes',
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
    'ci/cd': 'ci/cd',
    'ror': 'ruby on rails',
    'ef': 'entity framework',
    'net': '.net',
    'dotnet': '.net',
    'es6': 'javascript',
    'es2015': 'javascript',
    'es2016': 'javascript',
    'es2017': 'javascript',
    'mssql': 'sql server',
    'msql': 'sql server',
    'pgsql': 'postgresql',
    'nosql': 'nosql',
    'scss': 'sass',
    'expressjs': 'express.js',
    'express js': 'express.js',
    'nestjs': 'nest.js',
    'nuxtjs': 'nuxt.js',
    'springboot': 'spring boot',
    'spring boot': 'spring boot',
}

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

# capitalised tech phrases from text
TECH_PHRASE = re.compile(
    r'\b([A-Z][A-Za-z0-9#\.\+\-]{1,29}(?:\s+[A-Z][A-Za-z0-9#\.\+\-]{1,29}){0,2})\b'
)


class SkillMiningAgent(BaseAgent):

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("SkillMiningAgent", config)
        self._model: Optional[SentenceTransformer] = None
        self._taxonomy: Optional[pd.DataFrame] = None
        self._index: Optional[faiss.IndexFlatIP] = None
        self._skill_names: Optional[List[str]] = None
        self._skill_categories: Optional[Dict[str, str]] = None
        self._skill_freq: Optional[Dict[str, int]] = None
        self._max_freq: int = 1
        self._known_tech_pattern: Optional[re.Pattern] = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return

        logger.info("Loading embedding model...")
        self._model = SentenceTransformer(EMBEDDING_MODEL)

        logger.info("Loading skills taxonomy...")
        if not SKILLS_TAXONOMY_FILE.exists():
            raise FileNotFoundError(
                f"skills_master.csv not found at {SKILLS_TAXONOMY_FILE}. "
                "Run python3 src/preprocess/rebuild_taxonomy.py first."
            )
        df = pd.read_csv(SKILLS_TAXONOMY_FILE)
        df['skill'] = df['skill'].str.replace('[​‌‍﻿]', '', regex=True).str.strip()
        df = df[df['skill'].str.len() > 0].drop_duplicates('skill')

        self._skill_names = df['skill'].tolist()
        self._skill_categories = dict(zip(df['skill'], df['category']))

        # frequency data for confidence weighting
        self._skill_freq = dict(zip(df['skill'], df['total_freq']))
        self._max_freq = int(df['total_freq'].max())

        # build tech pattern dynamically from top technical skills in taxonomy
        top_tech = (
            df[df['category'] == 'technical']
            .nlargest(150, 'total_freq')['skill']
            .tolist()
        )
        pattern = '|'.join(re.escape(s) for s in top_tech if len(s) >= 2)
        self._known_tech_pattern = re.compile(r'\b(' + pattern + r')\b', re.IGNORECASE)

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
        raw_text   = input_data.get("raw_text", "")
        source = input_data.get("source", "resume")

        # normalize explicit skills and get their canonical embeddings in one pass
        explicit_entries, explicit_vecs = self._normalize_skills(
            raw_skills, confidence=EXPLICIT_SKILL_CONFIDENCE
        )

        latent_entries = []
        latent_vecs    = []
        if len(explicit_entries) < LATENT_TRIGGER and raw_text:
            already_found = {e.canonical for e in explicit_entries}
            candidates = self._extract_candidates_from_text(raw_text)
            latent_entries, latent_vecs = self._normalize_skills(
                candidates,
                confidence=LATENT_SKILL_CONFIDENCE,
                threshold=LATENT_THRESHOLD,
                exclude=already_found,
                use_freq_weight=True,
            )
            # filter low confidence latent skills
            filtered = [(e, v) for e, v in zip(latent_entries, latent_vecs) if e.confidence >= 0.65]
            latent_entries = [x[0] for x in filtered][:MAX_LATENT]
            latent_vecs    = [x[1] for x in filtered][:MAX_LATENT]

        all_entries = explicit_entries + latent_entries
        # reuse embeddings from normalisation - no second encode call needed
        skill_vectors = (explicit_vecs + latent_vecs) if all_entries else []

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

        return MinedSkills(
            skills=all_entries,
            grouped=grouped,
            skill_vectors=[v.tolist() for v in skill_vectors],
        ).model_dump()

    @staticmethod
    def _clean_skill(skill: str) -> str:
        # strip control characters (tabs, newlines etc)
        s = re.sub(r'[\x00-\x1f\x7f]', ' ', skill)
        # remove wrapping quotes and leading punctuation artifacts
        s = re.sub(r'^[\s"\'`!@#\$%\^&\*\(\)\[\]{}<>\\|/]+', '', s)
        s = re.sub(r'[\s"\'`!@#\$%\^&\*\(\)\[\]{}<>\\|/]+$', '', s)
        # collapse multiple spaces
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def _normalize_skills(
        self,
        skills: List[str],
        confidence: float = 1.0,
        threshold: float = NORM_THRESHOLD,
        exclude: set = None,
        use_freq_weight: bool = False,
    ) -> Tuple[List[SkillEntry], List[np.ndarray]]:
        if not skills:
            return [], []

        exclude = exclude or set()

        # clean all skills before encoding
        cleaned = [self._clean_skill(s) for s in skills]
        # pair cleaned with original for traceability, drop empty
        pairs = [(orig, c) for orig, c in zip(skills, cleaned) if c and len(c) >= 2]
        if not pairs:
            return [], []

        originals, cleaned_skills = zip(*pairs)

        embeddings = self._model.encode(
            list(cleaned_skills),
            batch_size=256,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

        # batch FAISS search - one call instead of N individual searches
        all_scores, all_indices = self._index.search(embeddings, 1)

        entries = []
        vecs    = []
        seen    = set()

        for i, (_, skill_clean) in enumerate(zip(originals, cleaned_skills)):
            if not skill_clean:
                continue

            # resolve alias without needing FAISS
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
                    # use the input embedding for aliases since they resolved without FAISS
                    vecs.append(embeddings[i])
                continue

            score     = float(all_scores[i][0])
            idx       = int(all_indices[i][0])
            canonical = self._skill_names[idx] if score >= threshold else skill_clean.lower()

            if canonical in exclude or canonical in seen:
                continue
            seen.add(canonical)

            # apply frequency weighting for latent skills only
            final_confidence = round(min(confidence * score, 1.0), 3)
            if use_freq_weight and canonical in self._skill_freq:
                freq        = self._skill_freq[canonical]
                freq_weight = math.log(freq + 1) / math.log(self._max_freq + 1)
                # gentle adjustment: common skills get small boost, rare ones small penalty
                final_confidence = round(min(final_confidence * (0.8 + 0.2 * freq_weight), 1.0), 3)

            entries.append(SkillEntry(
                skill=skill_clean,
                canonical=canonical,
                category=self._skill_categories.get(canonical, "technical"),
                confidence=final_confidence,
                source="explicit" if confidence == EXPLICIT_SKILL_CONFIDENCE else "latent",
            ))

            # retrieve canonical embedding directly from index - no re-encoding needed
            if score >= threshold:
                canonical_vec = self._index.reconstruct(idx)
            else:
                canonical_vec = embeddings[i]
            vecs.append(canonical_vec)

        return entries, vecs

    def _extract_candidates_from_text(self, text: str) -> List[str]:
        candidates = []

        # use taxonomy-driven pattern instead of hardcoded list
        for m in self._known_tech_pattern.finditer(text):
            candidates.append(m.group(0).lower())

        # also pick up capitalised multi-word tech phrases
        for m in TECH_PHRASE.finditer(text):
            token = m.group(1).strip()
            lower = token.lower()
            if (1 <= len(token.split()) <= 3
                    and len(lower) >= 3
                    and lower not in STOPWORDS
                    and not (token.islower() and not any(c in token for c in '.#+0123456789'))):
                candidates.append(lower)

        seen   = set()
        unique = []
        for c in candidates:
            if c in seen or c in STOPWORDS:
                continue
            if re.search(r'[.,!?;:]\s', c):
                continue
            words = c.split()
            if words[-1] in STOPWORDS or words[0] in STOPWORDS:
                continue
            seen.add(c)
            unique.append(c)

        return unique[:50]
