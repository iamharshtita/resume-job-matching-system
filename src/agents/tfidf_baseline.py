"""
TF-IDF Baseline Agent

Classic information-retrieval baseline: fit a TF-IDF vectorizer on a corpus
of resume + JD texts, then score any resume–JD pair as the cosine similarity
of their TF-IDF vectors.

This is the primary baseline required by the lit review (Salton 1988) against
which the multi-agent semantic scoring system is compared.

Usage
-----
Fit once on the full processed corpus, then call score_pair() or score_batch()
as many times as needed.  The fitted vectorizer is cached in memory.

    agent = TFIDFBaselineAgent()
    agent.fit_from_parquet(resumes_path, jds_path)
    score = agent.score_pair(resume_text, jd_text)
    results = agent.score_batch(resume_jd_pairs)
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base_agent import BaseAgent


class TFIDFBaselineAgent(BaseAgent):

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("TFIDFBaseline", config)
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._fitted = False

    # Fit
    def fit(self, corpus: List[str]) -> None:
        """
        Fit the TF-IDF vectorizer on a corpus of texts.
        Typical corpus = all resume raw_texts + all JD raw_texts.
        """
        logger.info(f"Fitting TF-IDF on {len(corpus)} documents…")
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b[A-Za-z][\w\.\+#\-]{1,}\b",  # keeps C#, C++, .NET
            ngram_range=(1, 2),     # unigrams + bigrams catch "machine learning" etc.
            min_df=3,               # ignore terms in < 3 docs
            max_df=0.90,            # ignore terms in > 90% of docs (stop-words)
            sublinear_tf=True,      # log(1 + tf) dampens very frequent terms
            max_features=50_000,
        )
        self._vectorizer.fit(corpus)
        self._fitted = True
        vocab_size = len(self._vectorizer.vocabulary_)
        logger.info(f"TF-IDF ready — vocabulary size: {vocab_size:,}")

    def fit_from_parquet(self, resumes_path, jds_path) -> None:
        """Convenience: load raw_text from both parquets and fit."""
        import pandas as pd
        logger.info("Loading corpora from parquet…")
        r_texts = pd.read_parquet(resumes_path, columns=["raw_text"])["raw_text"].fillna("").tolist()
        j_texts = pd.read_parquet(jds_path,     columns=["raw_text"])["raw_text"].fillna("").tolist()
        self.fit(r_texts + j_texts)

    # Scoring

    def score_pair(self, resume_text: str, jd_text: str) -> float:
        """Return cosine similarity in [0, 1] for a single resume–JD pair."""
        self._assert_fitted()
        vecs = self._vectorizer.transform([resume_text, jd_text])
        score = float(cosine_similarity(vecs[0], vecs[1])[0][0])
        return round(score, 4)

    def score_batch(
        self,
        pairs: List[Tuple[str, str, str, str]],
    ) -> List[Dict[str, Any]]:
        """
        Score multiple pairs efficiently with a single vectorizer call.

        pairs: list of (resume_id, jd_id, resume_text, jd_text)
        Returns: list of {resume_id, jd_id, tfidf_score}
        """
        self._assert_fitted()
        if not pairs:
            return []

        resume_ids, jd_ids, resume_texts, jd_texts = zip(*pairs)
        all_texts = list(resume_texts) + list(jd_texts)
        vecs = self._vectorizer.transform(all_texts)

        n = len(pairs)
        r_vecs = vecs[:n]
        j_vecs = vecs[n:]

        # Compute diagonal of the cosine similarity matrix
        scores = np.array(cosine_similarity(r_vecs, j_vecs).diagonal(), dtype=float)

        results = [
            {
                "resume_id":    resume_ids[i],
                "jd_id":        jd_ids[i],
                "tfidf_score":  round(float(scores[i]), 4),
            }
            for i in range(n)
        ]

        self.log_metrics({
            "pairs_scored": n,
            "mean_score":   round(float(scores.mean()), 4),
            "max_score":    round(float(scores.max()),  4),
            "min_score":    round(float(scores.min()),  4),
        })

        return results

    # Base agent

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thin wrapper for single-pair scoring that matches BaseAgent interface.

        Expected keys: resume_id, jd_id, resume_text, jd_text
        """
        required = ["resume_text", "jd_text"]
        if not self.validate_input(input_data, required):
            return {"error": "Missing required fields"}
        self._assert_fitted()

        score = self.score_pair(input_data["resume_text"], input_data["jd_text"])
        return {
            "resume_id":   input_data.get("resume_id", ""),
            "jd_id":       input_data.get("jd_id", ""),
            "tfidf_score": score,
        }

    def _assert_fitted(self):
        if not self._fitted:
            raise RuntimeError("TFIDFBaselineAgent is not fitted. Call fit() or fit_from_parquet() first.")
