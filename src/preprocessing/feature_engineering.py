"""
Feature Engineering untuk SkillAlign AI.

Modul ini menyediakan fitur tambahan seperti TF-IDF vectorization,
skill overlap scoring, dan similarity metrics.
"""

import logging
from typing import List, Tuple, Set, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature Engineering untuk CV-Job Matching.

    Menyediakan beberapa metode untuk mengekstrak fitur tambahan
    dari teks CV dan Job Description.

    Args:
        max_tfidf_features: Jumlah fitur maksimum TF-IDF. Default 1000.
        ngram_range: Range n-gram untuk TF-IDF. Default (1, 2).

    Example:
        >>> fe = FeatureEngineer(max_tfidf_features=1000)
        >>> fe.fit(all_texts)
        >>> cv_tfidf = fe.transform_tfidf(cv_texts)
    """

    def __init__(
        self,
        max_tfidf_features: int = 1000,
        ngram_range: Tuple[int, int] = (1, 2)
    ):
        self.max_tfidf_features = max_tfidf_features
        self.ngram_range = ngram_range

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents='unicode'
        )
        self.is_fitted = False

    def fit(self, texts: List[str]) -> 'FeatureEngineer':
        """
        Fit TF-IDF vectorizer pada semua teks.

        Args:
            texts: List of preprocessed texts (CV + Job combined).

        Returns:
            self: Untuk method chaining.
        """
        self.tfidf_vectorizer.fit(texts)
        self.is_fitted = True
        logger.info(
            f"TF-IDF fitted. Features: {self.max_tfidf_features}"
        )
        return self

    def extract_tfidf_features(
        self,
        cv_texts: List[str],
        job_texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract TF-IDF features dari CV dan Job Description.

        Fit pada combined texts dan transform masing-masing.

        Args:
            cv_texts: List of preprocessed CV texts.
            job_texts: List of preprocessed job description texts.

        Returns:
            Tuple (cv_tfidf, job_tfidf): sparse matrices.
        """
        # Fit pada combined texts
        all_texts = list(cv_texts) + list(job_texts)
        self.fit(all_texts)

        # Transform masing-masing
        cv_tfidf = self.tfidf_vectorizer.transform(cv_texts)
        job_tfidf = self.tfidf_vectorizer.transform(job_texts)

        return cv_tfidf, job_tfidf

    def transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """
        Transform teks menggunakan fitted TF-IDF vectorizer.

        Args:
            texts: List of preprocessed texts.

        Returns:
            np.ndarray: TF-IDF feature matrix.

        Raises:
            RuntimeError: Jika vectorizer belum di-fit.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "TF-IDF vectorizer belum di-fit. "
                "Panggil fit() atau extract_tfidf_features() dahulu."
            )
        return self.tfidf_vectorizer.transform(texts)

    @staticmethod
    def compute_skill_overlap(
        cv_skills: Set[str],
        job_skills: Set[str]
    ) -> dict:
        """
        Compute skill overlap score antara CV dan Job.

        Menghitung intersection, union, dan Jaccard similarity
        antara set skill dari CV dan Job Description.

        Args:
            cv_skills: Set of skills dari CV.
            job_skills: Set of skills dari Job Description.

        Returns:
            dict: {
                'overlap_count': int,
                'overlap_ratio': float (0-1),
                'jaccard_similarity': float (0-1),
                'missing_skills': set,
                'matched_skills': set
            }
        """
        if not cv_skills or not job_skills:
            return {
                'overlap_count': 0,
                'overlap_ratio': 0.0,
                'jaccard_similarity': 0.0,
                'missing_skills': job_skills or set(),
                'matched_skills': set()
            }

        # Normalize skills ke lowercase
        cv_norm = {s.lower().strip() for s in cv_skills}
        job_norm = {s.lower().strip() for s in job_skills}

        # Intersection dan Union
        matched = cv_norm & job_norm
        union = cv_norm | job_norm
        missing = job_norm - cv_norm

        # Overlap ratio terhadap job requirements
        overlap_ratio = len(matched) / len(job_norm) if job_norm else 0.0

        # Jaccard similarity
        jaccard = len(matched) / len(union) if union else 0.0

        return {
            'overlap_count': len(matched),
            'overlap_ratio': round(overlap_ratio, 4),
            'jaccard_similarity': round(jaccard, 4),
            'missing_skills': missing,
            'matched_skills': matched
        }

    @staticmethod
    def compute_tfidf_similarity(
        cv_text: str,
        job_text: str,
        vectorizer: Optional[TfidfVectorizer] = None
    ) -> float:
        """
        Hitung cosine similarity antara CV dan Job
        berdasarkan TF-IDF representation.

        Args:
            cv_text: Preprocessed CV text.
            job_text: Preprocessed job description text.
            vectorizer: TfidfVectorizer instance (opsional).

        Returns:
            float: Cosine similarity score (0-1).
        """
        if vectorizer is None:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2)
            )
            vectors = vectorizer.fit_transform([cv_text, job_text])
        else:
            vectors = vectorizer.transform([cv_text, job_text])

        similarity = cosine_similarity(
            vectors[0:1], vectors[1:2]
        )[0][0]

        return float(round(similarity, 4))

    @staticmethod
    def extract_skills_from_text(
        text: str,
        skill_dictionary: List[str]
    ) -> Set[str]:
        """
        Extract skills dari teks berdasarkan skill dictionary.

        Args:
            text: Input text (CV atau Job Description).
            skill_dictionary: List of known skills.

        Returns:
            Set of matched skills.
        """
        text_lower = text.lower()
        found_skills = set()

        for skill in skill_dictionary:
            skill_lower = skill.lower().strip()
            if skill_lower and skill_lower in text_lower:
                found_skills.add(skill)

        return found_skills