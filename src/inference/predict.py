"""
Inference Module untuk SkillAlign AI.

Menyediakan fungsi dan class untuk melakukan prediksi
matching score antara CV dan Job Description.
"""

import os
import time
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import joblib

from src.models.custom_layers import CustomAttentionLayer
from src.models.custom_loss import focal_loss

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """
    Data class untuk hasil prediksi.

    Attributes:
        matching_score: Skor matching (0.0 - 1.0).
        confidence: Level confidence ('High', 'Medium', 'Low').
        recommendation: Rekomendasi ('Highly Recommended', 'Consider',
            'Not Recommended').
        inference_time_ms: Waktu inferensi dalam milidetik.
        skill_gap: Skill yang perlu ditingkatkan (opsional).
    """
    matching_score: float
    confidence: str
    recommendation: str
    inference_time_ms: float
    skill_gap: Optional[List[str]] = None


class SkillAlignPredictor:
    """
    Predictor untuk SkillAlign model.

    Menghandle loading model, preprocessing input,
    dan generating predictions.

    Args:
        model_path: Path ke saved model (.keras atau SavedModel).
            Default 'models/skillalign_matcher.keras'.
        preprocessor_path: Path ke saved preprocessor (.pkl).
            Default 'preprocessors/nlp_preprocessor.pkl'.

    Example:
        >>> predictor = SkillAlignPredictor()
        >>> predictor.load()
        >>> result = predictor.predict(
        ...     cv_text="3 years Python, Machine Learning...",
        ...     job_description="Looking for Data Scientist..."
        ... )
        >>> print(result.matching_score)
    """

    def __init__(
        self,
        model_path: str = 'models/skillalign_matcher.keras',
        preprocessor_path: str = 'preprocessors/nlp_preprocessor.pkl'
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model: Optional[tf.keras.Model] = None
        self.preprocessor = None
        self.is_loaded = False

    def load(self) -> 'SkillAlignPredictor':
        """
        Load model dan preprocessor dari disk.

        Returns:
            self: Untuk method chaining.

        Raises:
            FileNotFoundError: Jika model/preprocessor tidak ditemukan.
        """
        # Load model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model tidak ditemukan: {self.model_path}"
            )

        logger.info(f"Loading model dari: {self.model_path}")

        custom_objects = {
            'CustomAttentionLayer': CustomAttentionLayer,
            'focal_loss': focal_loss(),
            'loss': focal_loss()
        }

        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects=custom_objects
        )

        # Load preprocessor
        if os.path.exists(self.preprocessor_path):
            logger.info(
                f"Loading preprocessor dari: {self.preprocessor_path}"
            )
            self.preprocessor = joblib.load(self.preprocessor_path)
        else:
            logger.warning(
                f"Preprocessor tidak ditemukan: "
                f"{self.preprocessor_path}. "
                f"Pastikan preprocessor tersedia sebelum predict."
            )

        self.is_loaded = True
        logger.info("Model dan preprocessor loaded successfully.")
        return self

    def predict(
        self,
        cv_text: str,
        job_description: str
    ) -> PredictionResult:
        """
        Predict matching score antara CV dan Job Description.

        Args:
            cv_text: Teks CV pengguna.
            job_description: Teks deskripsi lowongan kerja.

        Returns:
            PredictionResult: Hasil prediksi lengkap.

        Raises:
            RuntimeError: Jika model/preprocessor belum di-load.
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Model belum di-load. Panggil load() terlebih dahulu."
            )
        if self.preprocessor is None:
            raise RuntimeError(
                "Preprocessor belum tersedia."
            )

        start_time = time.time()

        # Preprocessing
        cv_seq = self.preprocessor.process(cv_text)
        job_seq = self.preprocessor.process(job_description)

        # Prediction
        score = self.model.predict(
            [cv_seq, job_seq], verbose=0
        )[0][0]

        inference_time = (time.time() - start_time) * 1000

        # Determine confidence dan recommendation
        confidence = self._get_confidence(float(score))
        recommendation = self._get_recommendation(float(score))

        return PredictionResult(
            matching_score=round(float(score), 4),
            confidence=confidence,
            recommendation=recommendation,
            inference_time_ms=round(inference_time, 2)
        )

    def predict_batch(
        self,
        cv_texts: List[str],
        job_descriptions: List[str]
    ) -> List[PredictionResult]:
        """
        Batch prediction untuk multiple CV-Job pairs.

        Args:
            cv_texts: List of CV texts.
            job_descriptions: List of job description texts.

        Returns:
            List of PredictionResult.

        Raises:
            ValueError: Jika panjang list tidak sama.
        """
        if len(cv_texts) != len(job_descriptions):
            raise ValueError(
                f"Jumlah CV ({len(cv_texts)}) dan "
                f"Job ({len(job_descriptions)}) harus sama."
            )

        if not self.is_loaded or self.preprocessor is None:
            raise RuntimeError("Model/preprocessor belum di-load.")

        start_time = time.time()

        # Batch preprocessing
        cv_seqs = self.preprocessor.transform(cv_texts)
        job_seqs = self.preprocessor.transform(job_descriptions)

        # Batch prediction
        scores = self.model.predict(
            [cv_seqs, job_seqs], verbose=0
        ).flatten()

        total_time = (time.time() - start_time) * 1000
        per_item_time = total_time / len(cv_texts)

        results = []
        for score in scores:
            score_val = float(score)
            results.append(PredictionResult(
                matching_score=round(score_val, 4),
                confidence=self._get_confidence(score_val),
                recommendation=self._get_recommendation(score_val),
                inference_time_ms=round(per_item_time, 2)
            ))

        return results

    def predict_top_jobs(
        self,
        cv_text: str,
        job_descriptions: List[str],
        job_titles: Optional[List[str]] = None,
        top_n: int = 5
    ) -> List[Tuple[int, PredictionResult, Optional[str]]]:
        """
        Ranking top matching jobs untuk satu CV.

        Args:
            cv_text: Single CV text.
            job_descriptions: List of job descriptions.
            job_titles: List of job titles (opsional).
            top_n: Jumlah top jobs yang diambil. Default 5.

        Returns:
            List of (index, PredictionResult, title) tuples,
            sorted by matching_score descending.
        """
        cv_texts = [cv_text] * len(job_descriptions)
        results = self.predict_batch(cv_texts, job_descriptions)

        # Pair dengan index dan title
        paired = []
        for i, result in enumerate(results):
            title = job_titles[i] if job_titles else None
            paired.append((i, result, title))

        # Sort by matching_score descending
        paired.sort(key=lambda x: x[1].matching_score, reverse=True)

        return paired[:top_n]

    @staticmethod
    def _get_confidence(score: float) -> str:
        """Map score ke confidence level."""
        if score > 0.7:
            return "High"
        elif score > 0.4:
            return "Medium"
        else:
            return "Low"

    @staticmethod
    def _get_recommendation(score: float) -> str:
        """Map score ke recommendation."""
        if score > 0.7:
            return "Highly Recommended"
        elif score > 0.4:
            return "Consider"
        else:
            return "Not Recommended"
