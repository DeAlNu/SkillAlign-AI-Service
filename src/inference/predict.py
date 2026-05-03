"""
Inference Module untuk SkillAlign AI (v2 — dengan Hybrid Scoring).

Menyediakan fungsi dan class untuk melakukan prediksi
matching score antara CV dan Job Description.

Perubahan dari versi sebelumnya:
- Tambah integrasi HybridScorer (default ON)
- Final score = alpha*model_score + (1-alpha)*structured_score
- Backward compatible: kalau use_hybrid=False, behavior sama seperti sebelumnya
"""

import os
import time
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf
import joblib

from src.models.custom_layers import CustomAttentionLayer
from src.models.custom_loss import focal_loss
from src.inference.hybrid_scorer import HybridScorer, HybridScorerConfig

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """
    Data class untuk hasil prediksi.

    Attributes:
        matching_score: Final score (hybrid kalau enabled, atau raw model score).
        confidence: Level confidence ('High', 'Medium', 'Low').
        recommendation: Rekomendasi.
        inference_time_ms: Waktu inferensi dalam milidetik.
        raw_model_score: Score dari neural model saja (sebelum hybrid).
        structured_score: Score dari structured features (kalau hybrid enabled).
        skill_gap: Skill yang perlu ditingkatkan (opsional).
    """
    matching_score: float
    confidence: str
    recommendation: str
    inference_time_ms: float
    raw_model_score: Optional[float] = None
    structured_score: Optional[float] = None
    skill_gap: Optional[List[str]] = None


class SkillAlignPredictor:
    """
    Predictor untuk SkillAlign model dengan optional Hybrid Scoring.

    Args:
        model_path: Path ke saved model (.keras atau SavedModel).
        preprocessor_path: Path ke saved preprocessor (.pkl).
        use_hybrid: Aktifkan hybrid scoring (default True).
        hybrid_config: Custom HybridScorerConfig (default uses defaults).

    Example:
        >>> predictor = SkillAlignPredictor()
        >>> predictor.load()
        >>> result = predictor.predict(
        ...     cv_text="3 years Python, Machine Learning...",
        ...     job_description="Looking for Data Scientist..."
        ... )
        >>> print(result.matching_score)        # final hybrid score
        >>> print(result.raw_model_score)       # raw neural model output
        >>> print(result.structured_score)      # structured features score
    """

    def __init__(
        self,
        model_path: str = 'models/skillalign_matcher.keras',
        preprocessor_path: str = 'preprocessors/nlp_preprocessor.pkl',
        use_hybrid: bool = True,
        hybrid_config: Optional[HybridScorerConfig] = None,
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.use_hybrid = use_hybrid
        self.hybrid_scorer: Optional[HybridScorer] = None
        if use_hybrid:
            self.hybrid_scorer = HybridScorer(config=hybrid_config)

        self.model: Optional[tf.keras.Model] = None
        self.preprocessor = None
        self.is_loaded = False

    def load(self) -> 'SkillAlignPredictor':
        """Load model dan preprocessor dari disk."""
        # Load model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model tidak ditemukan: {self.model_path}"
            )

        logger.info(f"Loading model dari: {self.model_path}")

        custom_objects = {
            'CustomAttentionLayer': CustomAttentionLayer,
            'focal_loss': focal_loss(),
            'loss': focal_loss(),
        }

        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects=custom_objects,
            compile=False,  # tidak perlu compile untuk inference
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
        mode = "Hybrid" if self.use_hybrid else "Raw model"
        logger.info(f"Model loaded successfully. Mode: {mode}")
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
            PredictionResult dengan matching_score (hybrid kalau enabled).
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Model belum di-load. Panggil load() terlebih dahulu."
            )
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor belum tersedia.")

        start_time = time.time()

        # Preprocessing
        cv_seq = self.preprocessor.process(cv_text)
        job_seq = self.preprocessor.process(job_description)

        # Model prediction (raw)
        raw_score = float(self.model.predict(
            [cv_seq, job_seq], verbose=0
        )[0][0])

        # Optional hybrid scoring
        structured_score: Optional[float] = None
        if self.use_hybrid and self.hybrid_scorer is not None:
            structured_score = self.hybrid_scorer.compute_structured(
                cv_text, job_description
            )
            final_score = self.hybrid_scorer.compute(
                model_score=raw_score,
                cv_text=cv_text,
                job_text=job_description,
            )
        else:
            final_score = raw_score

        inference_time = (time.time() - start_time) * 1000

        confidence = self._get_confidence(final_score)
        recommendation = self._get_recommendation(final_score)

        return PredictionResult(
            matching_score=round(final_score, 4),
            confidence=confidence,
            recommendation=recommendation,
            inference_time_ms=round(inference_time, 2),
            raw_model_score=round(raw_score, 4),
            structured_score=round(structured_score, 4) if structured_score is not None else None,
        )

    def predict_batch(
        self,
        cv_texts: List[str],
        job_descriptions: List[str]
    ) -> List[PredictionResult]:
        """Batch prediction untuk multiple CV-Job pairs."""
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

        # Batch model prediction
        raw_scores = self.model.predict(
            [cv_seqs, job_seqs], verbose=0
        ).flatten()

        # Per-pair hybrid scoring (structured features can't be batched easily)
        results = []
        for i, raw_score in enumerate(raw_scores):
            raw_score = float(raw_score)
            structured_score: Optional[float] = None
            if self.use_hybrid and self.hybrid_scorer is not None:
                structured_score = self.hybrid_scorer.compute_structured(
                    cv_texts[i], job_descriptions[i]
                )
                final_score = self.hybrid_scorer.compute(
                    model_score=raw_score,
                    cv_text=cv_texts[i],
                    job_text=job_descriptions[i],
                )
            else:
                final_score = raw_score

            results.append(PredictionResult(
                matching_score=round(final_score, 4),
                confidence=self._get_confidence(final_score),
                recommendation=self._get_recommendation(final_score),
                inference_time_ms=0.0,  # set later
                raw_model_score=round(raw_score, 4),
                structured_score=round(structured_score, 4) if structured_score is not None else None,
            ))

        total_time = (time.time() - start_time) * 1000
        per_item_time = total_time / max(len(cv_texts), 1)
        for r in results:
            r.inference_time_ms = round(per_item_time, 2)

        return results

    def predict_top_jobs(
        self,
        cv_text: str,
        job_descriptions: List[str],
        job_titles: Optional[List[str]] = None,
        top_n: int = 5
    ) -> List[Tuple[int, PredictionResult, Optional[str]]]:
        """Ranking top matching jobs untuk satu CV."""
        cv_texts = [cv_text] * len(job_descriptions)
        results = self.predict_batch(cv_texts, job_descriptions)

        paired = []
        for i, result in enumerate(results):
            title = job_titles[i] if job_titles else None
            paired.append((i, result, title))

        paired.sort(key=lambda x: x[1].matching_score, reverse=True)
        return paired[:top_n]

    @staticmethod
    def _get_confidence(score: float) -> str:
        if score > 0.7:
            return "High"
        elif score > 0.4:
            return "Medium"
        else:
            return "Low"

    @staticmethod
    def _get_recommendation(score: float) -> str:
        if score > 0.7:
            return "Highly Recommended"
        elif score > 0.4:
            return "Consider"
        else:
            return "Not Recommended"