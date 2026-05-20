"""
Inference Module untuk SkillAlign AI (v4 — Multi-Scale CNN + Regression).

Perubahan dari v2:
- Model v4: SkillAlignMatcherV4 (attention_units=192, conv_filters=(256,64))
- Loss: Huber (regression) — bukan binary cross-entropy/focal_loss
- OPTIMAL_THRESHOLD: dikalibrasi dari val set (default 0.44), bukan hardcode 0.5
- auto-load threshold dari model_config_v4.json jika tersedia
- Hybrid Scoring tetap tersedia (default ON, bisa di-override env USE_HYBRID=false)
"""

import os
import json
import time
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf
import joblib

from src.models.custom_layers import CustomAttentionLayer
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
    Predictor untuk SkillAlign model v4 dengan optional Hybrid Scoring.

    Args:
        model_path: Path ke saved model (.keras).
        preprocessor_path: Path ke saved preprocessor (.pkl).
        config_path: Path ke model_config_v4.json (auto-read OPTIMAL_THRESHOLD).
        optimal_threshold: Threshold klasifikasi MATCH/NO MATCH.
            - Dibaca dari config_path jika tersedia.
            - Override via env var OPTIMAL_THRESHOLD.
            - Default 0.44 (dikalibrasi Fase 3B dari val set).
        use_hybrid: Aktifkan hybrid scoring (default True).
        hybrid_config: Custom HybridScorerConfig.

    Example:
        >>> predictor = SkillAlignPredictor()
        >>> predictor.load()
        >>> result = predictor.predict(
        ...     cv_text="3 years Python, Machine Learning...",
        ...     job_description="Looking for Data Scientist..."
        ... )
        >>> print(result.matching_score)        # final score (hybrid jika ON)
        >>> print(result.raw_model_score)       # raw v4 model output
        >>> print(result.structured_score)      # structured features score
    """

    # Threshold dari Fase 3B threshold calibration
    DEFAULT_OPTIMAL_THRESHOLD = 0.44

    def __init__(
        self,
        model_path: str = 'models/skillalign_matcher_v4.keras',
        preprocessor_path: str = 'preprocessors/nlp_preprocessor_v4.pkl',
        config_path: str = 'models/model_config_v4.json',
        optimal_threshold: Optional[float] = None,
        use_hybrid: bool = True,
        hybrid_config: Optional[HybridScorerConfig] = None,
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.config_path = config_path
        self.use_hybrid = use_hybrid
        self.hybrid_scorer: Optional[HybridScorer] = None
        if use_hybrid:
            self.hybrid_scorer = HybridScorer(config=hybrid_config)

        self.model: Optional[tf.keras.Model] = None
        self.preprocessor = None
        self.is_loaded = False

        # Threshold resolution order:
        # 1. explicit argument  2. env var  3. config JSON  4. default 0.44
        if optimal_threshold is not None:
            self.optimal_threshold = float(optimal_threshold)
        else:
            env_val = os.getenv('OPTIMAL_THRESHOLD')
            self.optimal_threshold = float(env_val) if env_val else self.DEFAULT_OPTIMAL_THRESHOLD

    def load(self) -> 'SkillAlignPredictor':
        """Load model, preprocessor, dan config dari disk."""
        # ── 1. Load model ──────────────────────────────────────────────────
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model tidak ditemukan: {self.model_path}\n"
                f"Pastikan model v4 sudah di-download ke folder models/\n"
                f"  atau set env MODEL_DOWNLOAD_URL dan restart service."
            )

        logger.info(f"Loading model dari: {self.model_path}")

        # v4 menggunakan Huber loss (built-in) — hanya perlu CustomAttentionLayer.
        # focal_loss disertakan untuk backward-compat jika ada legacy model v3.
        custom_objects = {
            'CustomAttentionLayer': CustomAttentionLayer,
        }
        try:
            from src.models.custom_loss import focal_loss
            custom_objects['focal_loss'] = focal_loss()
            custom_objects['loss'] = focal_loss()
        except Exception:
            pass  # v4 tidak butuh focal_loss

        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects=custom_objects,
            compile=False,  # tidak perlu compile untuk inference
        )

        # ── 2. Load preprocessor ───────────────────────────────────────────
        if os.path.exists(self.preprocessor_path):
            logger.info(f"Loading preprocessor dari: {self.preprocessor_path}")
            self.preprocessor = joblib.load(self.preprocessor_path)
        else:
            logger.warning(
                f"Preprocessor tidak ditemukan: {self.preprocessor_path}. "
                f"Pastikan preprocessor tersedia sebelum predict."
            )

        # ── 3. Load model config → auto-read OPTIMAL_THRESHOLD ────────────
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                cfg_threshold = (
                    config.get('evaluation_metrics', {}).get('optimal_threshold')
                )
                if cfg_threshold is not None and os.getenv('OPTIMAL_THRESHOLD') is None:
                    # Config takes precedence only if env var not set explicitly
                    self.optimal_threshold = float(cfg_threshold)
                    logger.info(
                        f"OPTIMAL_THRESHOLD={self.optimal_threshold:.2f} "
                        f"(dari {self.config_path})"
                    )
            except Exception as e:
                logger.warning(f"Gagal baca model config: {e}")

        self.is_loaded = True
        mode = "Hybrid" if self.use_hybrid else "Raw model"
        logger.info(
            f"Model loaded successfully. Mode: {mode} | "
            f"OPTIMAL_THRESHOLD={self.optimal_threshold:.2f}"
        )
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

        confidence = self._get_confidence(final_score, self.optimal_threshold)
        recommendation = self._get_recommendation(final_score, self.optimal_threshold)

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
                confidence=self._get_confidence(final_score, self.optimal_threshold),
                recommendation=self._get_recommendation(final_score, self.optimal_threshold),
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
    def _get_confidence(score: float, threshold: float = 0.44) -> str:
        """
        Klasifikasi confidence berdasarkan OPTIMAL_THRESHOLD terkalibrasi.

        Tiers:
          High   : score > threshold + 0.25  (jauh di atas threshold)
          Medium : score > threshold          (di atas threshold)
          Low    : score <= threshold         (di bawah threshold)
        """
        high_band = threshold + 0.25
        if score > high_band:
            return "High"
        elif score > threshold:
            return "Medium"
        else:
            return "Low"

    @staticmethod
    def _get_recommendation(score: float, threshold: float = 0.44) -> str:
        """
        Teks rekomendasi berdasarkan OPTIMAL_THRESHOLD terkalibrasi.

        Tiers (sama dengan _get_confidence):
          Highly Recommended : score > threshold + 0.25
          Consider           : score > threshold
          Not Recommended    : score <= threshold
        """
        high_band = threshold + 0.25
        if score > high_band:
            return "Highly Recommended"
        elif score > threshold:
            return "Consider"
        else:
            return "Not Recommended"