"""
API Service (FastAPI Router) untuk SkillAlign AI.

Router terpisah yang bisa di-include ke main FastAPI app.
Menyediakan endpoint /predict dan /predict/batch.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Router instance
router = APIRouter(prefix="/api/v1", tags=["prediction"])


# ==========================================
# Request / Response Schemas
# ==========================================

class SinglePredictionRequest(BaseModel):
    """Request body untuk single prediction."""
    cv_text: str = Field(
        ...,
        min_length=50,
        max_length=10000,
        description="Teks CV pengguna (minimal 50 karakter)"
    )
    job_description: str = Field(
        ...,
        min_length=30,
        max_length=10000,
        description="Teks deskripsi lowongan kerja"
    )
    user_id: Optional[str] = Field(
        None,
        description="ID pengguna (opsional)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "cv_text": (
                    "Experienced Data Scientist with 3 years of experience "
                    "in Python, Machine Learning, TensorFlow, and data analysis."
                ),
                "job_description": (
                    "Looking for a Data Scientist with strong Python skills, "
                    "experience in ML frameworks and data visualization."
                ),
                "user_id": "user_123"
            }]
        }
    }


class BatchPredictionRequest(BaseModel):
    """Request body untuk batch prediction."""
    cv_text: str = Field(
        ...,
        min_length=50,
        max_length=10000,
        description="Teks CV pengguna"
    )
    job_descriptions: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of job descriptions (max 50)"
    )
    user_id: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response body untuk single prediction."""
    matching_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Skor matching 0-1"
    )
    confidence: str = Field(
        ...,
        description="Confidence level: High/Medium/Low"
    )
    recommendation: str = Field(
        ...,
        description="Recommendation text"
    )
    inference_time_ms: float = Field(
        ...,
        description="Waktu inferensi (ms)"
    )


class BatchPredictionResponse(BaseModel):
    """Response body untuk batch prediction."""
    results: List[PredictionResponse]
    total_items: int
    total_time_ms: float


# ==========================================
# Dependency: Predictor instance
# Akan di-set dari main.py saat startup
# ==========================================
_predictor = None


def set_predictor(predictor) -> None:
    """Set predictor instance (dipanggil dari main.py startup)."""
    global _predictor
    _predictor = predictor


def get_predictor():
    """Get predictor instance."""
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model belum loaded. Coba lagi nanti."
        )
    return _predictor


# ==========================================
# Endpoints
# ==========================================

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict CV-Job Matching Score",
    description="Prediksi skor kecocokan antara CV dan satu job description."
)
async def predict_single(request: SinglePredictionRequest):
    """
    Predict matching score antara CV dan Job Description.

    - **cv_text**: Teks CV pengguna (hasil ekstraksi PDF)
    - **job_description**: Deskripsi lowongan kerja
    - **user_id**: ID pengguna (opsional, untuk tracking)

    Returns matching score (0-1), confidence level, dan recommendation.
    """
    predictor = get_predictor()

    try:
        result = predictor.predict(
            cv_text=request.cv_text,
            job_description=request.job_description
        )

        return PredictionResponse(
            matching_score=result.matching_score,
            confidence=result.confidence,
            recommendation=result.recommendation,
            inference_time_ms=result.inference_time_ms
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Predict CV-Job Matching",
    description="Prediksi skor kecocokan CV terhadap multiple job descriptions."
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction: satu CV terhadap multiple job descriptions.

    Berguna untuk ranking jobs yang paling cocok dengan CV pengguna.
    Maksimum 50 job descriptions per request.
    """
    predictor = get_predictor()

    try:
        cv_texts = [request.cv_text] * len(request.job_descriptions)
        results = predictor.predict_batch(
            cv_texts=cv_texts,
            job_descriptions=request.job_descriptions
        )

        total_time = sum(r.inference_time_ms for r in results)

        response_results = [
            PredictionResponse(
                matching_score=r.matching_score,
                confidence=r.confidence,
                recommendation=r.recommendation,
                inference_time_ms=r.inference_time_ms
            )
            for r in results
        ]

        return BatchPredictionResponse(
            results=response_results,
            total_items=len(results),
            total_time_ms=round(total_time, 2)
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )
