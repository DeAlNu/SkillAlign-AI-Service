"""
SkillAlign AI Service — FastAPI Entry Point.

API Service untuk CV-Job Matching menggunakan Deep Learning.
Menyediakan endpoint untuk prediksi matching score.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models.custom_layers import CustomAttentionLayer
from src.models.custom_loss import focal_loss
from src.inference.predict import SkillAlignPredictor
from src.inference.api_service import (
    router as api_router,
    set_predictor
)

# ==========================================
# Logging Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# Global Predictor
# ==========================================
predictor = None


# ==========================================
# Lifespan (startup/shutdown)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Load model saat startup, cleanup saat shutdown.
    """
    global predictor

    model_path = os.getenv(
        'MODEL_PATH', 'models/skillalign_matcher_v2.keras'
    )
    preprocessor_path = os.getenv(
        'PREPROCESSOR_PATH', 'preprocessors/nlp_preprocessor_v2.pkl'
    )

    # Coba load model (jika tersedia)
    try:
        predictor = SkillAlignPredictor(
            model_path=model_path,
            preprocessor_path=preprocessor_path
        )
        predictor.load()
        set_predictor(predictor)
        logger.info("Model loaded successfully on startup.")
    except FileNotFoundError:
        logger.warning(
            "Model/preprocessor belum tersedia. "
            "Service berjalan tanpa model. "
            "Train dan save model terlebih dahulu."
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")

    yield  # Application runs

    # Cleanup
    logger.info("Shutting down SkillAlign AI Service.")


# ==========================================
# FastAPI App
# ==========================================
app = FastAPI(
    title="SkillAlign AI Service",
    description=(
        "API untuk CV-Job Matching menggunakan Deep Learning.\n\n"
        "**Endpoints:**\n"
        "- `POST /predict` — Single prediction (CV vs 1 Job)\n"
        "- `POST /api/v1/predict` — Single prediction (versioned)\n"
        "- `POST /api/v1/predict/batch` — Batch prediction (CV vs ≤50 Jobs)\n"
        "- `GET /health` — Health check\n\n"
        "**Model:** SkillAlign Matcher v2 (Dual-Input CNN + Custom Attention)\n"
        "**Tim:** CC26-PSU318"
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# ==========================================
# CORS Middleware
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Sesuaikan untuk production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Include API Router
# ==========================================
app.include_router(api_router)


# ==========================================
# Root & Health Endpoints
# ==========================================

class PredictionRequest(BaseModel):
    """Request body untuk /predict endpoint."""
    cv_text: str = Field(
        ...,
        min_length=50,
        description="Teks CV pengguna"
    )
    job_description: str = Field(
        ...,
        min_length=30,
        description="Teks deskripsi lowongan kerja"
    )
    user_id: str | None = None


class PredictionResponse(BaseModel):
    """Response body untuk /predict endpoint."""
    matching_score: float
    confidence: str
    recommendation: str


@app.get("/")
async def root():
    """Root endpoint — API info."""
    return {
        "service": "SkillAlign AI Service",
        "version": "2.0.0",
        "status": "running",
        "model": "skillalign_matcher_v2",
        "endpoints": {
            "single": "/predict  atau  /api/v1/predict",
            "batch": "/api/v1/predict/batch",
            "docs": "/docs",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None and predictor.is_loaded
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict matching score antara CV dan Job Description.

    Args:
        request: PredictionRequest dengan cv_text dan job_description

    Returns:
        PredictionResponse dengan matching_score
    """
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model belum loaded. "
                "Train dan deploy model terlebih dahulu."
            )
        )

    try:
        result = predictor.predict(
            cv_text=request.cv_text,
            job_description=request.job_description
        )

        return PredictionResponse(
            matching_score=result.matching_score,
            confidence=result.confidence,
            recommendation=result.recommendation
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))