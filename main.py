"""
SkillAlign AI Service — FastAPI Entry Point.

API Service untuk CV-Job Matching menggunakan Deep Learning.
Menyediakan endpoint untuk prediksi matching score.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv
load_dotenv()  # Baca .env sebelum apapun

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference.predict import SkillAlignPredictor
from src.inference.api_service import (
    router as api_router,
    set_predictor,
    warmup_skill_gap,
)
from src.inference.learning_path_router import router as lp_router

# ==========================================
# Logging Configuration
# ==========================================

def setup_logging() -> None:
    os.makedirs("logs", exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    file_handler = RotatingFileHandler(
        "logs/skillaign.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)


setup_logging()
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

    model_path        = os.getenv('MODEL_PATH',        'models/skillalign_matcher_v4.keras')
    preprocessor_path = os.getenv('PREPROCESSOR_PATH', 'preprocessors/nlp_preprocessor_v4.pkl')
    config_path       = os.getenv('CONFIG_PATH',       'models/model_config_v4.json')
    use_hybrid        = os.getenv('USE_HYBRID', 'true').lower() != 'false'

    # Coba download artifacts jika belum ada (untuk Railway deployment)
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        try:
            from scripts.download_models import ensure_models
            logger.info("Model belum ada, mencoba download dari cloud storage...")
            ensure_models()
        except Exception as dl_err:
            logger.warning(f"Model download gagal: {dl_err}")

    # Load model
    try:
        predictor = SkillAlignPredictor(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            config_path=config_path,
            use_hybrid=use_hybrid,
        )
        predictor.load()
        set_predictor(predictor)
        logger.info(
            f"✅ Model v4 loaded | threshold={predictor.optimal_threshold:.2f} | "
            f"hybrid={'ON' if use_hybrid else 'OFF'}"
        )
    except FileNotFoundError as e:
        logger.warning(
            f"⚠️  Model v4 belum tersedia: {e}\n"
            f"   Service berjalan tanpa model. Set MODEL_DOWNLOAD_URL di env vars\n"
            f"   dan restart, atau upload model ke folder models/ secara manual."
        )
    except Exception as e:
        logger.error(f"❌ Error loading model v4: {e}", exc_info=True)

    # Warmup SkillNer di background thread agar tidak blokir startup probe Cloud Run.
    # spaCy en_core_web_lg + EMSI 31k skills ~23 detik — terlalu lama untuk blocking startup.
    import threading
    threading.Thread(target=warmup_skill_gap, daemon=True, name="skillner-warmup").start()
    logger.info("SkillNer warmup dimulai di background thread.")

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
        "- `POST /api/v1/recommend` — Ranking job postings untuk satu CV\n"
        "- `POST /api/v1/skill-gap` — Analisis skill gap CV vs Job\n"
        "- `POST /api/v1/analyze-cv` — Ekstraksi profil CV + saran job title\n"
        "- `GET /health` — Health check\n\n"
        "**Model:** SkillAlignMatcherV4 (Multi-Scale CNN + Cross-Attention, Regression)\n"
        "**Threshold:** OPTIMAL_THRESHOLD=0.44 (dikalibrasi via F1-sweep pada val set)\n"
        "**Tim:** CC26-PSU318"
    ),
    version="4.0.0",
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
# Request Timing Middleware
# ==========================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed_ms = round((time.time() - start) * 1000, 2)
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({elapsed_ms}ms)"
    )
    return response


# ==========================================
# Include API Router
# ==========================================
app.include_router(api_router)
app.include_router(lp_router)


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
        "version": "4.0.0",
        "status": "running",
        "model": "skillalign_matcher_v4",
        "model_loaded": predictor is not None and predictor.is_loaded,
        "optimal_threshold": predictor.optimal_threshold if predictor and predictor.is_loaded else 0.44,
        "endpoints": {
            "single":   "/predict  atau  /api/v1/predict",
            "batch":    "/api/v1/predict/batch",
            "recommend":"/api/v1/recommend",
            "skill_gap":"/api/v1/skill-gap",
            "analyze_cv":"/api/v1/analyze-cv",
            "docs":     "/docs",
            "health":   "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint — dipakai Railway untuk startup probe."""
    model_ok = predictor is not None and predictor.is_loaded
    return {
        "status": "healthy",
        "model_loaded": model_ok,
        "model_version": "v4",
        "optimal_threshold": predictor.optimal_threshold if model_ok else None,
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