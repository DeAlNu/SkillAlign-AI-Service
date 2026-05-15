"""
API Service (FastAPI Router) untuk SkillAlign AI.

Router terpisah yang bisa di-include ke main FastAPI app.
Menyediakan endpoint:
  - /predict, /predict/batch   — single & batch CV-Job matching
  - /skill-gap                 — skill gap analysis
  - /extract-cv-skills         — ekstraksi skill dari CV (SkillNer)
  - /analyze-cv                — analisis CV: profil + suggested job titles  [NEW]
  - /recommend                 — ranking job postings berdasarkan CV + job title  [NEW]
"""

import logging
import time
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.inference.skill_gap import SkillGapAnalyzer
from src.inference.cv_profile_extractor import CvProfileExtractor
from src.inference.job_title_suggester import JobTitleSuggester
from src.inference.industry_skill_analyzer import IndustrySkillAnalyzer

logger = logging.getLogger(__name__)

# Router instance
router = APIRouter(prefix="/api/v1", tags=["prediction"])

# Singleton instances (stateless, aman dipakai bersamaan)
_skill_gap_analyzer      = SkillGapAnalyzer()
_cv_profile_extractor    = CvProfileExtractor()
_job_title_suggester     = JobTitleSuggester()
_industry_skill_analyzer = IndustrySkillAnalyzer()


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


class BatchPredictionItem(BaseModel):
    """Satu item hasil batch prediction, sudah diurutkan berdasarkan skor."""
    rank: int = Field(
        ...,
        description="Peringkat job (1 = paling cocok)"
    )
    job_index: int = Field(
        ...,
        description="Posisi job dalam array input (0-based), untuk tracing di Backend"
    )
    matching_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Skor kecocokan CV-Job (0.0-1.0)"
    )
    confidence: str = Field(
        ...,
        description="Confidence level: High / Medium / Low"
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
    """Response body untuk batch prediction, diurutkan dari skor tertinggi."""
    results: List[BatchPredictionItem]
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
        raw_results = predictor.predict_batch(
            cv_texts=cv_texts,
            job_descriptions=request.job_descriptions
        )

        total_time = sum(r.inference_time_ms for r in raw_results)

        # Gabungkan hasil dengan job_index asli, lalu urutkan berdasarkan skor
        indexed = [
            {
                "job_index": i,
                "result": r
            }
            for i, r in enumerate(raw_results)
        ]
        indexed.sort(key=lambda x: x["result"].matching_score, reverse=True)

        # Buat response dengan rank yang jelas
        ranked_results = [
            BatchPredictionItem(
                rank=rank + 1,                          # 1-based rank
                job_index=item["job_index"],            # posisi di input array
                matching_score=item["result"].matching_score,
                confidence=item["result"].confidence,
                recommendation=item["result"].recommendation,
                inference_time_ms=item["result"].inference_time_ms
            )
            for rank, item in enumerate(indexed)
        ]

        return BatchPredictionResponse(
            results=ranked_results,
            total_items=len(ranked_results),
            total_time_ms=round(total_time, 2)
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


# ==========================================
# Skill Gap Endpoint
# ==========================================

class SkillGapRequest(BaseModel):
    """Request body untuk skill gap analysis."""
    cv_text: str = Field(
        ...,
        min_length=50,
        max_length=10000,
        description="Teks CV pengguna"
    )
    job_description: str = Field(
        ...,
        min_length=30,
        max_length=10000,
        description="Teks deskripsi lowongan kerja target"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "cv_text": (
                    "Data Analyst with 3 years of experience. "
                    "Skilled in SQL, Excel, and PowerBI for reporting. "
                    "Basic knowledge of Python for data cleaning."
                ),
                "job_description": (
                    "Data Scientist position requiring Python, machine learning, "
                    "TensorFlow, statistical modeling, and A/B testing."
                )
            }]
        }
    }


class SkillItemResponse(BaseModel):
    """Satu skill yang terdeteksi oleh SkillNer dari EMSI database."""
    skill: str = Field(..., description="Nama skill (dari EMSI skill database)")
    skill_id: str = Field("", description="EMSI canonical skill ID")
    match_score: float = Field(0.0, description="Confidence SkillNer (1.0=full match, <1=ngram match)")
    priority: int = Field(0, description="Urutan prioritas untuk missing skills (1=paling penting)")


class SkillGapResponse(BaseModel):
    """Response skill gap analysis."""
    skill_gap_score: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Skor kesesuaian skill (0.0 = tidak ada yg match, 1.0 = semua match)"
    )
    skill_coverage_percent: str = Field(
        ...,
        description="Persentase skill job requirement yang sudah dimiliki CV"
    )
    top_priority_skill: str = Field(
        ...,
        description="Skill paling penting yang perlu dipelajari terlebih dahulu"
    )
    present_skills: List[SkillItemResponse] = Field(
        ...,
        description="Skill yang sudah ada di CV sesuai requirement job"
    )
    missing_skills: List[SkillItemResponse] = Field(
        ...,
        description="Skill yang belum ada di CV tapi dibutuhkan job, diurutkan berdasarkan prioritas"
    )
    recommendation_summary: str = Field(
        ...,
        description="Ringkasan rekomendasi dalam bahasa natural"
    )
    analysis_time_ms: float = Field(
        ...,
        description="Waktu analisis (ms)"
    )


@router.post(
    "/skill-gap",
    response_model=SkillGapResponse,
    summary="Analyze CV Skill Gap vs Job Description",
    description=(
        "Analisis gap antara skill di CV dan requirement di Job Description. "
        "Mengembalikan daftar skill yang sudah dimiliki, skill yang masih kurang, "
        "dan rekomendasi prioritas belajar. Tidak membutuhkan model DL — menggunakan TF-IDF."
    )
)
async def analyze_skill_gap(request: SkillGapRequest):
    """
    Analisis skill gap antara CV dan Job Description target.

    - **cv_text**: Teks CV pengguna (min 50 karakter)
    - **job_description**: Teks deskripsi lowongan kerja target (min 30 karakter)

    Returns skill yang sudah dimiliki, skill yang masih kurang (berurutan prioritas),
    persentase coverage, dan ringkasan rekomendasi.
    """
    try:
        start = time.time()
        result = _skill_gap_analyzer.analyze(
            cv_text=request.cv_text,
            job_description=request.job_description
        )
        elapsed_ms = round((time.time() - start) * 1000, 2)

        return SkillGapResponse(
            skill_gap_score=result.skill_gap_score,
            skill_coverage_percent=result.skill_coverage_percent,
            top_priority_skill=result.top_priority_skill,
            present_skills=[
                SkillItemResponse(
                    skill=s.skill,
                    skill_id=s.skill_id,
                    match_score=s.match_score,
                    priority=s.priority
                )
                for s in result.present_skills
            ],
            missing_skills=[
                SkillItemResponse(
                    skill=s.skill,
                    skill_id=s.skill_id,
                    match_score=s.match_score,
                    priority=s.priority
                )
                for s in result.missing_skills
            ],
            recommendation_summary=result.recommendation_summary,
            analysis_time_ms=elapsed_ms
        )

    except Exception as e:
        logger.error(f"Skill gap analysis error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Skill gap analysis failed: {str(e)}"
        )


# ==========================================
# CV Skill Extraction Endpoint
# ==========================================

class CvSkillExtractionRequest(BaseModel):
    """Request body untuk ekstraksi skill dari CV."""
    cv_text: str = Field(
        ...,
        min_length=50,
        max_length=10000,
        description="Teks CV pengguna (raw text dari PDF)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "cv_text": (
                    "Software Engineer with 5 years experience in Python, Django, "
                    "React.js, PostgreSQL, Docker, and AWS cloud infrastructure. "
                    "Strong background in CI/CD pipelines and microservices architecture."
                )
            }]
        }
    }


class ExtractedSkillItem(BaseModel):
    """Satu skill yang terdeteksi SkillNer dari teks CV."""
    skill: str    = Field(..., description="Nama skill (lowercase, dari EMSI database)")
    skill_id: str = Field(..., description="EMSI canonical skill ID")
    confidence: float = Field(..., description="Confidence score (1.0=full match, <1=ngram)")


class CvSkillExtractionResponse(BaseModel):
    """Response ekstraksi skill dari CV."""
    skills: List[ExtractedSkillItem] = Field(
        ..., description="Daftar skill yang terdeteksi di CV, diurutkan by confidence"
    )
    skill_names: List[str] = Field(
        ..., description="Nama skill saja (tanpa metadata), berguna untuk DB keyword search"
    )
    total_skills: int = Field(..., description="Jumlah skill yang terdeteksi")
    extraction_time_ms: float = Field(..., description="Waktu ekstraksi (ms)")


@router.post(
    "/extract-cv-skills",
    response_model=CvSkillExtractionResponse,
    summary="Extract Skills from CV Text",
    description=(
        "Ekstrak daftar skill dari teks CV menggunakan SkillNer (EMSI skill database). "
        "Digunakan oleh backend untuk candidate generation sebelum batch prediction — "
        "menghasilkan keyword yang jauh lebih akurat daripada frequency-based extraction. "
        "Tidak membutuhkan job description."
    )
)
async def extract_cv_skills(request: CvSkillExtractionRequest):
    """
    Ekstrak skills dari CV text menggunakan SkillNer.

    Endpoint ini digunakan oleh backend Express.js sebagai Stage 1 dari
    2-stage recommendation pipeline:
      1. extract-cv-skills → dapatkan skill names dari CV
      2. Query DB dengan skill names sebagai keyword filter
      3. predict/batch → ranking job hasil filter

    Returns skills yang terdeteksi SkillNer (EMSI database, 31k+ skills),
    diurutkan dari confidence tertinggi.
    """
    try:
        start = time.time()

        # Gunakan _extract_skills internal dari SkillGapAnalyzer
        # Return: Dict[skill_id, (skill_name, confidence)]
        raw_skills = _skill_gap_analyzer._extract_skills(request.cv_text)

        # Sort by confidence descending
        sorted_skills = sorted(
            raw_skills.items(),
            key=lambda x: x[1][1],
            reverse=True
        )

        extracted = [
            ExtractedSkillItem(
                skill=name,
                skill_id=sid,
                confidence=round(conf, 4)
            )
            for sid, (name, conf) in sorted_skills
        ]

        elapsed_ms = round((time.time() - start) * 1000, 2)

        logger.info(
            f"CV skill extraction: {len(extracted)} skills found "
            f"in {elapsed_ms}ms"
        )

        return CvSkillExtractionResponse(
            skills=extracted,
            skill_names=[s.skill for s in extracted],
            total_skills=len(extracted),
            extraction_time_ms=elapsed_ms
        )

    except Exception as e:
        logger.error(f"CV skill extraction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"CV skill extraction failed: {str(e)}"
        )


# ==========================================
# Analyze CV Endpoint  [NEW]
# ==========================================

class AnalyzeCvRequest(BaseModel):
    """Request body untuk analisis profil CV."""
    cv_text: str = Field(
        ...,
        min_length=50,
        max_length=10000,
        description="Teks CV pengguna (raw text dari PDF)"
    )
    top_n_titles: int = Field(
        5,
        ge=1,
        le=10,
        description="Jumlah maksimal saran job title yang dikembalikan"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "cv_text": (
                    "Frontend Developer with 4 years of experience. "
                    "B.S. Computer Science, Universitas Indonesia. "
                    "Skills: React.js, TypeScript, Node.js, PostgreSQL, Docker."
                ),
                "top_n_titles": 5
            }]
        }
    }


class SuggestedJobTitleItem(BaseModel):
    """Satu saran job title dari analisis CV."""
    title:          str   = Field(..., description="Nama job title")
    role_key:       str   = Field(..., description="Internal role key")
    confidence:     float = Field(..., description="Skor kesesuaian (0.0–1.0)")
    reason:         str   = Field(..., description="Alasan saran")
    matched_skills: List[str] = Field(..., description="Skill CV yang relevan untuk role ini")


class AnalyzeCvResponse(BaseModel):
    """Response analisis profil CV."""
    # Profil kandidat
    current_role: Optional[str] = Field(None, description="Jabatan/posisi terdeteksi di CV")
    experience:   Optional[str] = Field(None, description="Total pengalaman (misal '4 Years')")
    education:    Optional[str] = Field(None, description="Gelar pendidikan tertinggi")

    # Skill yang terdeteksi di CV
    extracted_skills: List[str] = Field(
        ..., description="Kategori skill yang terdeteksi dari CV (dari EMSI/HybridScorer)"
    )

    # Saran job title untuk dipilih user
    suggested_job_titles: List[SuggestedJobTitleItem] = Field(
        ..., description="Daftar job title yang disarankan, diurutkan by confidence"
    )

    analysis_time_ms: float = Field(..., description="Waktu analisis (ms)")


@router.post(
    "/analyze-cv",
    response_model=AnalyzeCvResponse,
    summary="Analyze CV: Profile + Job Title Suggestions",
    description=(
        "Analisis teks CV untuk mengekstrak profil kandidat (jabatan, pengalaman, pendidikan) "
        "dan memberikan saran job title yang paling sesuai. "
        "Output digunakan oleh user untuk memilih target job title sebelum ke /recommend."
    ),
    tags=["recommendation"]
)
async def analyze_cv(request: AnalyzeCvRequest):
    """
    Analisis CV secara menyeluruh.

    **Tahap 1** dari alur rekomendasi 2-tahap:
    1. Ekstrak profil kandidat: current_role, experience, education
    2. Deteksi skill kategori dari CV (reuse HybridScorer)
    3. Buat saran job title berdasarkan skill overlap

    User kemudian memilih satu job title dari `suggested_job_titles`
    dan mengirimkannya ke `POST /api/v1/recommend`.
    """
    try:
        start = time.time()

        # 1. Ekstrak profil CV
        profile = _cv_profile_extractor.extract(request.cv_text)

        # 2. Suggest job titles (sekaligus deteksi skill & current_role_key)
        suggestions, current_role_key, cv_skill_set = (
            _job_title_suggester.suggest_from_cv_text(
                cv_text=request.cv_text,
                top_n=request.top_n_titles,
            )
        )

        elapsed_ms = round((time.time() - start) * 1000, 2)

        logger.info(
            f"CV analyzed: role={profile.current_role}, "
            f"exp={profile.experience}, edu={profile.education}, "
            f"skills={len(cv_skill_set)}, suggestions={len(suggestions)}"
        )

        return AnalyzeCvResponse(
            current_role=profile.current_role,
            experience=profile.experience,
            education=profile.education,
            extracted_skills=sorted(cv_skill_set),
            suggested_job_titles=[
                SuggestedJobTitleItem(
                    title=s.title,
                    role_key=s.role_key,
                    confidence=s.confidence,
                    reason=s.reason,
                    matched_skills=s.matched_skills,
                )
                for s in suggestions
            ],
            analysis_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.error(f"CV analysis error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"CV analysis failed: {str(e)}"
        )


# ==========================================
# Recommend Endpoint  [NEW]
# ==========================================

class JobPosting(BaseModel):
    """Satu job posting dari database backend."""
    job_id:          str = Field(..., description="ID unik job dari database backend")
    job_title:       str = Field(..., description="Judul pekerjaan")
    job_description: str = Field(..., description="Deskripsi lengkap pekerjaan")


class RecommendRequest(BaseModel):
    """Request body untuk rekomendasi job berdasarkan CV + job title pilihan."""
    cv_text: str = Field(
        ...,
        min_length=50,
        max_length=10000,
        description="Teks CV pengguna"
    )
    job_title: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Job title pilihan user (dari suggested_job_titles di /analyze-cv)"
    )
    job_postings: List[JobPosting] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Daftar job posting relevan dari backend (sudah difilter berdasarkan job_title)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "cv_text": "Frontend Developer with 4 years of experience...",
                "job_title": "Full Stack Developer",
                "job_postings": [
                    {
                        "job_id": "job_001",
                        "job_title": "Full Stack Dev at Startup",
                        "job_description": "Looking for React + Node.js developer..."
                    }
                ]
            }]
        }
    }


class RecommendItem(BaseModel):
    """Satu hasil rekomendasi job, sudah diranking."""
    rank:              int   = Field(..., description="Peringkat (1 = paling cocok)")
    job_id:            str   = Field(..., description="ID job dari backend")
    job_title:         str   = Field(..., description="Judul pekerjaan")
    matching_score:    float = Field(..., ge=0.0, le=1.0, description="Skor akhir (hybrid)")
    confidence:        str   = Field(..., description="High / Medium / Low")
    recommendation:    str   = Field(..., description="Teks rekomendasi")
    raw_model_score:   Optional[float] = Field(None, description="Skor dari neural model saja")
    structured_score:  Optional[float] = Field(None, description="Skor structured features")
    inference_time_ms: float = Field(..., description="Waktu inferensi (ms)")


class SkillTierResponse(BaseModel):
    """Hasil analisis satu tier skill (core / common)."""
    required:      List[str] = Field(..., description="Semua skill yang dibutuhkan di tier ini")
    matched:       List[str] = Field(..., description="Skill yang sudah dimiliki kandidat ✅")
    missing:       List[str] = Field(..., description="Skill yang belum dimiliki kandidat ❌")
    readiness_pct: float     = Field(..., description="Persentase kesiapan tier ini (0–100)")


class PriorityGapResponse(BaseModel):
    """Satu skill yang kurang, diurutkan by urgensi."""
    skill:     str   = Field(..., description="Nama skill kategori")
    frequency: float = Field(..., description="Proporsi posting yang menyebut skill ini (0–1)")
    tier:      str   = Field(..., description="'core' atau 'common'")


class IndustrySkillAnalysisResponse(BaseModel):
    """
    Analisis kesiapan skill berbasis job postings nyata.

    Skill dikelompokkan tiga tier berdasarkan frekuensi kemunculan di postings:
    - Core   (≥ 60%) : skill wajib industri, paling urgent jika kurang
    - Common (30-59%): skill penting, bernilai plus jika dimiliki
    - Optional (<30%): nice-to-have, tidak dihitung readiness
    """
    core:     SkillTierResponse = Field(..., description="Tier skill wajib (≥60% postings)")
    common:   SkillTierResponse = Field(..., description="Tier skill penting (30–59% postings)")
    optional: List[str]         = Field(..., description="Skill nice-to-have (<30% postings)")

    readiness_percentage: float = Field(..., description="Readiness keseluruhan (0–100, weighted)")
    readiness_label:      str   = Field(..., description="Label kesiapan: Expert Ready / Almost Ready / dll.")

    priority_gaps: List[PriorityGapResponse] = Field(
        ..., description="Skill yang kurang, diurutkan dari yang paling banyak dicari industri"
    )
    bonus_skills: List[str] = Field(
        ..., description="Skill CV di luar requirement (keunikan kandidat)"
    )

    postings_analyzed: int              = Field(..., description="Jumlah posting yang dijadikan basis analisis")
    skill_frequency:   Dict[str, float] = Field(..., description="Frekuensi tiap skill dari posting (skill → 0–1)")
    high_diversity:    bool             = Field(
        False,
        description=(
            "True jika posting sangat beragam sehingga threshold diturunkan adaptif. "
            "Artinya tidak ada satu pun skill yang mendominasi semua posting."
        )
    )
    effective_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Threshold yang benar-benar dipakai: {'core': 0.4, 'common': 0.2} untuk n kecil"
    )


class RecommendResponse(BaseModel):
    """Response daftar job yang direkomendasikan, diurutkan dari skor tertinggi."""
    job_title_selected: str = Field(..., description="Job title yang dipilih user")
    results:            List[RecommendItem]
    total_items:        int
    total_time_ms:      float
    industry_skill_analysis: Optional[IndustrySkillAnalysisResponse] = Field(
        None,
        description=(
            "Analisis kesiapan skill vs industri nyata (dari job postings). "
            "None jika ekstraksi skill gagal."
        )
    )


@router.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Recommend Jobs: CV + Job Title → Ranked Job Postings",
    description=(
        "**Tahap 2** dari alur rekomendasi. "
        "Menerima CV, job title pilihan user, dan daftar job posting relevan dari backend. "
        "Menjalankan batch prediction (HybridScorer) dan mengembalikan job posting "
        "yang diurutkan dari yang paling cocok."
    ),
    tags=["recommendation"]
)
async def recommend_jobs(request: RecommendRequest):
    """
    Ranking job postings untuk CV berdasarkan job title pilihan.

    **Tahap 2** dari alur rekomendasi 2-tahap:
    1. Backend Express query DB → kirim job_postings ke sini
    2. AI service jalankan batch predict: CV vs setiap job posting
    3. Agregasi skill dari postings → hitung industry skill readiness
    4. Return job_postings terurut + industry skill analysis

    Membutuhkan model DL (SkillAlignPredictor) yang sudah di-load.
    """
    predictor = get_predictor()

    try:
        start = time.time()

        job_descriptions = [jp.job_description for jp in request.job_postings]
        cv_texts         = [request.cv_text] * len(request.job_postings)

        # ── Batch predict (CV vs setiap job posting)
        raw_results = predictor.predict_batch(
            cv_texts=cv_texts,
            job_descriptions=job_descriptions,
        )

        # ── Gabungkan + sort by matching score
        indexed = [
            {"posting": jp, "result": r}
            for jp, r in zip(request.job_postings, raw_results)
        ]
        indexed.sort(key=lambda x: x["result"].matching_score, reverse=True)

        ranked = [
            RecommendItem(
                rank=rank + 1,
                job_id=item["posting"].job_id,
                job_title=item["posting"].job_title,
                matching_score=item["result"].matching_score,
                confidence=item["result"].confidence,
                recommendation=item["result"].recommendation,
                raw_model_score=item["result"].raw_model_score,
                structured_score=item["result"].structured_score,
                inference_time_ms=item["result"].inference_time_ms,
            )
            for rank, item in enumerate(indexed)
        ]

        # ── Industry skill analysis
        # Derive required skills dari frekuensi job_descriptions (bukan ROLE_SKILL_MAP statis)
        industry_analysis_response: Optional[IndustrySkillAnalysisResponse] = None
        try:
            report = _industry_skill_analyzer.analyze(
                cv_text=request.cv_text,
                job_descriptions=job_descriptions,
            )
            industry_analysis_response = IndustrySkillAnalysisResponse(
                core=SkillTierResponse(
                    required=report.core.required,
                    matched=report.core.matched,
                    missing=report.core.missing,
                    readiness_pct=report.core.readiness_pct,
                ),
                common=SkillTierResponse(
                    required=report.common.required,
                    matched=report.common.matched,
                    missing=report.common.missing,
                    readiness_pct=report.common.readiness_pct,
                ),
                optional=report.optional,
                readiness_percentage=report.readiness_percentage,
                readiness_label=report.readiness_label,
                priority_gaps=[
                    PriorityGapResponse(
                        skill=g.skill,
                        frequency=g.frequency,
                        tier=g.tier,
                    )
                    for g in report.priority_gaps
                ],
                bonus_skills=report.bonus_skills,
                postings_analyzed=report.postings_analyzed,
                skill_frequency=report.skill_frequency,
                high_diversity=report.high_diversity,
                effective_thresholds=report.effective_thresholds,
            )
        except Exception as analysis_err:
            logger.warning(f"Industry skill analysis failed (non-fatal): {analysis_err}")

        elapsed_ms = round((time.time() - start) * 1000, 2)

        logger.info(
            f"Recommend: {len(ranked)} jobs ranked for title='{request.job_title}' "
            f"in {elapsed_ms}ms | "
            f"readiness={industry_analysis_response.readiness_percentage if industry_analysis_response else 'N/A'}%"
        )

        return RecommendResponse(
            job_title_selected=request.job_title,
            results=ranked,
            total_items=len(ranked),
            total_time_ms=elapsed_ms,
            industry_skill_analysis=industry_analysis_response,
        )

    except Exception as e:
        logger.error(f"Recommend error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(e)}"
        )
