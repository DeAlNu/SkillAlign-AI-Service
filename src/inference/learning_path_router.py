"""
Learning Path Router — SkillAlign AI.

Endpoint:
  POST /api/v1/learning-path/analyze   → dari priority_gaps, cari & return courses
  POST /api/v1/learning-path/refresh   → reset cache + fetch ulang (background)
  GET  /api/v1/learning-path/courses/{skill_name} → ambil courses dari cache
"""

import logging
import time
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from src.database.supabase_client import (
    delete_courses_by_skill,
    get_cached_courses,
    save_courses,
    save_session,
)
from src.inference.course_finder import CourseFinder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/learning-path", tags=["learning-path"])

_PLATFORMS = ("coursera", "free_resources", "youtube")

# Singleton — dibuat sekali saat pertama dipakai
_course_finder: Optional[CourseFinder] = None


def _get_finder() -> CourseFinder:
    global _course_finder
    if _course_finder is None:
        _course_finder = CourseFinder()
    return _course_finder


def _normalize(skill: str) -> str:
    return skill.lower().strip().replace(" ", "_")


# ══════════════════════════════════════════
# Schemas
# ══════════════════════════════════════════

class PriorityGapInput(BaseModel):
    """Satu skill gap dari output /recommend → industry_skill_analysis.priority_gaps."""
    skill:     str   = Field(..., description="Nama skill kategori")
    frequency: float = Field(..., description="Proporsi posting yang menyebut skill (0–1)")
    tier:      str   = Field(..., description="'core' atau 'common'")


class CourseItem(BaseModel):
    title:         str            = Field(..., description="Judul kursus / resource")
    url:           str            = Field(..., description="URL langsung ke resource")
    level:         Optional[str]  = Field(None, description="Level kesulitan")
    thumbnail_url: Optional[str]  = Field(None, description="Thumbnail (YouTube)")
    platform:      str            = Field(..., description="coursera | free_resources | youtube")
    source_type:   Optional[str]  = Field(None, description="Tipe resource: docs | roadmap | github | free_course")


class SkillLearningPath(BaseModel):
    skill:     str                          = Field(..., description="Nama skill")
    tier:      str                          = Field(..., description="core | common")
    frequency: float                        = Field(..., description="Frekuensi di industri")
    courses:   Dict[str, List[CourseItem]]  = Field(
        ..., description="{ 'coursera': [...], 'free_resources': [...], 'youtube': [...] }"
    )


class AnalyzeLearningPathRequest(BaseModel):
    target_job_title: str = Field(
        ..., description="Job title target (dari /recommend hasil pilih user)"
    )
    priority_gaps: List[PriorityGapInput] = Field(
        ...,
        min_length=1,
        description="Dari industry_skill_analysis.priority_gaps di response /recommend"
    )
    user_id: Optional[str] = Field(None, description="ID user (opsional, untuk logging sesi)")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "target_job_title": "Frontend Engineer",
                "priority_gaps": [
                    {"skill": "typescript", "frequency": 0.8, "tier": "core"},
                    {"skill": "docker",     "frequency": 0.4, "tier": "common"},
                ],
                "user_id": "user_123"
            }]
        }
    }


class LearningPathResponse(BaseModel):
    target_job_title: str
    learning_path:    List[SkillLearningPath]
    total_skills:     int   = Field(..., description="Total skill yang diproses")
    cached_skills:    int   = Field(..., description="Skill yang diambil dari cache")
    fetched_skills:   int   = Field(..., description="Skill yang baru di-fetch (Gemini + YouTube)")
    total_time_ms:    float


class RefreshRequest(BaseModel):
    target_job_title: str                  = Field(..., description="Job title untuk konteks re-fetch")
    priority_gaps:    List[PriorityGapInput] = Field(..., description="Priority gaps dari /recommend")
    skills:           List[str]            = Field(..., description="Skill yang ingin di-refresh")
    platforms:        Optional[List[str]]  = Field(
        None,
        description="Platform yang di-refresh. None = semua (coursera, free_resources, youtube)"
    )


# ══════════════════════════════════════════
# Helper
# ══════════════════════════════════════════

def _infer_source_type(url: str) -> Optional[str]:
    """Infer tipe resource dari URL — dipakai saat membaca dari cache (tidak ada source_type)."""
    if not url:
        return None
    if "roadmap.sh" in url:
        return "roadmap"
    if "github.com" in url:
        return "github"
    if any(d in url for d in (
        "freecodecamp.org", "theodinproject.com",
        "training.linuxfoundation.org", "edx.org",
    )):
        return "free_course"
    if any(x in url for x in ("coursera.org", "youtube.com", "youtu.be")):
        return None  # platform ini tidak perlu source_type
    return "docs"


def _build_skill_path(
    gap: PriorityGapInput,
    raw_courses: Dict[str, List[Dict]],
) -> SkillLearningPath:
    """Konversi raw dict courses → SkillLearningPath Pydantic model."""
    platform_courses: Dict[str, List[CourseItem]] = {}
    for platform in _PLATFORMS:
        items = raw_courses.get(platform, [])
        platform_courses[platform] = [
            CourseItem(
                title=c.get("title", ""),
                url=c.get("url", ""),
                level=c.get("level"),
                thumbnail_url=c.get("thumbnail"),
                platform=platform,
                # source_type: dari CourseFinder langsung, atau di-infer dari URL saat dari cache
                source_type=(
                    c.get("source_type") or _infer_source_type(c.get("url", ""))
                    if platform == "free_resources" else None
                ),
            )
            for c in items
            if c.get("url") and c.get("title")
        ]
    return SkillLearningPath(
        skill=gap.skill,
        tier=gap.tier,
        frequency=gap.frequency,
        courses=platform_courses,
    )


def _courses_to_db_rows(
    skill_normalized: str, platform: str, course_list: List[Dict]
) -> List[Dict]:
    """Format courses untuk di-upsert ke Supabase."""
    return [
        {
            "skill_normalized": skill_normalized,
            "platform":         platform,
            "title":            c.get("title", ""),
            "url":              c.get("url", ""),
            "thumbnail_url":    c.get("thumbnail"),
            # Untuk free_resources: simpan source_type di kolom level (workaround — DB tidak
            # punya kolom source_type). Saat dibaca ulang, _infer_source_type() di-infer dari URL.
            "level":            c.get("level"),
        }
        for c in course_list
        if c.get("url") and c.get("title")
    ]


# ══════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════

@router.post(
    "/analyze",
    response_model=LearningPathResponse,
    summary="Generate Learning Path from Priority Gaps",
    description=(
        "Terima priority_gaps dari /recommend, cari kursus di Coursera/Free Resources/YouTube, "
        "cache hasilnya di Supabase (30 hari). "
        "Skill yang sudah ada di cache dikembalikan langsung tanpa memanggil API eksternal."
    ),
)
async def analyze_learning_path(request: AnalyzeLearningPathRequest):
    start = time.time()
    finder = _get_finder()

    learning_path: List[SkillLearningPath] = []
    skills_to_fetch: List[PriorityGapInput] = []
    cached_count = 0

    # ── 1. Pisahkan: skill yang sudah di-cache vs yang perlu di-fetch
    for gap in request.priority_gaps:
        norm = _normalize(gap.skill)
        cached_by_platform: Dict[str, List[Dict]] = {}
        fully_cached = True

        for platform in _PLATFORMS:
            rows = get_cached_courses(norm, platform)
            if rows is not None:
                cached_by_platform[platform] = rows
            else:
                fully_cached = False

        if fully_cached:
            learning_path.append(_build_skill_path(gap, cached_by_platform))
            cached_count += 1
        else:
            skills_to_fetch.append(gap)

    # ── 2. Fetch skills yang belum di-cache
    if skills_to_fetch:
        skill_names = [g.skill for g in skills_to_fetch]
        try:
            fetched = await finder.find_courses_for_skills(
                skills=skill_names,
                job_title=request.target_job_title,
            )

            for gap in skills_to_fetch:
                norm = _normalize(gap.skill)
                courses_by_platform = fetched.get(gap.skill, {})

                # Simpan ke Supabase cache per platform.
                # Platform yang hasilnya kosong tetap disimpan sebagai penanda
                # "sudah pernah di-search" (is_valid=False) agar warm cache tidak
                # me-refetch platform tersebut pada request berikutnya.
                for platform in _PLATFORMS:
                    rows = _courses_to_db_rows(
                        norm, platform, courses_by_platform.get(platform, [])
                    )
                    if rows:
                        save_courses(rows)
                    else:
                        # Simpan placeholder agar cache tahu platform ini sudah pernah di-search
                        save_courses([{
                            "skill_normalized": norm,
                            "platform":         platform,
                            "title":            "_searched_no_results",
                            "url":              "_placeholder",
                            "thumbnail_url":    None,
                            "level":            None,
                            "is_valid":         False,
                        }])

                learning_path.append(_build_skill_path(gap, courses_by_platform))

        except Exception as e:
            logger.error(f"CourseFinder error: {e}")
            # Non-blocking: return skill tanpa courses daripada error 500
            for gap in skills_to_fetch:
                learning_path.append(SkillLearningPath(
                    skill=gap.skill, tier=gap.tier, frequency=gap.frequency,
                    courses={p: [] for p in _PLATFORMS},
                ))

    # ── 3. Simpan session (non-blocking, best-effort)
    save_session(
        user_id=request.user_id,
        target_job_title=request.target_job_title,
        missing_skills=[g.model_dump() for g in request.priority_gaps],
    )

    # ── 4. Urutkan kembali sesuai urutan priority_gaps asli
    order = {g.skill: i for i, g in enumerate(request.priority_gaps)}
    learning_path.sort(key=lambda x: order.get(x.skill, 999))

    elapsed = round((time.time() - start) * 1000, 2)
    logger.info(
        f"Learning path: {len(request.priority_gaps)} skills, "
        f"{cached_count} cached, {len(skills_to_fetch)} fetched, {elapsed}ms"
    )

    return LearningPathResponse(
        target_job_title=request.target_job_title,
        learning_path=learning_path,
        total_skills=len(request.priority_gaps),
        cached_skills=cached_count,
        fetched_skills=len(skills_to_fetch),
        total_time_ms=elapsed,
    )


@router.post(
    "/refresh",
    summary="Refresh Course Cache for Specific Skills",
    description=(
        "Hapus cache skill yang dipilih lalu fetch ulang di background. "
        "Response langsung dikembalikan — proses fetch berjalan di belakang layar."
    ),
)
async def refresh_courses(
    request: RefreshRequest,
    background_tasks: BackgroundTasks,
):
    platforms_to_clear = request.platforms or list(_PLATFORMS)

    # ── Hapus cache sekarang (synchronous, cepat)
    for skill in request.skills:
        norm = _normalize(skill)
        for platform in platforms_to_clear:
            delete_courses_by_skill(norm, platform)

    # ── Filter priority_gaps hanya untuk skill yang di-refresh
    gaps_to_refresh = [
        g for g in request.priority_gaps if g.skill in request.skills
    ]

    # ── Fetch ulang di background (tidak memblokir response)
    async def _do_refresh() -> None:
        finder = _get_finder()
        skill_names = [g.skill for g in gaps_to_refresh]
        try:
            fetched = await finder.find_courses_for_skills(
                skills=skill_names,
                job_title=request.target_job_title,
            )
            for gap in gaps_to_refresh:
                norm = _normalize(gap.skill)
                for platform in platforms_to_clear:
                    rows = _courses_to_db_rows(
                        norm, platform,
                        fetched.get(gap.skill, {}).get(platform, [])
                    )
                    if rows:
                        save_courses(rows)
            logger.info(f"Background refresh done: {skill_names}")
        except Exception as e:
            logger.error(f"Background refresh failed: {e}")

    background_tasks.add_task(_do_refresh)

    return {
        "message": (
            f"Cache cleared for {len(request.skills)} skill(s). "
            "Fetching new courses in background."
        ),
        "skills_refreshing": request.skills,
        "platforms":         platforms_to_clear,
    }


@router.get(
    "/courses/{skill_name}",
    summary="Get Cached Courses for a Skill",
    description=(
        "Ambil courses yang sudah di-cache untuk satu skill. "
        "Jika belum ada, panggil /analyze terlebih dahulu."
    ),
)
async def get_skill_courses(skill_name: str):
    norm = _normalize(skill_name)
    result: Dict[str, List[Dict]] = {}

    for platform in _PLATFORMS:
        cached = get_cached_courses(norm, platform)
        rows = cached or []
        # Inject source_type untuk free_resources (tidak disimpan di DB, di-infer dari URL)
        if platform == "free_resources":
            rows = [
                {**r, "source_type": _infer_source_type(r.get("url", ""))}
                for r in rows
            ]
        result[platform] = rows

    if not any(result.values()):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Belum ada courses untuk '{skill_name}'. "
                "Panggil POST /api/v1/learning-path/analyze terlebih dahulu."
            ),
        )

    return {"skill": skill_name, "courses": result}
