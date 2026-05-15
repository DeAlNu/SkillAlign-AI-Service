"""
Supabase Client untuk SkillAlign Learning Path.

Menyediakan fungsi CRUD untuk tabel skill_courses dan learning_path_sessions.
Menggunakan service_role key sehingga bisa bypass RLS.
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from supabase import create_client, Client

logger = logging.getLogger(__name__)

CACHE_TTL_DAYS = 30

_client: Optional[Client] = None


def get_supabase() -> Client:
    """Singleton Supabase client."""
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL dan SUPABASE_SERVICE_ROLE_KEY harus di-set di .env"
            )
        _client = create_client(url, key)
        logger.info("Supabase client initialized.")
    return _client


# ──────────────────────────────────────────
# skill_courses — Cache CRUD
# ──────────────────────────────────────────

def get_cached_courses(skill_normalized: str, platform: str) -> Optional[List[Dict]]:
    """
    Ambil courses dari cache Supabase jika ada dan belum expired (< 30 hari).

    Dua-tahap:
    1. Cek apakah platform ini pernah di-search (row apapun, termasuk is_valid=False).
       Jika belum pernah → return None (trigger re-fetch).
    2. Jika sudah pernah → return hanya kursus valid (bisa [] kalau memang kosong).

    Ini penting agar platform yang hasilnya kosong (misal Dicoding tidak punya kursus
    untuk skill tertentu) tidak terus-menerus di-fetch ulang setiap request.
    """
    try:
        db = get_supabase()
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=CACHE_TTL_DAYS)
        ).isoformat()

        # Tahap 1 — apakah platform ini pernah di-search dalam TTL?
        searched = (
            db.table("skill_courses")
            .select("id")
            .eq("skill_normalized", skill_normalized)
            .eq("platform", platform)
            .gte("created_at", cutoff)
            .limit(1)
            .execute()
        )
        if not searched.data:
            logger.debug(f"Cache MISS (not searched): {skill_normalized}/{platform}")
            return None

        # Tahap 2 — ambil kursus valid saja (boleh kosong [])
        result = (
            db.table("skill_courses")
            .select("title, url, thumbnail_url, level, platform")
            .eq("skill_normalized", skill_normalized)
            .eq("platform", platform)
            .eq("is_valid", True)
            .gte("created_at", cutoff)
            .execute()
        )

        logger.debug(
            f"Cache HIT: {skill_normalized}/{platform} ({len(result.data)} courses)"
        )
        return result.data  # [] kalau kosong — bukan None

    except Exception as e:
        logger.warning(f"Cache read error [{skill_normalized}/{platform}]: {e}")
        return None


def save_courses(courses: List[Dict]) -> None:
    """
    Simpan courses ke Supabase. Pakai upsert untuk handle duplikat URL.

    Args:
        courses: List of dicts dengan key:
                 skill_normalized, platform, title, url,
                 thumbnail_url (opsional), level (opsional)
    """
    if not courses:
        return
    try:
        db = get_supabase()
        db.table("skill_courses").upsert(
            courses,
            on_conflict="skill_normalized,platform,url"
        ).execute()
        logger.info(f"Saved {len(courses)} courses to Supabase cache.")
    except Exception as e:
        logger.warning(f"Cache write error: {e}")


def delete_courses_by_skill(
    skill_normalized: str,
    platform: Optional[str] = None
) -> None:
    """
    Hapus cache courses untuk skill tertentu (dipakai saat refresh).

    Args:
        skill_normalized: Skill yang cache-nya ingin dihapus.
        platform:         Jika None, hapus semua platform.
    """
    try:
        db = get_supabase()
        query = (
            db.table("skill_courses")
            .delete()
            .eq("skill_normalized", skill_normalized)
        )
        if platform:
            query = query.eq("platform", platform)
        query.execute()
        logger.info(
            f"Cache deleted: {skill_normalized}/{platform or 'all platforms'}"
        )
    except Exception as e:
        logger.warning(f"Cache delete error [{skill_normalized}]: {e}")


# ──────────────────────────────────────────
# learning_path_sessions — Session logging
# ──────────────────────────────────────────

def save_session(
    user_id: Optional[str],
    target_job_title: str,
    missing_skills: List[Dict],
) -> Optional[str]:
    """
    Simpan session learning path ke Supabase.

    Returns:
        Session ID (UUID string) atau None jika gagal.
    """
    try:
        db = get_supabase()
        result = (
            db.table("learning_path_sessions")
            .insert({
                "user_id": user_id,
                "target_job_title": target_job_title,
                "missing_skills": missing_skills,
            })
            .execute()
        )
        if result.data:
            session_id = result.data[0]["id"]
            logger.info(f"Session saved: {session_id}")
            return session_id
    except Exception as e:
        logger.warning(f"Session save error: {e}")
    return None
