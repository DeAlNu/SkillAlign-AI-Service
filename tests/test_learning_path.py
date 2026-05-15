"""
Integration tests untuk Learning Path feature.

Pakai FastAPI TestClient + mock external dependencies:
  - CourseFinder (Gemini REST + YouTube API) → mocked
  - Supabase client functions → mocked

Bisa dijalankan offline tanpa API keys.

Cara run:
    cd SkillAlign-AI-Service
    python -m pytest tests/test_learning_path.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from main import app

# ─────────────────────────────────────────────
# Fixture & shared data
# ─────────────────────────────────────────────

CV_TEXT = (
    "Frontend Developer with 4 years of experience. "
    "B.S. Computer Science, Universitas Indonesia. "
    "Skills: React.js, TypeScript, Node.js, PostgreSQL, Docker, "
    "Git, REST API design, responsive UI with Tailwind CSS."
)

PRIORITY_GAPS = [
    {"skill": "TypeScript", "frequency": 0.8, "tier": "core"},
    {"skill": "Docker",     "frequency": 0.5, "tier": "common"},
]

MOCK_COURSES = {
    "TypeScript": {
        "coursera": [
            {"title": "Search 'TypeScript' on Coursera", "url": "https://www.coursera.org/search?query=TypeScript", "level": "Various"}
        ],
        "free_resources": [
            {"title": "TypeScript Roadmap & Learning Guide", "url": "https://roadmap.sh/typescript", "source_type": "roadmap", "level": None},
            {"title": "TypeScript Handbook", "url": "https://www.typescriptlang.org/docs/handbook/intro.html", "source_type": "docs", "level": None},
        ],
        "youtube": [
            {"title": "TypeScript Full Course", "url": "https://www.youtube.com/watch?v=abc123", "thumbnail": "https://i.ytimg.com/vi/abc123/mqdefault.jpg"}
        ],
    },
    "Docker": {
        "coursera": [
            {"title": "Search 'Docker' on Coursera", "url": "https://www.coursera.org/search?query=Docker", "level": "Various"}
        ],
        "free_resources": [
            {"title": "Docker Official Docs", "url": "https://docs.docker.com/get-started/", "source_type": "docs", "level": None},
        ],
        "youtube": [
            {"title": "Docker Tutorial", "url": "https://www.youtube.com/watch?v=xyz789", "thumbnail": "https://i.ytimg.com/vi/xyz789/mqdefault.jpg"}
        ],
    },
}

CACHED_ROWS_TYPESCRIPT_COURSERA = [
    {"title": "TypeScript for Beginners", "url": "https://www.coursera.org/learn/typescript", "level": "Beginner", "platform": "coursera", "thumbnail_url": None}
]


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def analyze_payload():
    return {
        "target_job_title": "Frontend Engineer",
        "priority_gaps": PRIORITY_GAPS,
        "user_id": "test_user_001",
    }


# ─────────────────────────────────────────────
# 1. /analyze — cold cache (semua di-fetch)
# ─────────────────────────────────────────────

class TestAnalyzeColCache:

    @patch("src.inference.learning_path_router.save_session")
    @patch("src.inference.learning_path_router.save_courses")
    @patch("src.inference.learning_path_router.get_cached_courses", return_value=None)
    @patch("src.inference.learning_path_router._get_finder")
    def test_returns_200_with_all_skills(
        self,
        mock_finder_factory,
        mock_cache_get,
        mock_cache_save,
        mock_session,
        client,
        analyze_payload,
    ):
        mock_finder = MagicMock()
        mock_finder.find_courses_for_skills = AsyncMock(return_value=MOCK_COURSES)
        mock_finder_factory.return_value = mock_finder

        resp = client.post("/api/v1/learning-path/analyze", json=analyze_payload)

        assert resp.status_code == 200
        body = resp.json()
        assert body["target_job_title"] == "Frontend Engineer"
        assert body["total_skills"] == 2
        assert body["cached_skills"] == 0
        assert body["fetched_skills"] == 2
        assert len(body["learning_path"]) == 2

    @patch("src.inference.learning_path_router.save_session")
    @patch("src.inference.learning_path_router.save_courses")
    @patch("src.inference.learning_path_router.get_cached_courses", return_value=None)
    @patch("src.inference.learning_path_router._get_finder")
    def test_courses_structure_correct(
        self,
        mock_finder_factory,
        mock_cache_get,
        mock_cache_save,
        mock_session,
        client,
        analyze_payload,
    ):
        mock_finder = MagicMock()
        mock_finder.find_courses_for_skills = AsyncMock(return_value=MOCK_COURSES)
        mock_finder_factory.return_value = mock_finder

        resp = client.post("/api/v1/learning-path/analyze", json=analyze_payload)
        body = resp.json()

        ts_path = next(lp for lp in body["learning_path"] if lp["skill"] == "TypeScript")
        assert ts_path["tier"] == "core"
        assert ts_path["frequency"] == 0.8
        assert "coursera"       in ts_path["courses"]
        assert "free_resources" in ts_path["courses"]
        assert "youtube"        in ts_path["courses"]
        assert len(ts_path["courses"]["coursera"]) == 1
        assert ts_path["courses"]["coursera"][0]["platform"] == "coursera"
        assert len(ts_path["courses"]["free_resources"]) == 2
        assert ts_path["courses"]["free_resources"][0]["source_type"] == "roadmap"

    @patch("src.inference.learning_path_router.save_session")
    @patch("src.inference.learning_path_router.save_courses")
    @patch("src.inference.learning_path_router.get_cached_courses", return_value=None)
    @patch("src.inference.learning_path_router._get_finder")
    def test_order_preserved(
        self,
        mock_finder_factory,
        mock_cache_get,
        mock_cache_save,
        mock_session,
        client,
        analyze_payload,
    ):
        mock_finder = MagicMock()
        mock_finder.find_courses_for_skills = AsyncMock(return_value=MOCK_COURSES)
        mock_finder_factory.return_value = mock_finder

        resp = client.post("/api/v1/learning-path/analyze", json=analyze_payload)
        body = resp.json()

        skills_order = [lp["skill"] for lp in body["learning_path"]]
        assert skills_order == ["TypeScript", "Docker"]

    @patch("src.inference.learning_path_router.save_session")
    @patch("src.inference.learning_path_router.save_courses")
    @patch("src.inference.learning_path_router.get_cached_courses", return_value=None)
    @patch("src.inference.learning_path_router._get_finder")
    def test_save_courses_called(
        self,
        mock_finder_factory,
        mock_cache_get,
        mock_cache_save,
        mock_session,
        client,
        analyze_payload,
    ):
        mock_finder = MagicMock()
        mock_finder.find_courses_for_skills = AsyncMock(return_value=MOCK_COURSES)
        mock_finder_factory.return_value = mock_finder

        client.post("/api/v1/learning-path/analyze", json=analyze_payload)

        assert mock_cache_save.called, "save_courses harus dipanggil untuk caching"


# ─────────────────────────────────────────────
# 2. /analyze — warm cache (semua dari cache)
# ─────────────────────────────────────────────

class TestAnalyzeWarmCache:

    def _mock_cache_hit(self, skill_normalized, platform):
        """Simulasi semua platform cache HIT."""
        return [{"title": "Cached Course", "url": "https://example.com", "level": "All", "platform": platform, "thumbnail_url": None}]

    @patch("src.inference.learning_path_router.save_session")
    @patch("src.inference.learning_path_router._get_finder")
    def test_all_cached_no_fetch(
        self,
        mock_finder_factory,
        mock_session,
        client,
        analyze_payload,
    ):
        mock_finder = MagicMock()
        mock_finder_factory.return_value = mock_finder

        with patch(
            "src.inference.learning_path_router.get_cached_courses",
            side_effect=self._mock_cache_hit,
        ):
            resp = client.post("/api/v1/learning-path/analyze", json=analyze_payload)

        assert resp.status_code == 200
        body = resp.json()
        assert body["cached_skills"] == 2
        assert body["fetched_skills"] == 0
        mock_finder.find_courses_for_skills.assert_not_called()


# ─────────────────────────────────────────────
# 3. /analyze — partial cache (1 cache, 1 fetch)
# ─────────────────────────────────────────────

class TestAnalyzePartialCache:

    def _partial_cache(self, skill_normalized, platform):
        """typescript = HIT, docker = MISS."""
        if "typescript" in skill_normalized:
            return [{"title": "Cached TS", "url": "https://example.com", "level": "All", "platform": platform, "thumbnail_url": None}]
        return None

    @patch("src.inference.learning_path_router.save_session")
    @patch("src.inference.learning_path_router.save_courses")
    @patch("src.inference.learning_path_router._get_finder")
    def test_partial_cache_hit(
        self,
        mock_finder_factory,
        mock_cache_save,
        mock_session,
        client,
        analyze_payload,
    ):
        mock_finder = MagicMock()
        mock_finder.find_courses_for_skills = AsyncMock(return_value={"Docker": MOCK_COURSES["Docker"]})
        mock_finder_factory.return_value = mock_finder

        with patch(
            "src.inference.learning_path_router.get_cached_courses",
            side_effect=self._partial_cache,
        ):
            resp = client.post("/api/v1/learning-path/analyze", json=analyze_payload)

        assert resp.status_code == 200
        body = resp.json()
        assert body["cached_skills"] == 1
        assert body["fetched_skills"] == 1
        mock_finder.find_courses_for_skills.assert_called_once_with(
            skills=["Docker"],
            job_title="Frontend Engineer",
        )


# ─────────────────────────────────────────────
# 4. /analyze — CourseFinder gagal → graceful
# ─────────────────────────────────────────────

class TestAnalyzeFallback:

    @patch("src.inference.learning_path_router.save_session")
    @patch("src.inference.learning_path_router.get_cached_courses", return_value=None)
    @patch("src.inference.learning_path_router._get_finder")
    def test_finder_error_returns_empty_courses(
        self,
        mock_finder_factory,
        mock_cache_get,
        mock_session,
        client,
        analyze_payload,
    ):
        mock_finder = MagicMock()
        mock_finder.find_courses_for_skills = AsyncMock(
            side_effect=RuntimeError("Gemini API timeout")
        )
        mock_finder_factory.return_value = mock_finder

        resp = client.post("/api/v1/learning-path/analyze", json=analyze_payload)

        assert resp.status_code == 200
        body = resp.json()
        for lp in body["learning_path"]:
            for platform in ("coursera", "free_resources", "youtube"):
                assert lp["courses"][platform] == []


# ─────────────────────────────────────────────
# 5. /analyze — validasi request
# ─────────────────────────────────────────────

class TestAnalyzeValidation:

    def test_missing_target_job_title(self, client):
        resp = client.post("/api/v1/learning-path/analyze", json={
            "priority_gaps": PRIORITY_GAPS
        })
        assert resp.status_code == 422

    def test_empty_priority_gaps(self, client):
        resp = client.post("/api/v1/learning-path/analyze", json={
            "target_job_title": "Frontend Engineer",
            "priority_gaps": []
        })
        assert resp.status_code == 422

    def test_invalid_frequency(self, client):
        resp = client.post("/api/v1/learning-path/analyze", json={
            "target_job_title": "Frontend Engineer",
            "priority_gaps": [
                {"skill": "TypeScript", "frequency": "high", "tier": "core"}
            ]
        })
        assert resp.status_code == 422


# ─────────────────────────────────────────────
# 6. POST /refresh
# ─────────────────────────────────────────────

class TestRefresh:

    @patch("src.inference.learning_path_router.delete_courses_by_skill")
    def test_refresh_returns_immediately(self, mock_delete, client):
        payload = {
            "target_job_title": "Frontend Engineer",
            "priority_gaps": PRIORITY_GAPS,
            "skills": ["TypeScript"],
        }
        resp = client.post("/api/v1/learning-path/refresh", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        assert "TypeScript" in body["skills_refreshing"]
        assert "message" in body

    @patch("src.inference.learning_path_router.delete_courses_by_skill")
    def test_refresh_deletes_cache(self, mock_delete, client):
        payload = {
            "target_job_title": "Frontend Engineer",
            "priority_gaps": PRIORITY_GAPS,
            "skills": ["TypeScript"],
            "platforms": ["coursera"],
        }
        client.post("/api/v1/learning-path/refresh", json=payload)

        mock_delete.assert_called()
        call_args = mock_delete.call_args_list
        assert any("typescript" in str(call) for call in call_args)

    @patch("src.inference.learning_path_router.delete_courses_by_skill")
    def test_refresh_specific_platforms(self, mock_delete, client):
        payload = {
            "target_job_title": "Frontend Engineer",
            "priority_gaps": PRIORITY_GAPS,
            "skills": ["Docker"],
            "platforms": ["youtube"],
        }
        resp = client.post("/api/v1/learning-path/refresh", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        assert body["platforms"] == ["youtube"]


# ─────────────────────────────────────────────
# 7. GET /courses/{skill_name}
# ─────────────────────────────────────────────

class TestGetCourses:

    @patch("src.inference.learning_path_router.get_cached_courses")
    def test_found_in_cache(self, mock_cache, client):
        def cached(norm, platform):
            if platform == "coursera":
                return CACHED_ROWS_TYPESCRIPT_COURSERA
            return []

        mock_cache.side_effect = cached

        resp = client.get("/api/v1/learning-path/courses/TypeScript")

        assert resp.status_code == 200
        body = resp.json()
        assert body["skill"] == "TypeScript"
        assert len(body["courses"]["coursera"]) == 1
        assert body["courses"]["coursera"][0]["title"] == "TypeScript for Beginners"

    @patch("src.inference.learning_path_router.get_cached_courses", return_value=None)
    def test_not_cached_returns_404(self, mock_cache, client):
        resp = client.get("/api/v1/learning-path/courses/UnknownSkill")
        assert resp.status_code == 404

    @patch("src.inference.learning_path_router.get_cached_courses", return_value=[])
    def test_empty_cache_returns_404(self, mock_cache, client):
        resp = client.get("/api/v1/learning-path/courses/EmptySkill")
        assert resp.status_code == 404


# ─────────────────────────────────────────────
# 8. Normalize helper (internal)
# ─────────────────────────────────────────────

class TestNormalize:

    def test_normalize_cases(self):
        from src.inference.learning_path_router import _normalize

        assert _normalize("TypeScript") == "typescript"
        assert _normalize("Machine Learning") == "machine_learning"
        assert _normalize("  react.js  ") == "react.js"
        assert _normalize("CI/CD") == "ci/cd"
