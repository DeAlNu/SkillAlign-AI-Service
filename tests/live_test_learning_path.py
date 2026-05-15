"""
Live integration test untuk Learning Path endpoints.

Butuh server running: python -m uvicorn main:app --host 0.0.0.0 --port 8000

Cara run:
    python tests/live_test_learning_path.py
"""

import json
import sys
import httpx

BASE_URL = "http://localhost:8000"

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


def _ok(label: str):
    print(f"  [PASS] {label}")

def _fail(label: str, detail: str = ""):
    print(f"  [FAIL] {label}" + (f" — {detail}" if detail else ""))

def _section(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def check_server():
    _section("0. Health Check")
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
        data = r.json()
        _ok(f"Server up | model_loaded={data.get('model_loaded')}")
        return True
    except Exception as e:
        _fail("Server tidak bisa dijangkau", str(e))
        print("\n  Jalankan dulu: python -m uvicorn main:app --host 0.0.0.0 --port 8000")
        return False


def test_analyze_cold():
    _section("1. POST /analyze — Cold Cache (Gemini + YouTube real call)")
    payload = {
        "target_job_title": "Frontend Engineer",
        "priority_gaps": PRIORITY_GAPS,
        "user_id": "live_test_001",
    }

    print("  Memanggil Gemini + YouTube API ... (bisa 5-20 detik)")
    try:
        r = httpx.post(
            f"{BASE_URL}/api/v1/learning-path/analyze",
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
    except Exception as e:
        _fail("Request gagal", str(e))
        return None

    body = r.json()

    # Cek struktur utama
    assert body["target_job_title"] == "Frontend Engineer", "target_job_title salah"
    assert body["total_skills"] == 2, "total_skills harus 2"
    _ok(f"Response 200 | total_skills={body['total_skills']} | time={body['total_time_ms']}ms")
    _ok(f"Cache stats: cached={body['cached_skills']}, fetched={body['fetched_skills']}")

    # Cek setiap skill
    for lp in body["learning_path"]:
        skill          = lp["skill"]
        coursera       = lp["courses"].get("coursera", [])
        free_resources = lp["courses"].get("free_resources", [])
        youtube        = lp["courses"].get("youtube", [])
        total          = len(coursera) + len(free_resources) + len(youtube)

        _ok(
            f"Skill '{skill}': coursera={len(coursera)}, "
            f"free_resources={len(free_resources)}, youtube={len(youtube)}"
        )

        if total == 0:
            _fail(f"  Skill '{skill}' tidak punya resource sama sekali")
        else:
            for platform, items in [
                ("coursera", coursera),
                ("free_resources", free_resources),
                ("youtube", youtube),
            ]:
                if items:
                    first = items[0]
                    st = f" [{first.get('source_type')}]" if first.get("source_type") else ""
                    print(f"       {platform}{st}: \"{first['title']}\"")
                    print(f"               {first['url']}")

    return body


def test_analyze_warm(first_body):
    _section("2. POST /analyze — Warm Cache (tidak re-call Gemini)")
    if first_body is None:
        print("  Skip (test sebelumnya gagal)")
        return

    payload = {
        "target_job_title": "Frontend Engineer",
        "priority_gaps": PRIORITY_GAPS,
        "user_id": "live_test_001",
    }

    print("  Memanggil ulang endpoint yang sama ...")
    try:
        r = httpx.post(
            f"{BASE_URL}/api/v1/learning-path/analyze",
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
    except Exception as e:
        _fail("Request gagal", str(e))
        return

    body = r.json()
    cached = body["cached_skills"]
    fetched = body["fetched_skills"]

    if cached == 2 and fetched == 0:
        _ok(f"Cache HIT sempurna: cached={cached}, fetched={fetched}")
        _ok(f"Response time: {body['total_time_ms']}ms (harusnya jauh lebih cepat)")
    elif cached > 0:
        _ok(f"Partial cache: cached={cached}, fetched={fetched}")
    else:
        _fail("Cache tidak bekerja — semua skill masih di-fetch ulang")


def test_get_courses():
    _section("3. GET /courses/TypeScript — Ambil dari cache")
    try:
        r = httpx.get(
            f"{BASE_URL}/api/v1/learning-path/courses/TypeScript",
            timeout=10,
        )
    except Exception as e:
        _fail("Request gagal", str(e))
        return

    if r.status_code == 200:
        body = r.json()
        total = sum(len(v) for v in body["courses"].values())
        _ok(f"Found {total} courses cached untuk 'TypeScript'")
    elif r.status_code == 404:
        _fail("404 — cache belum ada, pastikan test /analyze dijalankan dulu")
    else:
        _fail(f"Status {r.status_code}", r.text[:200])


def test_refresh():
    _section("4. POST /refresh — Reset cache TypeScript")
    payload = {
        "target_job_title": "Frontend Engineer",
        "priority_gaps": PRIORITY_GAPS,
        "skills": ["TypeScript"],
        "platforms": ["coursera"],
    }
    try:
        r = httpx.post(
            f"{BASE_URL}/api/v1/learning-path/refresh",
            json=payload,
            timeout=10,
        )
        r.raise_for_status()
        body = r.json()
        _ok(f"Refresh triggered: {body['message']}")
        _ok(f"Skills refreshing: {body['skills_refreshing']}")
        _ok(f"Platforms: {body['platforms']}")
    except Exception as e:
        _fail("Request gagal", str(e))


def test_404_unknown_skill():
    _section("5. GET /courses/SkillYangTidakAda — Harusnya 404")
    try:
        r = httpx.get(
            f"{BASE_URL}/api/v1/learning-path/courses/SkillYangTidakAda123",
            timeout=5,
        )
        if r.status_code == 404:
            _ok("404 dikembalikan dengan benar")
        else:
            _fail(f"Harusnya 404, dapat {r.status_code}")
    except Exception as e:
        _fail("Request gagal", str(e))


if __name__ == "__main__":
    print("\nSkillAlign Learning Path — Live Test")
    print("Target:", BASE_URL)

    if not check_server():
        sys.exit(1)

    first_body = test_analyze_cold()
    test_analyze_warm(first_body)
    test_get_courses()
    test_refresh()
    test_404_unknown_skill()

    print(f"\n{'='*55}")
    print("  Live test selesai.")
    print(f"{'='*55}\n")
