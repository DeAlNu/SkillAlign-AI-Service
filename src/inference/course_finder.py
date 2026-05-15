"""
Course Finder untuk SkillAlign Learning Path.

Menggunakan Gemini REST API secara langsung (aiohttp) — TANPA google-generativeai SDK
agar tidak konflik dengan protobuf yang dibutuhkan TensorFlow 2.21.

Platform yang didukung:
  - coursera       : Selalu mengembalikan search page (reliable, tidak ada URL-slug lookup)
  - free_resources : Gemini + Search Grounding mengkurasi dokumentasi resmi,
                     roadmap.sh, free courses, GitHub repos; divalidasi via GET
  - youtube        : YouTube Data API v3 + Gemini re-ranking

Kenapa tidak pakai URL spesifik Coursera lagi?
  - Gemini kadang mengonstruksi slug yang plausible tapi salah (404)
  - HEAD request diblokir Coursera; GET validation menambah latency signifikan
  - Search page (coursera.org/search?query=SKILL) selalu ada dan langsung berguna

Kenapa Dicoding diganti free_resources?
  - Dicoding menggunakan numeric ID (/academies/NUMBER) yang tidak bisa ditemukan
    secara reliabel via Google Search
  - Dicoding selalu mengembalikan HTTP 200 bahkan untuk ID yang tidak ada
  - free_resources (docs resmi, roadmap.sh, freeCodeCamp) jauh lebih reliable
    dan mencakup skill apapun
"""

import asyncio
import json
import logging
import os
import re
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import aiohttp

logger = logging.getLogger(__name__)

# ── Gemini REST
_GEMINI_BASE  = "https://generativelanguage.googleapis.com/v1beta/models"
_GEMINI_MODEL = "gemini-2.5-flash"
_GENERATE_URL = f"{_GEMINI_BASE}/{_GEMINI_MODEL}:generateContent"

# ── YouTube
_YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
_YT_FETCH = 10
_YT_TOP   = 5

_MAX_PER_PLATFORM = 3
_MAX_FREE         = 3          # max free_resources per skill
_HTTP_TIMEOUT     = aiohttp.ClientTimeout(total=60)
_VALIDATE_TIMEOUT = aiohttp.ClientTimeout(total=10)

# Batasi concurrent Gemini calls — mencegah 429 rate limit tanpa mengorbankan hasil
_GEMINI_SEMAPHORE = asyncio.Semaphore(5)

# Regex: pasangan "Judul" https://url dari bullet list Gemini
_RE_RESOURCE = re.compile(
    r'"([^"]{5,120})"\s+(https?://[^\s\)\]\'"<>\n,]+)'
)
# Fallback: URL polos tanpa judul
_RE_URL_BROAD = re.compile(r'https?://[^\s\)\]\'"<>\n,]+')

# Domain yang dikecualikan dari free_resources
_SOCIAL_DOMAINS = frozenset({
    'twitter.com', 'x.com', 'facebook.com', 'instagram.com',
    'linkedin.com', 'reddit.com', 'quora.com', 'medium.com',
    'dev.to', 'hackernoon.com', 'tiktok.com',
})


class CourseFinder:
    """
    Mencari kursus / resource belajar untuk daftar skill yang diberikan.
    Semua HTTP calls berjalan async. Singleton-safe (stateless setelah init).
    """

    def __init__(self) -> None:
        self._gemini_key = os.getenv("GEMINI_API_KEY")
        self._yt_key     = os.getenv("YOUTUBE_API_KEY")
        if not self._gemini_key:
            raise RuntimeError("GEMINI_API_KEY tidak ditemukan di environment.")
        logger.info(f"CourseFinder initialized (model={_GEMINI_MODEL})")

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    async def find_courses_for_skills(
        self,
        skills: List[str],
        job_title: str,
    ) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Cari resource belajar untuk beberapa skill sekaligus (batch, paralel).

        Returns:
            {
              "TypeScript": {
                "coursera":       [{"title": ..., "url": ...}],
                "free_resources": [{"title": ..., "url": ..., "source_type": ...}],
                "youtube":        [{"title": ..., "url": ..., "thumbnail": ...}],
              }, ...
            }
        """
        # Free resources dan YouTube dijalankan paralel per skill
        free_tasks = [self._find_free_resources(s, job_title) for s in skills]
        yt_tasks   = [self._find_youtube_courses(s, job_title) for s in skills]

        all_results = await asyncio.gather(*free_tasks, *yt_tasks, return_exceptions=True)
        free_results = all_results[:len(skills)]
        yt_results   = all_results[len(skills):]

        results: Dict[str, Dict[str, List[Dict]]] = {}
        for i, skill in enumerate(skills):
            results[skill] = {
                # Coursera: selalu search page — reliable, tidak butuh Gemini call
                "coursera": self._coursera_search_url(skill),
                "free_resources": (
                    free_results[i]
                    if not isinstance(free_results[i], Exception)
                    else []
                ),
                "youtube": (
                    yt_results[i]
                    if not isinstance(yt_results[i], Exception)
                    else []
                ),
            }
            if isinstance(free_results[i], Exception):
                logger.warning(f"free_resources failed for '{skill}': {free_results[i]}")
            if isinstance(yt_results[i], Exception):
                logger.warning(f"YouTube failed for '{skill}': {yt_results[i]}")

        return results

    # ──────────────────────────────────────────────────────────────────
    # Free Resources: Gemini Search Grounding + GET validation
    # ──────────────────────────────────────────────────────────────────

    async def _find_free_resources(
        self,
        skill: str,
        job_title: str,
    ) -> List[Dict]:
        """
        Temukan resource belajar gratis berkualitas untuk satu skill.

        Alur:
        1. Cek roadmap.sh/{slug} secara paralel dengan Gemini call
        2. Gemini + Search Grounding → daftar resource terbaik (plain text)
        3. Regex mengekstrak pasangan judul+URL
        4. GET validation memfilter URL yang tidak accessible
        5. Gabungkan: roadmap.sh (jika ada) + hasil Gemini tervalidasi
        """
        slug = re.sub(r'[^a-z0-9]+', '-', skill.lower()).strip('-')
        roadmap_url = f"https://roadmap.sh/{slug}"

        prompt = f"""Search Google for the best FREE learning resources for "{skill}" \
targeting the role of "{job_title}".

Search for:
1. "{skill} official documentation"
2. "{skill} free course freeCodeCamp OR Linux Foundation OR edX OR The Odin Project"
3. "{skill} tutorial github"

List the 3 best resources you found with their EXACT URLs.

Format your response as:
- "Resource Title" https://exact-url.com/page
- "Resource Title" https://exact-url.com/page
- "Resource Title" https://exact-url.com/page

Rules:
- Official documentation first (e.g. kubernetes.io/docs, typescriptlang.org/docs)
- Then free courses, then GitHub tutorials
- Never include Medium, Reddit, social media, or paid-only courses
- Only use URLs found via actual Google Search"""

        # Jalankan Gemini call dan roadmap.sh check secara paralel
        gemini_coro  = self._call_gemini(prompt, use_search_grounding=True)
        roadmap_coro = self._url_exists_get(roadmap_url)

        gemini_result, roadmap_ok = await asyncio.gather(
            gemini_coro, roadmap_coro, return_exceptions=True
        )

        resources: List[Dict] = []

        # ── roadmap.sh (prioritas pertama jika ada)
        if roadmap_ok is True:
            resources.append({
                "title":       f"{skill} Roadmap & Learning Guide",
                "url":         roadmap_url,
                "source_type": "roadmap",
                "level":       None,
            })
            logger.info(f"[{skill}] roadmap.sh found: {roadmap_url}")

        # ── Gemini candidates
        if isinstance(gemini_result, Exception):
            logger.warning(f"[{skill}] Gemini free_resources failed: {gemini_result}")
        else:
            raw = gemini_result
            logger.debug(f"[{skill}] free_resources raw ({len(raw)} chars): {raw[:400]}")
            candidates = _extract_resource_items(raw)

            # Filter domain yang tidak diinginkan dan duplikat dengan roadmap
            candidates = [
                c for c in candidates
                if not any(d in c["url"] for d in _SOCIAL_DOMAINS)
                and c["url"] != roadmap_url
            ]

            # Validasi URL via GET (docs resmi mengembalikan 200/404 dengan benar)
            remaining = _MAX_FREE - len(resources)
            if candidates and remaining > 0:
                # Check lebih banyak kandidat dari yang dibutuhkan untuk ada cadangan
                to_check = candidates[: remaining * 2]
                validations = await asyncio.gather(
                    *[self._url_exists_get(c["url"]) for c in to_check],
                    return_exceptions=True,
                )
                for c, ok in zip(to_check, validations):
                    if ok is True and len(resources) < _MAX_FREE:
                        resources.append({
                            "title":       c["title"],
                            "url":         c["url"],
                            "source_type": _classify_source(c["url"]),
                            "level":       None,
                        })
                        logger.info(f"[{skill}] free resource validated: {c['url']}")
                    elif ok is not True:
                        logger.debug(f"[{skill}] free resource invalid: {c['url']}")

        return resources

    # ──────────────────────────────────────────────────────────────────
    # Coursera — selalu search page (tidak ada URL-slug lookup)
    # ──────────────────────────────────────────────────────────────────

    def _coursera_search_url(self, skill: str) -> List[Dict]:
        return [{
            "title": f"Search '{skill}' on Coursera",
            "url":   f"https://www.coursera.org/search?query={quote_plus(skill)}",
            "level": "Various",
        }]

    # ──────────────────────────────────────────────────────────────────
    # URL Validation via GET
    # ──────────────────────────────────────────────────────────────────

    async def _url_exists_get(self, url: str) -> bool:
        """
        Validasi URL via GET request dengan browser-like headers.
        Lebih reliabel dari HEAD untuk situs seperti Coursera dan docs resmi.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            async with aiohttp.ClientSession(timeout=_VALIDATE_TIMEOUT) as session:
                async with session.get(
                    url, allow_redirects=True, headers=headers
                ) as resp:
                    ok = resp.status < 400
                    logger.debug(f"GET {url} -> {resp.status}")
                    return ok
        except Exception as e:
            logger.debug(f"GET failed {url}: {e}")
            return False

    async def _url_exists(self, url: str) -> bool:
        """HEAD-based check — dipakai untuk debugging saja."""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; SkillAlign/1.0)"}
            async with aiohttp.ClientSession(timeout=_VALIDATE_TIMEOUT) as session:
                async with session.head(
                    url, allow_redirects=True, headers=headers
                ) as resp:
                    return resp.status < 400
        except Exception:
            return False

    # ──────────────────────────────────────────────────────────────────
    # Gemini REST helper
    # ──────────────────────────────────────────────────────────────────

    async def _call_gemini(
        self,
        prompt: str,
        json_mode: bool = False,
        use_search_grounding: bool = False,
    ) -> str:
        async with _GEMINI_SEMAPHORE:
            return await self._call_gemini_inner(prompt, json_mode, use_search_grounding)

    async def _call_gemini_inner(
        self,
        prompt: str,
        json_mode: bool = False,
        use_search_grounding: bool = False,
    ) -> str:
        payload: Dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4096},
        }
        if json_mode and not use_search_grounding:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        if use_search_grounding:
            payload["tools"] = [{"google_search": {}}]

        url = f"{_GENERATE_URL}?key={self._gemini_key}"
        async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
            async with session.post(
                url, json=payload, headers={"Content-Type": "application/json"}
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Gemini API error {resp.status}: {body[:300]}")
                data = await resp.json()

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Gemini response structure: {e}") from e

    def _parse_json(self, text: str) -> Optional[Dict]:
        text = text.strip()
        if "```" in text:
            for part in text.split("```"):
                part = part.strip().lstrip("json").strip()
                if part.startswith("{"):
                    try:
                        return json.loads(part)
                    except json.JSONDecodeError:
                        pass
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        if start != -1:
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break
        logger.warning(f"JSON parse failed: {text[:200]}")
        return None

    # ──────────────────────────────────────────────────────────────────
    # YouTube via YouTube Data API v3 + Gemini re-ranking
    # ──────────────────────────────────────────────────────────────────

    async def _find_youtube_courses(self, skill: str, job_title: str) -> List[Dict]:
        if not self._yt_key:
            logger.warning("YOUTUBE_API_KEY tidak di-set.")
            return []
        videos = await self._youtube_search(f"{skill} tutorial course {job_title}")
        if not videos:
            return []
        return await self._rank_youtube(videos, skill, job_title)

    async def _youtube_search(self, query: str) -> List[Dict]:
        params = {
            "part": "snippet", "q": query, "type": "video",
            "maxResults": _YT_FETCH, "relevanceLanguage": "en",
            "videoDuration": "medium", "key": self._yt_key,
        }
        try:
            async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
                async with session.get(_YOUTUBE_SEARCH_URL, params=params) as resp:
                    if resp.status != 200:
                        logger.warning(f"YouTube API {resp.status} for '{query}'")
                        return []
                    data = await resp.json()
            return [
                {
                    "title":     item["snippet"].get("title", ""),
                    "channel":   item["snippet"].get("channelTitle", ""),
                    "desc":      item["snippet"].get("description", "")[:120],
                    "url":       f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                    "thumbnail": (
                        item["snippet"].get("thumbnails", {}).get("medium", {}).get("url", "")
                    ),
                }
                for item in data.get("items", [])
                if item.get("id", {}).get("videoId")
            ]
        except Exception as e:
            logger.warning(f"YouTube search error: {e}")
            return []

    async def _rank_youtube(self, videos: List[Dict], skill: str, job_title: str) -> List[Dict]:
        payload = json.dumps(
            [{"i": i, "title": v["title"], "ch": v["channel"], "desc": v["desc"]}
             for i, v in enumerate(videos)],
            ensure_ascii=False,
        )
        prompt = (
            f'From these YouTube videos about "{skill}" for someone targeting {job_title}, '
            f"pick the {_YT_TOP} most educational and relevant ones.\n\n"
            f"Videos:\n{payload}\n\n"
            f"Return ONLY a JSON array of 0-based indices, best first. Example: [2, 0, 4, 1, 3]\n"
            f"Return ONLY the JSON array, nothing else."
        )
        try:
            raw = await self._call_gemini(prompt, json_mode=False)
            raw = raw.strip().replace("```json", "").replace("```", "").strip()
            indices: List[int] = json.loads(raw)
            return [
                videos[i] for i in indices[:_YT_TOP]
                if isinstance(i, int) and 0 <= i < len(videos)
            ]
        except Exception as e:
            logger.warning(f"Gemini YouTube ranking error for '{skill}': {e}")
            return videos[:_YT_TOP]


# ──────────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────────

def _extract_resource_items(raw: str) -> List[Dict]:
    """
    Ekstrak pasangan (judul, URL) dari plain text Gemini.
    Coba format "Judul" URL terlebih dahulu, fallback ke URL polos.
    """
    items: List[Dict] = []
    seen: set = set()

    for m in _RE_RESOURCE.finditer(raw):
        url = m.group(2).rstrip(".,)>]")
        if url not in seen:
            seen.add(url)
            items.append({"title": m.group(1).strip(), "url": url})

    # Fallback: URL polos, judul dari domain+path
    if not items:
        for url in _RE_URL_BROAD.findall(raw):
            url = url.rstrip(".,)>]")
            if url not in seen:
                seen.add(url)
                items.append({"title": _title_from_url_path(url), "url": url})

    return items


def _classify_source(url: str) -> str:
    """Klasifikasikan tipe resource dari URL-nya."""
    if "roadmap.sh" in url:
        return "roadmap"
    if "github.com" in url:
        return "github"
    if any(d in url for d in (
        "freecodecamp.org", "theodinproject.com",
        "training.linuxfoundation.org", "edx.org",
        "khanacademy.org", "coursera.org",
    )):
        return "free_course"
    return "docs"


def _title_from_url_path(url: str) -> str:
    """Buat judul yang terbaca dari path URL sebagai fallback."""
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        domain = p.netloc.replace("www.", "")
        path   = p.path.strip("/").replace("-", " ").replace("/", " › ").title()
        return f"{path} — {domain}" if path else domain
    except Exception:
        return url
