"""
Job Title Suggester untuk SkillAlign AI.

Menyarankan daftar job title yang paling cocok berdasarkan profil CV,
menggunakan skill overlap antara skill CV dengan skill requirements
tiap role yang terdefinisi di SKILL_DICT / ROLE_PATTERNS.

Output digunakan di endpoint /api/v1/analyze-cv agar user dapat memilih
target job title sebelum masuk ke pipeline /api/v1/recommend.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.inference.role_config import ROLE_DISPLAY, ROLE_SKILL_MAP

logger = logging.getLogger(__name__)


@dataclass
class SuggestedJobTitle:
    """Satu saran job title dengan confidence dan alasan."""
    title:      str
    role_key:   str
    confidence: float
    reason:     str
    matched_skills: List[str] = field(default_factory=list)


class JobTitleSuggester:
    """
    Menyarankan job title yang cocok berdasarkan skill CV.

    Menggunakan skill overlap antara skill yang terdeteksi di CV
    dengan skill requirement tiap role di _ROLE_SKILL_MAP.

    Jika current_role terdeteksi di CV, role tersebut selalu masuk
    di posisi pertama dengan confidence 1.0.

    Example:
        >>> suggester = JobTitleSuggester()
        >>> suggestions = suggester.suggest(
        ...     cv_skills={"react", "typescript", "nodejs", "postgresql"},
        ...     current_role_key="frontend",
        ...     top_n=5
        ... )
        >>> for s in suggestions:
        ...     print(s.title, s.confidence)
    """

    def __init__(self, min_confidence: float = 0.2, min_matched: int = 1) -> None:
        """
        Args:
            min_confidence: Threshold minimum confidence (0.0–1.0).
            min_matched:    Minimal jumlah skill yang harus cocok.
        """
        self.min_confidence = min_confidence
        self.min_matched = min_matched

    def suggest(
        self,
        cv_skills: Set[str],
        current_role_key: Optional[str] = None,
        top_n: int = 5,
    ) -> List[SuggestedJobTitle]:
        """
        Buat daftar saran job title berdasarkan skill CV.

        Args:
            cv_skills:        Set skill kategori yang terdeteksi dari CV.
            current_role_key: Role key yang terdeteksi di CV (dari HybridScorer).
                              Jika ada, role ini selalu muncul di posisi 1.
            top_n:            Jumlah maksimal saran yang dikembalikan.

        Returns:
            List[SuggestedJobTitle] diurutkan by confidence DESC.
        """
        candidates: List[SuggestedJobTitle] = []
        seen_keys: Set[str] = set()

        # Prioritas pertama: current_role dari CV
        if current_role_key and current_role_key in ROLE_SKILL_MAP:
            role_skills = ROLE_SKILL_MAP[current_role_key]
            matched = sorted(cv_skills & role_skills)
            display = ROLE_DISPLAY.get(current_role_key, current_role_key.replace("_", " ").title())
            candidates.append(SuggestedJobTitle(
                title=display,
                role_key=current_role_key,
                confidence=1.0,
                reason=f"Detected as current role in CV",
                matched_skills=matched,
            ))
            seen_keys.add(current_role_key)

        # Hitung overlap untuk semua role lain
        scored: List[Tuple[float, str, List[str]]] = []
        for role_key, role_skills in ROLE_SKILL_MAP.items():
            if role_key in seen_keys:
                continue
            if not role_skills:
                continue
            matched = sorted(cv_skills & role_skills)
            n_matched = len(matched)
            if n_matched < self.min_matched:
                continue
            confidence = round(n_matched / len(role_skills), 4)
            if confidence < self.min_confidence:
                continue
            scored.append((confidence, role_key, matched))

        # Urutkan by confidence DESC
        scored.sort(key=lambda x: -x[0])

        for confidence, role_key, matched in scored:
            if len(candidates) >= top_n:
                break
            display = ROLE_DISPLAY.get(role_key, role_key.replace("_", " ").title())
            n = len(matched)
            skill_preview = ", ".join(matched[:4])
            suffix = f" and {len(matched)-4} more" if n > 4 else ""
            reason = f"{n} matching skill{'s' if n > 1 else ''}: {skill_preview}{suffix}"
            candidates.append(SuggestedJobTitle(
                title=display,
                role_key=role_key,
                confidence=confidence,
                reason=reason,
                matched_skills=matched,
            ))

        return candidates[:top_n]

    def suggest_from_cv_text(
        self,
        cv_text: str,
        top_n: int = 5,
    ) -> Tuple[List[SuggestedJobTitle], str, Set[str]]:
        """
        Shortcut: deteksi skill + role langsung dari raw CV text.

        Returns:
            (suggestions, current_role_key, cv_skill_categories)
        """
        from src.inference.hybrid_scorer import HybridScorer
        cv_skills: Set[str] = HybridScorer.extract_skill_categories(cv_text)
        current_role_key: str = HybridScorer.detect_role(cv_text)
        if current_role_key == "other":
            current_role_key = None
        suggestions = self.suggest(
            cv_skills=cv_skills,
            current_role_key=current_role_key,
            top_n=top_n,
        )
        return suggestions, current_role_key, cv_skills
