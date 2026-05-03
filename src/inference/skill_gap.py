"""
Skill Gap Analyzer untuk SkillAlign AI Service.

Menggunakan TF-IDF untuk mengekstrak keyword dari Job Description,
lalu mengecek keberadaan skill tersebut langsung di teks CV (bukan
di top-N keywords CV). Ini menghindari masalah cutoff top-N dan
normalisasi tech terms seperti React.js, Node.js, CI/CD.
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# ============================================================
# Kata-kata yang bukan nama skill (noise)
# ============================================================
_NOISE_WORDS = {
    # Kata umum bahasa Inggris
    "experience", "year", "years", "strong", "ability", "knowledge",
    "work", "working", "team", "role", "position", "candidate",
    "required", "requirement", "preferred", "including",
    "responsibilities", "skills", "skill", "job", "degree",
    "bachelor", "master", "phd", "university", "education",
    "looking", "seeking", "proficient", "familiar", "understanding",
    "good", "excellent", "great", "minimum", "plus", "bonus",
    "equivalent", "related", "commercial", "preferred",
    # Kata deskriptif
    "versatile", "comfortable", "end", "feature", "features",
    "technologies", "technology", "stack", "development",
    "implementation", "design", "developer", "engineer", "manager",
    "lead", "senior", "junior", "mid", "level", "full",
    # Frasa noise yang sering muncul
    "end to", "to end", "end end", "full stack", "stack engineer",
    "ui implementation", "schema design", "end to end",
}

# ============================================================
# Normalisasi tech terms
# ============================================================
_TECH_REPLACEMENTS = [
    # Framework / library dengan .js
    (r'\breact\.js\b',    'reactjs'),
    (r'\bvue\.js\b',      'vuejs'),
    (r'\bnode\.js\b',     'nodejs'),
    (r'\bnext\.js\b',     'nextjs'),
    (r'\bnuxt\.js\b',     'nuxtjs'),
    (r'\bangular\.js\b',  'angularjs'),
    (r'\bexpress\.js\b',  'expressjs'),
    # Bahasa pemrograman khusus
    (r'\bc\+\+\b',        'cplusplus'),
    (r'\bc#\b',           'csharp'),
    (r'\b\.net\b',        'dotnet'),
    (r'\basp\.net\b',     'aspnet'),
    # DevOps / infra
    (r'\bci/cd\b',        'cicd'),
    (r'\bci-cd\b',        'cicd'),
    # Slash-separated options → pisah jadi dua kata
    (r'python/node\.js',  'python nodejs'),
    (r'python/node\b',    'python node'),
    (r'react/vue\b',      'react vue'),
    (r'aws/gcp\b',        'aws gcp'),
    (r'aws/azure\b',      'aws azure'),
    # Tanda hubung khusus
    (r'\bend-to-end\b',   'end to end'),
]


@dataclass
class SkillItem:
    """Representasi satu skill dengan bobot kepentingannya."""
    skill: str
    weight: float
    priority: int = 0


@dataclass
class SkillGapResult:
    """Hasil analisis skill gap antara CV dan Job Description."""
    present_skills: List[SkillItem]
    missing_skills: List[SkillItem]
    skill_gap_score: float
    skill_coverage_percent: str
    top_priority_skill: str
    recommendation_summary: str


class SkillGapAnalyzer:
    """
    Menganalisis gap antara skill di CV dan requirement di Job Description.

    Algoritma:
    1. Normalisasi teks (handle React.js, CI/CD, dll.)
    2. Ekstrak keyword dari Job Description via TF-IDF (apa yang dibutuhkan job)
    3. Untuk setiap keyword job, cek keberadaannya langsung di teks CV
       (bukan di top-N keyword CV — menghindari masalah cutoff)
    4. Hitung present/missing dan ranking
    """

    def __init__(self, top_n_job: int = 15):
        """
        Args:
            top_n_job: Jumlah keyword teratas yang diekstrak dari Job Description.
        """
        self.top_n_job = top_n_job

    # ------------------------------------------------------------------
    # Normalisasi
    # ------------------------------------------------------------------

    def _normalize(self, text: str) -> str:
        """
        Normalisasi teks: lowercase, handle tech terms, hapus punctuation.

        Urutan penting: regex khusus dulu sebelum hapus punctuation.
        """
        text = text.lower()
        for pattern, replacement in _TECH_REPLACEMENTS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        # Hapus karakter non-alfanumerik kecuali spasi
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # ------------------------------------------------------------------
    # Ekstraksi keyword dari job description
    # ------------------------------------------------------------------

    def _extract_job_keywords(self, job_text_normalized: str) -> List[Tuple[str, float]]:
        """
        Ekstrak top-N keyword dari job description menggunakan TF-IDF.

        Keyword diambil dari teks job saja (bukan CV), tapi karena single-doc
        TF-IDF menghasilkan bobot sama, kita gunakan term frequency (TF) saja
        sebagai pengganti ranking — kata yang lebih sering disebut = lebih penting.

        Returns:
            List of (keyword, score) sorted descending.
        """
        try:
            tfidf = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words='english',
                max_features=200,
                min_df=1,
                sublinear_tf=True,      # Gunakan log(1+tf) untuk mengecilkan dominasi term umum
            )
            matrix = tfidf.fit_transform([job_text_normalized])
            feature_names = np.array(tfidf.get_feature_names_out())
            scores = matrix.toarray()[0]

            # Filter noise words
            results = []
            for i, (kw, sc) in enumerate(zip(feature_names, scores)):
                if sc <= 0:
                    continue
                # Cek apakah keyword mengandung noise word
                kw_words = set(kw.split())
                if kw_words.issubset(_NOISE_WORDS):
                    continue
                if kw in _NOISE_WORDS:
                    continue
                # Minimal satu kata bermakna (bukan noise)
                meaningful = [w for w in kw.split() if w not in _NOISE_WORDS and len(w) > 1]
                if not meaningful:
                    continue
                results.append((kw, round(float(sc), 4)))

            # Urutkan: bigram relevan dulu, lalu unigram; dalam kelompok sama urutkan by score
            results.sort(key=lambda x: (-len(x[0].split()), -x[1]))
            return results[:self.top_n_job]

        except Exception as e:
            logger.warning(f"TF-IDF extraction error: {e}")
            return []

    # ------------------------------------------------------------------
    # Cek keberadaan skill di CV
    # ------------------------------------------------------------------

    def _skill_in_cv(self, skill: str, cv_normalized: str) -> bool:
        """
        Cek apakah skill dari job description ada di teks CV.

        Menggunakan whole-word matching untuk menghindari false positive
        (misal: "react" tidak match dengan "reaction").

        Args:
            skill: keyword dari job description (sudah dinormalisasi)
            cv_normalized: teks CV yang sudah dinormalisasi
        """
        skill_words = skill.split()

        if len(skill_words) == 1:
            # Unigram: whole-word boundary match
            pattern = r'\b' + re.escape(skill) + r'\b'
            return bool(re.search(pattern, cv_normalized))
        else:
            # Bigram/phrase: cek exact phrase ATAU semua kata individual ada di CV
            if skill in cv_normalized:
                return True
            # Fallback: semua kata dalam frasa ada di CV (tidak harus berdampingan)
            return all(
                bool(re.search(r'\b' + re.escape(w) + r'\b', cv_normalized))
                for w in skill_words
                if w not in _NOISE_WORDS
            )

    # ------------------------------------------------------------------
    # Rekomendasi
    # ------------------------------------------------------------------

    def _build_recommendation(
        self,
        missing: List[SkillItem],
        present: List[SkillItem],
        coverage: float
    ) -> str:
        """Bangun teks ringkasan rekomendasi."""
        pct = int(coverage * 100)

        if coverage >= 0.8:
            level = "sangat baik"
            action = "Tingkatkan level keahlian di skill yang sudah dimiliki."
        elif coverage >= 0.5:
            level = "cukup baik"
            top3 = ", ".join(s.skill for s in missing[:3])
            action = f"Fokuskan belajar: {top3}."
        elif coverage >= 0.2:
            level = "perlu peningkatan"
            top3 = ", ".join(s.skill for s in missing[:3])
            action = f"Prioritaskan mempelajari: {top3}."
        else:
            level = "sangat kurang sesuai"
            top3 = ", ".join(s.skill for s in missing[:3])
            action = (
                f"CV belum memenuhi requirement posisi ini. "
                f"Mulai dari: {top3}."
            )

        present_str = (
            ", ".join(s.skill for s in present[:3]) if present else "belum ada yang terdeteksi"
        )

        return (
            f"Kesesuaian skill: {pct}% ({level}). "
            f"Skill yang sudah dimiliki: {present_str}. "
            f"{action}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, cv_text: str, job_description: str) -> SkillGapResult:
        """
        Analisis skill gap antara CV dan Job Description.

        Args:
            cv_text:         Teks CV pengguna (raw, belum dinormalisasi).
            job_description: Teks deskripsi lowongan kerja target.

        Returns:
            SkillGapResult dengan detail missing/present skills.
        """
        # Normalisasi kedua teks dengan cara yang sama
        cv_normalized  = self._normalize(cv_text)
        job_normalized = self._normalize(job_description)

        # Ekstrak keyword dari job description (ini adalah requirement)
        job_keywords = self._extract_job_keywords(job_normalized)

        if not job_keywords:
            logger.warning("No job keywords extracted. Job description mungkin terlalu pendek.")

        # Cek setiap keyword job apakah ada di teks CV (bukan di top-N CV)
        present_raw: Dict[str, float] = {}
        missing_raw: Dict[str, float] = {}

        for kw, weight in job_keywords:
            if self._skill_in_cv(kw, cv_normalized):
                present_raw[kw] = weight
            else:
                missing_raw[kw] = weight

        # Buat SkillItem list (sudah terurut dari ekstraksi)
        present_skills = [
            SkillItem(skill=kw, weight=w, priority=0)
            for kw, w in sorted(present_raw.items(), key=lambda x: -x[1])
        ]
        missing_skills = [
            SkillItem(skill=kw, weight=w, priority=rank + 1)
            for rank, (kw, w) in enumerate(
                sorted(missing_raw.items(), key=lambda x: -x[1])
            )
        ]

        # Hitung coverage score
        total = len(job_keywords)
        n_present = len(present_skills)
        skill_gap_score = round(n_present / total, 4) if total > 0 else 0.0
        skill_coverage_percent = f"{int(skill_gap_score * 100)}%"
        top_priority = missing_skills[0].skill if missing_skills else "-"
        summary = self._build_recommendation(missing_skills, present_skills, skill_gap_score)

        logger.info(
            f"Skill gap: {n_present}/{total} present ({skill_coverage_percent}), "
            f"{len(missing_skills)} missing. Top missing: {top_priority}"
        )

        return SkillGapResult(
            present_skills=present_skills,
            missing_skills=missing_skills,
            skill_gap_score=skill_gap_score,
            skill_coverage_percent=skill_coverage_percent,
            top_priority_skill=top_priority,
            recommendation_summary=summary,
        )
