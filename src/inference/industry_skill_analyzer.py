"""
Industry Skill Analyzer untuk SkillAlign AI.

Menganalisis kesiapan skill kandidat berdasarkan REAL job postings
(bukan data statis), dengan pendekatan frequency-threshold + tiering:

  Core     (≥ 60% postings) → skill wajib, paling penting
  Common   (30–59%)         → skill penting, bernilai plus
  Optional (< 30%)          → nice-to-have, tidak dihitung readiness

Persentase readiness dihitung dengan bobot:
  readiness = (core_readiness × 0.7) + (common_readiness × 0.3)

Hasilnya adalah `IndustrySkillReport` yang bisa langsung
dikembalikan di response endpoint /recommend.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# Threshold configuration
# =============================================================================

CORE_THRESHOLD   = 0.60   # skill muncul di ≥ 60% posting → Core   (untuk n ≥ 10)
COMMON_THRESHOLD = 0.30   # skill muncul di ≥ 30% posting → Common (untuk n ≥ 10)
                           # < threshold → Optional (tidak dihitung readiness)

# Untuk n < 10: threshold diturunkan adaptif agar tetap bermakna
# Core   = skill muncul di ≥ ceil(n × 0.4) posting, minimum 2
# Common = skill muncul di ≥ 1 posting
# (lihat _compute_adaptive_thresholds)

CORE_WEIGHT   = 0.70      # bobot core dalam readiness keseluruhan
COMMON_WEIGHT = 0.30      # bobot common dalam readiness keseluruhan

# Readiness label mapping (diurutkan dari tertinggi)
_READINESS_LABELS: List[Tuple[float, str]] = [
    (80.0, "Expert Ready"),
    (60.0, "Almost Ready"),
    (40.0, "Developing"),
    (20.0, "Beginner"),
    (0.0,  "Novice"),
]


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class SkillTier:
    """Satu tier skill (core / common / optional)."""
    required:     List[str] = field(default_factory=list)   # semua skill di tier ini
    matched:      List[str] = field(default_factory=list)   # CV punya ✅
    missing:      List[str] = field(default_factory=list)   # CV tidak punya ❌
    readiness_pct: float    = 0.0                           # matched/required × 100


@dataclass
class PriorityGap:
    """Satu skill yang kurang, dengan urgensi berdasarkan frekuensi posting."""
    skill:     str
    frequency: float   # proporsi posting yang menyebutkan skill ini (0.0–1.0)
    tier:      str     # "core" | "common"


@dataclass
class IndustrySkillReport:
    """
    Hasil analisis kesiapan skill kandidat vs industri nyata.

    Dibangun dari ekstraksi skill pada kumpulan job postings,
    bukan dari data statis ROLE_SKILL_MAP.
    """
    # ── Tier breakdown
    core:     SkillTier = field(default_factory=SkillTier)
    common:   SkillTier = field(default_factory=SkillTier)
    optional: List[str] = field(default_factory=list)   # hanya daftar, tidak dihitung

    # ── Ringkasan
    readiness_percentage: float = 0.0    # 0–100, weighted dari core + common
    readiness_label:      str   = "Novice"

    # ── Missing skills terurut by urgency (frekuensi tertinggi dulu)
    priority_gaps: List[PriorityGap] = field(default_factory=list)

    # ── Bonus: skill CV di luar semua requirement (menunjukkan keunikan)
    bonus_skills: List[str] = field(default_factory=list)

    # ── Metadata
    postings_analyzed:    int             = 0
    skill_frequency:      Dict[str, float] = field(default_factory=dict)
    high_diversity:       bool             = False   # True jika threshold diturunkan adaptif
    effective_thresholds: Dict[str, float] = field(default_factory=dict)  # threshold yg benar-benar dipakai


# =============================================================================
# Analyzer
# =============================================================================

class IndustrySkillAnalyzer:
    """
    Menganalisis kesiapan skill kandidat terhadap kebutuhan industri nyata.

    Menggunakan HybridScorer.extract_skill_categories() (yang sudah ada)
    untuk mengekstrak skill dari setiap job description, lalu mengagregasi
    frekuensinya untuk menentukan skill apa yang benar-benar dicari industri.

    Tidak memerlukan model baru — memanfaatkan ekstraksi yang sudah
    berjalan di pipeline batch predict.

    Example:
        >>> analyzer = IndustrySkillAnalyzer()
        >>> report = analyzer.analyze(
        ...     cv_text="Backend dev with Python, Docker...",
        ...     job_descriptions=["We need Python, Docker, K8s...", ...]
        ... )
        >>> print(report.readiness_percentage)   # 72.5
        >>> print(report.readiness_label)         # "Almost Ready"
    """

    @staticmethod
    def _compute_adaptive_thresholds(n: int) -> Tuple[float, float, bool]:
        """
        Hitung threshold adaptif berdasarkan jumlah job posting.

        Untuk dataset besar (n ≥ 10): pakai threshold standar (60%/30%).
        Untuk dataset kecil (n < 10): turunkan threshold agar analisis
        tetap bermakna meski postings sedikit dan beragam.

        Logika:
          Core   = muncul di ≥ max(2, floor(n × 0.4)) posting  → min 2 posting
          Common = muncul di ≥ 1 posting

        Contoh n=5:
          core_count  = max(2, floor(5×0.4)) = max(2,2) = 2
          core_thresh = 2/5 = 0.40  → perlu ≥2 dari 5 posting
          common_thresh = 1/5 = 0.20 → perlu ≥1 dari 5 posting

        Returns:
            (core_thresh, common_thresh, high_diversity)
            high_diversity=True  jika threshold diadaptasi (n < 10)
        """
        if n >= 10:
            return CORE_THRESHOLD, COMMON_THRESHOLD, False

        # Threshold adaptif untuk n kecil
        core_count   = max(2, int(n * 0.4))   # minimum 2 posting
        common_count = 1                        # minimum 1 posting

        core_thresh   = round(core_count   / n, 4)
        common_thresh = round(common_count / n, 4)

        return core_thresh, common_thresh, True

    def analyze(
        self,
        cv_text:          str,
        job_descriptions: List[str],
    ) -> IndustrySkillReport:
        """
        Analisis kesiapan skill kandidat vs job postings nyata.

        Args:
            cv_text:          Teks CV kandidat.
            job_descriptions: List teks job description dari postings yang relevan.

        Returns:
            IndustrySkillReport dengan tier breakdown, readiness %, dan priority gaps.
        """
        from src.inference.hybrid_scorer import HybridScorer

        n = len(job_descriptions)
        if n == 0:
            logger.warning("IndustrySkillAnalyzer: tidak ada job descriptions.")
            return IndustrySkillReport(postings_analyzed=0)

        # ── 0. Hitung adaptive threshold berdasarkan jumlah posting
        core_thresh, common_thresh, high_diversity = self._compute_adaptive_thresholds(n)

        # ── 1. Ekstrak skill dari setiap job description
        skill_counts: Dict[str, int] = {}
        for jd in job_descriptions:
            jd_skills = HybridScorer.extract_skill_categories(jd)
            for skill in jd_skills:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1

        # ── 2. Hitung frekuensi (0.0–1.0)
        skill_freq: Dict[str, float] = {
            skill: count / n
            for skill, count in skill_counts.items()
        }

        # ── 3. Ekstrak skill CV kandidat
        cv_skills: Set[str] = HybridScorer.extract_skill_categories(cv_text)

        # ── 4. Klasifikasikan skill berdasarkan adaptive threshold
        core_required:   List[str] = []
        common_required: List[str] = []
        optional:        List[str] = []

        for skill, freq in sorted(skill_freq.items(), key=lambda x: -x[1]):
            if freq >= core_thresh:
                core_required.append(skill)
            elif freq >= common_thresh:
                common_required.append(skill)
            else:
                optional.append(skill)

        # ── 5. Bandingkan vs CV skills
        all_required: Set[str] = set(core_required) | set(common_required)
        bonus_skills = sorted(cv_skills - all_required - set(optional))

        def _split(required: List[str]) -> Tuple[List[str], List[str]]:
            matched = sorted([s for s in required if s in cv_skills])
            missing = sorted([s for s in required if s not in cv_skills])
            return matched, missing

        core_matched,   core_missing   = _split(core_required)
        common_matched, common_missing = _split(common_required)

        # ── 6. Hitung readiness per tier
        def _pct(matched: List[str], required: List[str]) -> float:
            if not required:
                return 100.0   # tidak ada requirement → perfect
            return round(len(matched) / len(required) * 100, 1)

        core_pct   = _pct(core_matched,   core_required)
        common_pct = _pct(common_matched, common_required)

        # Weighted overall readiness
        if core_required and common_required:
            overall_pct = round(
                core_pct * CORE_WEIGHT + common_pct * COMMON_WEIGHT, 1
            )
        elif core_required:
            overall_pct = core_pct     # hanya core yang ada
        elif common_required:
            overall_pct = common_pct   # hanya common
        else:
            overall_pct = 100.0        # tidak ada requirement yang terdeteksi

        # ── 7. Tentukan readiness label
        label = _READINESS_LABELS[-1][1]
        for threshold, lbl in _READINESS_LABELS:
            if overall_pct >= threshold:
                label = lbl
                break

        # ── 8. Priority gaps (missing, diurutkan by frequency)
        priority_gaps: List[PriorityGap] = []
        for skill in core_missing:
            priority_gaps.append(PriorityGap(
                skill=skill,
                frequency=round(skill_freq[skill], 3),
                tier="core",
            ))
        for skill in common_missing:
            priority_gaps.append(PriorityGap(
                skill=skill,
                frequency=round(skill_freq[skill], 3),
                tier="common",
            ))
        priority_gaps.sort(key=lambda g: -g.frequency)

        logger.info(
            f"Industry analysis: {n} postings (diversity={'high' if high_diversity else 'normal'}), "
            f"thresholds=core≥{core_thresh:.0%}/common≥{common_thresh:.0%}, "
            f"core={len(core_required)} skills, "
            f"common={len(common_required)} skills, "
            f"readiness={overall_pct}% ({label})"
        )

        return IndustrySkillReport(
            core=SkillTier(
                required=core_required,
                matched=core_matched,
                missing=core_missing,
                readiness_pct=core_pct,
            ),
            common=SkillTier(
                required=common_required,
                matched=common_matched,
                missing=common_missing,
                readiness_pct=common_pct,
            ),
            optional=optional,
            readiness_percentage=overall_pct,
            readiness_label=label,
            priority_gaps=priority_gaps,
            bonus_skills=bonus_skills,
            postings_analyzed=n,
            skill_frequency={k: round(v, 3) for k, v in skill_freq.items()},
            high_diversity=high_diversity,
            effective_thresholds={"core": core_thresh, "common": common_thresh},
        )
