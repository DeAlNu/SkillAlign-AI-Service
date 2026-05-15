"""
Skill Gap Analyzer v6 — SkillNer (EMSI Skill Database).

SkillNer menggunakan EMSI/Lightcast skill database (~6000 skills)
yang mencakup berbagai industri: IT, healthcare, finance, logistics, dll.

Keunggulan vs pendekatan sebelumnya:
- Tidak perlu vocabulary manual atau CSV buatan sendiri
- Tidak ada noise words ("development", "design", "engineer")
- Mengenali skill multi-kata: "machine learning", "project management"
- Matching berdasarkan skill_id EMSI → canonical, lintas variasi penulisan
- Mencakup semua industri dalam database EMSI

Dependency:
    pip install skillNer spacy
    python -m spacy download en_core_web_lg
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SkillItem:
    skill: str
    skill_id: str       # EMSI canonical skill ID untuk de-duplikasi
    match_score: float  # Confidence dari SkillNer (1.0 = full match)
    priority: int = 0


@dataclass
class SkillGapResult:
    present_skills: List[SkillItem]
    missing_skills: List[SkillItem]
    skill_gap_score: float
    skill_coverage_percent: str
    top_priority_skill: str
    recommendation_summary: str


class SkillGapAnalyzer:
    """
    Skill Gap Analyzer menggunakan SkillNer (EMSI database).

    SkillNer mendeteksi skill dari teks menggunakan dua metode:
    1. Full match: exact phrase match terhadap EMSI skill names
       → confidence = 1.0
    2. N-gram scored: partial match dengan Jaro-Winkler similarity
       → confidence = 0.7 - 1.0

    Matching antara CV dan Job menggunakan skill_id (canonical),
    sehingga "Python" dan "Python Programming" → skill ID yang sama.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_lg",
        ngram_threshold: float = 0.85,
        semantic_threshold: float = 0.72,
    ):
        """
        Args:
            spacy_model:        spaCy model ('en_core_web_lg' akurat, 'en_core_web_sm' ringan).
            ngram_threshold:    Confidence minimum SkillNer ngram match (0.85).
            semantic_threshold: Cosine similarity minimum untuk SBERT fallback matching.
                                0.78 = tangkap sinonim dekat (cloud service ≈ cloud infra),
                                tapi tolak yang jauh (microservices ≠ docker).
        """
        self.spacy_model = spacy_model
        self.ngram_threshold = ngram_threshold
        self.semantic_threshold = semantic_threshold
        self._extractor = None
        self._encoder = None

    # ------------------------------------------------------------------
    # Lazy load SkillNer (berat, ~400MB model)
    # ------------------------------------------------------------------

    @property
    def extractor(self):
        """SkillNer extractor (lazy-loaded saat request pertama)."""
        if self._extractor is None:
            logger.info(f"Loading SkillNer with spaCy model: {self.spacy_model} ...")
            try:
                import spacy
                from spacy.matcher import PhraseMatcher
                from skillNer.general_params import SKILL_DB
                from skillNer.skill_extractor_class import SkillExtractor

                nlp = spacy.load(self.spacy_model)
                self._extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
                logger.info(
                    f"SkillNer ready. EMSI skill database loaded "
                    f"({len(SKILL_DB)} skills)."
                )
            except OSError as e:
                raise RuntimeError(
                    f"spaCy model '{self.spacy_model}' tidak ditemukan. "
                    f"Jalankan: python -m spacy download {self.spacy_model}"
                ) from e
            except ImportError as e:
                raise RuntimeError(
                    "SkillNer atau spaCy tidak terinstall. "
                    "Jalankan: pip install skillNer spacy"
                ) from e
        return self._extractor

    @property
    def encoder(self):
        """SentenceTransformer untuk semantic fallback (lazy-loaded)."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer for semantic fallback...")
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("SentenceTransformer ready.")
        return self._encoder

    def warmup(self):
        """Pre-load semua model saat startup (hindari cold start lag)."""
        _ = self.extractor
        _ = self.encoder
        logger.info("SkillGapAnalyzer warm-up complete.")

    def _semantic_fallback_batch(
        self,
        missing_job_skills: List[Tuple[str, str, float]],
        cv_skill_names: List[str],
    ) -> List[Tuple[bool, float]]:
        """
        Batch semantic fallback — encode SEMUA missing job skills dan SEMUA
        CV skills dalam 2 encode call, lalu hitung similarity matrix sekali.

        SEBELUM: 5 job skills × 2 encode calls = 10 SBERT calls
        SESUDAH: 2 encode calls total (batch) → jauh lebih cepat

        Returns:
            List[(is_match, sim_score)] dengan urutan sama seperti missing_job_skills.
        """
        n = len(missing_job_skills)
        if not cv_skill_names or n == 0:
            return [(False, 0.0)] * n
        try:
            from sentence_transformers import util as st_util

            job_names = [name for _, name, _ in missing_job_skills]

            # 2 encode calls saja, bukan n calls
            job_embs = self.encoder.encode(job_names,      convert_to_tensor=True, show_progress_bar=False)
            cv_embs  = self.encoder.encode(cv_skill_names, convert_to_tensor=True, show_progress_bar=False)

            # Matrix (n_missing_job × n_cv) — satu operasi
            sim_matrix = st_util.cos_sim(job_embs, cv_embs)

            results: List[Tuple[bool, float]] = []
            for i in range(n):
                max_sim = float(sim_matrix[i].max())
                results.append((max_sim >= self.semantic_threshold, round(max_sim, 4)))
            return results

        except Exception as e:
            logger.error(f"Semantic fallback batch error: {e}")
            return [(False, 0.0)] * n

    # ------------------------------------------------------------------
    # Ekstraksi skill dari teks
    # ------------------------------------------------------------------

    def _extract_skills(self, text: str) -> Dict[str, Tuple[str, float]]:
        """
        Ekstrak skills dari teks menggunakan SkillNer.

        Returns:
            Dict[skill_id, (skill_name, confidence)]
            Menggunakan skill_id sebagai key untuk de-duplikasi canonical.
        """
        if not text or len(text.strip()) < 10:
            return {}

        try:
            annotations = self.extractor.annotate(text)
            skills: Dict[str, Tuple[str, float]] = {}

            # Full matches — exact match terhadap EMSI skill names
            for match in annotations["results"]["full_matches"]:
                sid   = match["skill_id"]
                name  = match["doc_node_value"].lower().strip()
                # Full match = confidence 1.0
                if sid not in skills or skills[sid][1] < 1.0:
                    skills[sid] = (name, 1.0)

            # Ngram scored — partial/fuzzy matches
            for match in annotations["results"]["ngram_scored"]:
                confidence = match.get("score", 0.0)
                if confidence < self.ngram_threshold:
                    continue
                sid  = match["skill_id"]
                name = match["doc_node_value"].lower().strip()
                if sid not in skills or skills[sid][1] < confidence:
                    skills[sid] = (name, round(float(confidence), 4))

            return skills

        except Exception as e:
            logger.error(f"SkillNer extraction error: {e}")
            return {}

    # ------------------------------------------------------------------
    # Rekomendasi
    # ------------------------------------------------------------------

    def _build_recommendation(
        self,
        missing: List[SkillItem],
        present: List[SkillItem],
        coverage: float
    ) -> str:
        pct = int(coverage * 100)
        if coverage >= 0.8:
            level, action = "sangat baik", "Tingkatkan level keahlian di skill yang sudah dimiliki."
        elif coverage >= 0.6:
            top3 = ", ".join(s.skill for s in missing[:3])
            level, action = "cukup baik", f"Pertimbangkan untuk mempelajari: {top3}."
        elif coverage >= 0.3:
            top3 = ", ".join(s.skill for s in missing[:3])
            level, action = "perlu peningkatan", f"Prioritaskan mempelajari: {top3}."
        else:
            top3 = ", ".join(s.skill for s in missing[:3])
            level, action = "kurang sesuai", f"Perbedaan skill cukup besar. Fokus pada: {top3}."

        present_str = ", ".join(s.skill for s in present[:4]) if present else "belum terdeteksi"
        return (
            f"Kesesuaian skill: {pct}% ({level}). "
            f"Skill yang sudah dimiliki: {present_str}. {action}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, cv_text: str, job_description: str) -> SkillGapResult:
        """
        Analisis skill gap antara CV dan Job Description.

        Menggunakan SkillNer untuk mengekstrak named skill entities dari
        kedua teks, lalu membandingkan berdasarkan EMSI skill_id.

        Args:
            cv_text:         Teks CV pengguna (raw).
            job_description: Teks deskripsi lowongan kerja target (raw).
        """
        t0 = time.time()

        # Step 1: Ekstrak skills dari job description dan CV
        job_skills = self._extract_skills(job_description)
        cv_skills  = self._extract_skills(cv_text)

        logger.info(f"Job skills ({len(job_skills)}): {[n for n,_ in job_skills.values()]}")
        logger.info(f"CV  skills ({len(cv_skills)}):  {[n for n,_ in cv_skills.values()]}")

        if not job_skills:
            logger.warning("Tidak ada skill yang terdeteksi dari job description.")
            return SkillGapResult(
                present_skills=[], missing_skills=[],
                skill_gap_score=0.0, skill_coverage_percent="0%",
                top_priority_skill="-",
                recommendation_summary="Tidak ada skill yang terdeteksi dari job description."
            )

        # Step 2a: EMSI exact ID matching (primary)
        cv_skill_ids:   Set[str]  = set(cv_skills.keys())
        cv_skill_names: List[str] = [name for name, _ in cv_skills.values()]

        present_skills: List[SkillItem] = []
        still_missing:  List[Tuple[str, str, float]] = []   # (skill_id, name, confidence)

        for skill_id, (skill_name, confidence) in job_skills.items():
            if skill_id in cv_skill_ids:
                present_skills.append(SkillItem(
                    skill=skill_name, skill_id=skill_id,
                    match_score=confidence, priority=0
                ))
            else:
                still_missing.append((skill_id, skill_name, confidence))

        # Step 2b: SBERT semantic fallback — BATCH (1 matrix op, bukan per-skill)
        # Semua missing job skills di-encode sekaligus vs semua CV skills
        missing_skills: List[SkillItem] = []
        missing_rank = 1

        if still_missing:
            batch_results = self._semantic_fallback_batch(still_missing, cv_skill_names)
            for (skill_id, skill_name, confidence), (is_match, sim_score) in zip(
                still_missing, batch_results
            ):
                if is_match:
                    logger.info(
                        f"Semantic fallback match: '{skill_name}' ≈ CV skill (sim={sim_score})"
                    )
                    present_skills.append(SkillItem(
                        skill=skill_name, skill_id=skill_id,
                        match_score=sim_score, priority=0
                    ))
                else:
                    missing_skills.append(SkillItem(
                        skill=skill_name, skill_id=skill_id,
                        match_score=0.0, priority=missing_rank
                    ))
                    missing_rank += 1

        # Urutkan
        present_skills.sort(key=lambda x: -x.match_score)
        missing_skills.sort(key=lambda x: x.priority)

        # Step 3: Hitung coverage
        total     = len(job_skills)
        n_present = len(present_skills)
        skill_gap_score       = round(n_present / total, 4) if total > 0 else 0.0
        skill_coverage_percent = f"{int(skill_gap_score * 100)}%"
        top_priority = missing_skills[0].skill if missing_skills else "-"
        summary      = self._build_recommendation(missing_skills, present_skills, skill_gap_score)

        elapsed = round((time.time() - t0) * 1000, 1)
        logger.info(
            f"Skill gap done in {elapsed}ms: "
            f"{n_present}/{total} present ({skill_coverage_percent}), "
            f"top missing: '{top_priority}'"
        )

        return SkillGapResult(
            present_skills=present_skills,
            missing_skills=missing_skills,
            skill_gap_score=skill_gap_score,
            skill_coverage_percent=skill_coverage_percent,
            top_priority_skill=top_priority,
            recommendation_summary=summary,
        )
