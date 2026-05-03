"""
Pair Synthesizer untuk SkillAlign AI.

Modul ini bertanggung jawab menghasilkan training pairs (CV, Job, Label)
berkualitas yang memperbaiki tiga masalah utama supervisi pada notebook
02U_model_development.ipynb sebelumnya:

1. LEXICAL LEAKAGE FIX
   - CV positif TIDAK lagi disusun langsung dari keyword job-nya sendiri
     (yang membuat model hanya belajar string-overlap detection).
   - Format CV beragam menggunakan 7+ template dengan struktur kaya
     (Summary, Experience, Skills, Education).
   - Tambahkan 1–4 "noise skills" umum supaya overlap CV-Job tidak 100%.

2. HARD NEGATIVES
   Lima mode pair generation menutupi seluruh spektrum kecocokan:
   - 'high_match'           : CV cocok 50–90% skill, industri & seniority match
   - 'partial'              : CV punya 30–50% skill job + skill ekstra
   - 'seniority_mismatch'   : skill match tapi level (junior vs senior) beda
   - 'same_domain_diff_role': industri sama, role beda (Frontend vs Backend)
   - 'cross_domain'         : CV dari industri yang sama sekali berbeda

3. SOFT LABELS (kontinyu, bukan binary {0,1})
   - Skor = 0.50*skill_overlap + 0.25*seniority + 0.15*domain + 0.10*role
   - Range [0.0, 1.0], cocok untuk regression-style training agar target
     MAE ≤ 0.02 menjadi realistis.

Usage:
    from src.preprocessing.pair_synthesizer import (
        CVJobPairSynthesizer, PairSynthConfig
    )

    synth = CVJobPairSynthesizer(
        df_sample=df_sample,
        tfidf=tfidf,
        feature_names=feature_names,
        config=PairSynthConfig(seed=42),
    )
    cv_texts, job_texts, labels = synth.synthesize()
"""

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


# =============================================================================
# CV TEMPLATES — beragam panjang dan struktur supaya tidak template-locked
# =============================================================================

CV_TEMPLATES: List[str] = [
    # 1. Detailed professional with summary section (panjang)
    "{name} - {title}. Summary: {summary_line} Experience: {company} "
    "({years_start}-{years_end}) - {role_at_company}. Key responsibilities: "
    "{responsibilities}. Education: {education}. Technical Skills: "
    "{skills_list}. {extras}",

    # 2. Skills-focused, mid-length
    "Professional {title} with {years} years of experience in "
    "{industry_short_label}. Core competencies include {skills_list}. "
    "Worked at {company} where I {project_outcome}. Hold {education}.",

    # 3. Senior/lead style
    "Senior {title} with {years}+ years of leadership experience. Led "
    "cross-functional teams at {company} on {project_outcome}. "
    "Specializations: {skills_list}. Architected {arch_focus}. Mentored "
    "{mentee_count}+ junior engineers. {education}.",

    # 4. Junior/entry-level (pendek-medium)
    "Recent {education_field} graduate from {university}. Completed "
    "{bootcamp_or_internship} program focusing on {skills_list}. Familiar "
    "with industry tools like {top_skill_1} and {top_skill_2}. Looking for "
    "an entry-level role as {target_role}.",

    # 5. Concise pipe-separated (pendek)
    "{title} | {years} years | Skills: {skills_list} | {company} "
    "({years_start}-{years_end}) | {education}",

    # 6. Narrative first-person (medium)
    "As a dedicated {title}, I have spent {years} years building expertise "
    "in {focus_area}. My background includes substantial work with "
    "{skills_list}. At {company}, I {project_outcome}. I hold {education} "
    "and continue to expand my skillset through ongoing learning.",

    # 7. Short bullet-style
    "{title}. {years} yrs experience. Tools: {skills_list}. Most recent "
    "role: {role_at_company} at {company}. {education}.",
]

# Skill umum yang ditambahkan ke CV sebagai noise (supaya overlap tidak 100%)
COMMON_NOISE_SKILLS: List[str] = [
    "communication", "teamwork", "problem solving", "agile", "scrum",
    "git", "jira", "confluence", "slack", "microsoft office",
    "presentation", "documentation", "stakeholder management",
    "code review", "unit testing", "debugging", "english",
    "time management", "leadership", "mentoring",
]

# Pool deskripsi tanggung jawab (untuk template responsibilities)
RESPONSIBILITY_POOL: List[str] = [
    "designing scalable architectures",
    "implementing automated testing pipelines",
    "leading cross-functional initiatives",
    "performing root cause analysis on production issues",
    "collaborating with product and design teams",
    "writing technical documentation",
    "conducting code reviews",
    "delivering features ahead of schedule",
    "optimizing performance bottlenecks",
    "presenting findings to stakeholders",
    "managing release cycles",
    "participating in on-call rotations",
]

# Pool nama (anonim, hanya untuk variasi)
NAMES_POOL: List[str] = [
    "Alex Chen", "Maria Garcia", "John Smith", "Priya Patel", "Liu Wei",
    "Carlos Mendoza", "Aisha Khan", "David Park", "Sarah Johnson",
    "Marcus Thorne", "Sofia Petrova", "Hiroshi Tanaka", "Emily Wong",
    "Raj Kumar", "Elena Romano",
]

EDUCATION_POOL: List[str] = [
    "Bachelor of Science in Computer Science",
    "Master of Engineering in Software Systems",
    "Bachelor of Engineering in Information Technology",
    "Bachelor's degree in Mathematics",
    "Master of Business Administration",
    "Bachelor's degree in Electrical Engineering",
    "Master of Science in Data Science",
    "Bachelor's degree in Statistics",
    "Master of Science in Computer Engineering",
]

COMPANY_POOL: List[str] = [
    "TechCorp", "InnovateLabs", "GlobalSystems Inc.", "DataDynamics",
    "CloudFirst Ltd.", "BrightFuture Solutions", "AlphaWorks",
    "Nexus Industries", "Apex Digital", "Horizon Group",
]

PROJECT_OUTCOME_POOL: List[str] = [
    "led migration to microservices",
    "reduced operational costs by 30%",
    "delivered customer-facing features used by millions",
    "improved system reliability and uptime",
    "shipped a flagship product to market",
    "scaled platform to handle 10x traffic",
]


# =============================================================================
# Konfigurasi
# =============================================================================

@dataclass
class PairSynthConfig:
    """Konfigurasi untuk pair synthesis (v3: baseline 50% pass rate)."""
    seed: int = 42

    # Skill overlap untuk positive match (CV mengambil X% skill dari job)
    pos_overlap_min: float = 0.5
    pos_overlap_max: float = 0.9

    # Distribusi mode (akan dinormalisasi)
    p_high_match: float = 0.30
    p_partial: float = 0.25
    p_seniority_mismatch: float = 0.15
    p_same_domain_diff_role: float = 0.15
    p_cross_domain: float = 0.15

    # Bobot soft label (jumlah harus = 1.0)
    w_skill_overlap: float = 0.50
    w_seniority: float = 0.25
    w_domain: float = 0.15
    w_role: float = 0.10

    # Noise skills yang ditambahkan ke CV
    n_noise_skills_min: int = 1
    n_noise_skills_max: int = 4

    # Threshold minimum job skills untuk dianggap valid
    min_job_skills: int = 3

    # Topn keyword extraction
    topn_keywords_main: int = 8
    topn_keywords_extra: int = 6

    # Skill ratio untuk partial mode
    partial_skill_ratio_min: float = 0.30
    partial_skill_ratio_max: float = 0.50

    # Multiplicative penalties — DISABLED di v3 (set ke 1.0 = no-op)
    # Bisa di-tweak nanti untuk eksperimen
    cross_domain_multiplier: float = 1.0
    severe_seniority_gap_multiplier: float = 1.0


# =============================================================================
# Synthesizer
# =============================================================================

class CVJobPairSynthesizer:
    """
    Synthesizer untuk training pairs dengan soft labels & hard negatives.

    Args:
        df_sample: DataFrame berisi minimal kolom:
            ['title', 'job_description', 'required_skills', 'industry_name'].
            'required_skills' harus berupa list of strings.
        tfidf: Pre-fitted TfidfVectorizer untuk keyword extraction.
        feature_names: np.ndarray dari tfidf.get_feature_names_out().
        config: PairSynthConfig instance (opsional).
    """

    def __init__(
        self,
        df_sample: pd.DataFrame,
        tfidf: TfidfVectorizer,
        feature_names: np.ndarray,
        config: Optional[PairSynthConfig] = None,
    ) -> None:
        self.df = df_sample.copy()
        self.tfidf = tfidf
        self.feature_names = feature_names
        self.config = config or PairSynthConfig()

        # Set seeds (dual: numpy RNG + Python random untuk template choice)
        self.rng = np.random.default_rng(self.config.seed)
        random.seed(self.config.seed)

        # Pre-compute role + seniority + industry index
        self._enrich_dataframe()
        self._build_indices()

        logger.info(
            "Synthesizer ready. df=%d rows, industries=%d, role_categories=%d",
            len(self.df),
            len(self.valid_industries),
            self.df["_role_simple"].nunique(),
        )

    # ---------- Setup helpers ----------

    def _enrich_dataframe(self) -> None:
        """Tambahkan kolom turunan: _role_simple, _seniority."""
        self.df["_role_simple"] = self.df["title"].apply(self._simplify_title)
        self.df["_seniority"] = self.df["title"].apply(self._extract_seniority)
        self.df["industry_name"] = self.df["industry_name"].fillna("Other")

    def _build_indices(self) -> None:
        """Bangun index industry → role → indices untuk sampling cepat."""
        groups = self.df.groupby("industry_name")
        self.industry_to_indices: Dict[str, List[int]] = {
            ind: grp.index.tolist()
            for ind, grp in groups
            if len(grp) >= 3
        }
        self.valid_industries: List[str] = list(self.industry_to_indices.keys())

        self.industry_role_idx: Dict[str, Dict[str, List[int]]] = {}
        for ind in self.valid_industries:
            grp = self.df[self.df["industry_name"] == ind]
            self.industry_role_idx[ind] = {
                role: g.index.tolist() for role, g in grp.groupby("_role_simple")
            }

    # ---------- Static utilities ----------

    @staticmethod
    def _simplify_title(title: str) -> str:
        """Map raw title ke kategori role tersederhana."""
        if not isinstance(title, str):
            return "other"
        t = title.lower()
        # Order matters — pattern lebih spesifik di atas
        patterns: List[Tuple[str, str]] = [
            ("data_scientist", r"data scientist"),
            ("data_engineer", r"data engineer"),
            ("data_analyst", r"data analyst|business analyst"),
            ("ml_engineer", r"machine learning|ml engineer|ai engineer"),
            ("frontend", r"front[- ]end|frontend"),
            ("backend", r"back[- ]end|backend"),
            ("full_stack", r"full[- ]stack"),
            ("mobile", r"\b(mobile|ios|android)\b"),
            ("devops", r"devops|sre|site reliability|infrastructure"),
            ("designer", r"\b(designer|ux|ui)\b"),
            ("product_manager", r"product manager|\bpm\b"),
            ("project_manager", r"project manager|program manager"),
            ("marketing", r"marketing|seo|growth"),
            ("sales", r"sales|account executive|business development"),
            ("finance", r"finance|accountant|auditor|financial"),
            ("hr", r"human resources|\bhr\b|recruit|talent"),
            ("engineer", r"engineer|developer|programmer"),
            ("manager", r"manager|director|lead"),
        ]
        for label, pat in patterns:
            if re.search(pat, t):
                return label
        return "other"

    @staticmethod
    def _extract_seniority(title: str) -> int:
        """Map title ke level seniority (1=junior, 5=executive)."""
        if not isinstance(title, str):
            return 2
        t = title.lower()
        if re.search(r"\b(intern|junior|jr\.?|entry|trainee|associate|graduate)\b", t):
            return 1
        if re.search(r"\b(director|head|vp|chief|c[a-z]o)\b", t):
            return 5
        if re.search(r"\b(senior|sr\.?|principal|staff)\b", t):
            return 4
        if re.search(r"\b(manager|architect|lead)\b", t):
            return 3
        return 2  # default mid-level

    @staticmethod
    def _adjust_title_seniority(title: str, target_seniority: int) -> str:
        """Adjust title supaya match target seniority (untuk seniority_mismatch mode)."""
        stripped = re.sub(
            r"\b(senior|sr\.?|junior|jr\.?|lead|principal|staff|chief|"
            r"head|director|associate|intern)\b",
            "",
            title,
            flags=re.IGNORECASE,
        )
        stripped = re.sub(r"\s+", " ", stripped).strip()
        prefix_map = {1: "Junior", 2: "", 3: "", 4: "Senior", 5: "Lead"}
        prefix = prefix_map.get(target_seniority, "")
        return f"{prefix} {stripped}".strip() if prefix else stripped or title

    def _extract_keywords(self, text: str, topn: int) -> List[str]:
        """Extract top-N TF-IDF keywords dari teks."""
        try:
            vec = self.tfidf.transform([str(text)[:2000]])
            scores = vec.toarray()[0]
            top_idx = scores.argsort()[::-1][:topn]
            return [self.feature_names[i] for i in top_idx if scores[i] > 0]
        except Exception as e:
            logger.debug(f"keyword extraction failed: {e}")
            return []

    # ---------- CV builder ----------

    def _build_cv(
        self,
        title: str,
        skills: List[str],
        years: int,
        seniority_level: int,
        industry: str,
    ) -> str:
        """Bangun CV string menggunakan template terpilih + filler beragam."""
        # Pilih template: junior pakai template 4, senior pakai 3 atau 6, lainnya random
        if seniority_level == 1:
            template = CV_TEMPLATES[3]
        elif seniority_level >= 4:
            template = random.choice([CV_TEMPLATES[2], CV_TEMPLATES[5]])
        else:
            # Random kecuali template junior
            candidates = [CV_TEMPLATES[i] for i in (0, 1, 4, 5, 6)]
            template = random.choice(candidates)

        # Tambah noise skills supaya overlap CV-Job tidak sempurna
        n_noise = random.randint(
            self.config.n_noise_skills_min, self.config.n_noise_skills_max
        )
        noise = random.sample(
            COMMON_NOISE_SKILLS, min(n_noise, len(COMMON_NOISE_SKILLS))
        )
        all_cv_skills = list(skills) + noise
        random.shuffle(all_cv_skills)

        # Filler values
        years_end = int(self.rng.integers(2022, 2026))
        years_start = max(2010, years_end - max(years, 1))
        skills_list = ", ".join(all_cv_skills[:8])
        top_skill_1 = all_cv_skills[0] if all_cv_skills else "general tools"
        top_skill_2 = all_cv_skills[1] if len(all_cv_skills) > 1 else "collaboration"

        try:
            cv = template.format(
                name=random.choice(NAMES_POOL),
                title=title,
                summary_line=(
                    f"Detail-oriented professional with {years} years of "
                    f"hands-on experience in {industry[:40]}."
                ),
                company=random.choice(COMPANY_POOL),
                years_start=years_start,
                years_end=years_end,
                role_at_company=title,
                responsibilities=", ".join(
                    random.sample(RESPONSIBILITY_POOL, k=random.randint(2, 4))
                ),
                education=random.choice(EDUCATION_POOL),
                skills_list=skills_list,
                extras=(
                    f"Certifications: "
                    f"{random.choice(['AWS Solutions Architect', 'GCP Professional', 'Azure Fundamentals', 'PMP', 'Scrum Master'])}."
                ),
                years=years,
                industry_short_label=industry[:30],
                project_outcome=random.choice(PROJECT_OUTCOME_POOL),
                arch_focus=random.choice(
                    ["distributed systems", "data pipelines", "cloud platforms", "ML systems"]
                ),
                mentee_count=random.randint(3, 10),
                education_field=random.choice(
                    ["Computer Science", "Information Systems", "Mathematics", "Engineering"]
                ),
                university=random.choice(
                    ["State University", "Tech Institute", "National University", "Polytechnic College"]
                ),
                bootcamp_or_internship=random.choice(
                    ["a 6-month bootcamp", "a summer internship", "a research assistantship"]
                ),
                target_role=title,
                focus_area=random.choice(
                    ["scalable systems", "user experience", "data-driven decisions", "automation"]
                ),
                top_skill_1=top_skill_1,
                top_skill_2=top_skill_2,
            )
        except KeyError as e:
            logger.warning(f"template format error ({e}), fallback to simple form")
            cv = (
                f"Experienced {title} with {years} years. "
                f"Skills: {skills_list}. Education: {random.choice(EDUCATION_POOL)}."
            )
        return cv.strip()

    # ---------- Soft label computation ----------

    def _compute_soft_label(
        self,
        cv_skills: Set[str],
        job_skills: Set[str],
        cv_seniority: int,
        job_seniority: int,
        cv_industry: str,
        job_industry: str,
        cv_role: str,
        job_role: str,
    ) -> float:
        """
        Hitung skor matching kontinyu di [0, 1].

        Komposit:
        - skill_overlap: berapa persen skill job ada di CV (coverage)
        - seniority    : seberapa dekat level CV vs requirement job
        - domain       : industri CV vs industri job
        - role         : kategori role CV vs role job
        """
        # 1. Skill coverage
        skill_overlap = (
            len(cv_skills & job_skills) / len(job_skills) if job_skills else 0.0
        )

        # 2. Seniority match — penalti makin besar untuk gap makin lebar
        sen_diff = abs(cv_seniority - job_seniority)
        seniority = {0: 1.0, 1: 0.7, 2: 0.4}.get(sen_diff, 0.1)

        # 3. Domain match
        domain = 1.0 if cv_industry == job_industry else 0.2

        # 4. Role match: 1.0 jika sama, 0.5 jika kategori sama, 0.1 jika beda
        if cv_role == job_role:
            role = 1.0
        elif cv_role.split("_")[0] == job_role.split("_")[0]:
            role = 0.5
        else:
            role = 0.1

        c = self.config
        score = (
            c.w_skill_overlap * skill_overlap
            + c.w_seniority * seniority
            + c.w_domain * domain
            + c.w_role * role
        )

        # Optional multiplicative penalties (default OFF di v3, multiplier = 1.0)
        if sen_diff >= 3 and c.severe_seniority_gap_multiplier < 1.0:
            score *= c.severe_seniority_gap_multiplier
        if cv_industry != job_industry and c.cross_domain_multiplier < 1.0:
            score *= c.cross_domain_multiplier

        return float(np.clip(score, 0.0, 1.0))

    # ---------- Mode generators ----------

    def _gen_high_match(self, row: pd.Series, job_skills_full: List[str]) -> Tuple[str, dict]:
        """High match: CV punya 50–90% skill job, industri & seniority match."""
        ratio = self.rng.uniform(self.config.pos_overlap_min, self.config.pos_overlap_max)
        n = max(1, int(len(job_skills_full) * ratio))
        picked = self.rng.choice(job_skills_full, size=n, replace=False).tolist()

        meta = {
            "cv_skills_set": {s.lower() for s in picked},
            "cv_seniority": int(row["_seniority"]),
            "cv_industry": row["industry_name"],
            "cv_role": row["_role_simple"],
        }
        years = max(1, int(row["_seniority"]) * 2 + int(self.rng.integers(-1, 3)))
        cv_text = self._build_cv(
            str(row["title"]), picked, years, int(row["_seniority"]), row["industry_name"]
        )
        return cv_text, meta

    def _gen_partial(self, row: pd.Series, job_skills_full: List[str]) -> Tuple[str, dict]:
        """Partial: CV punya 10–40% skill + ekstra dari job lain (v4: lower ratio)."""
        n_keep = max(
            1,
            int(len(job_skills_full) * self.rng.uniform(
                self.config.partial_skill_ratio_min,
                self.config.partial_skill_ratio_max
            ))
        )
        kept = self.rng.choice(job_skills_full, size=n_keep, replace=False).tolist()

        # Skill ekstra dari job random (mungkin domain berbeda)
        other_idx = int(self.rng.choice(self.df.index))
        other_skills = self.df.loc[other_idx, "required_skills"]
        if isinstance(other_skills, list) and other_skills:
            n_extra = min(3, len(other_skills))
            extra = self.rng.choice(other_skills, size=n_extra, replace=False).tolist()
        else:
            extra = []
        cv_skills = list(set(kept + extra))

        meta = {
            "cv_skills_set": {s.lower() for s in cv_skills},
            "cv_seniority": int(row["_seniority"]),
            "cv_industry": row["industry_name"],
            "cv_role": row["_role_simple"],
        }
        years = max(1, int(row["_seniority"]) * 2)
        cv_text = self._build_cv(
            str(row["title"]), cv_skills, years, int(row["_seniority"]), row["industry_name"]
        )
        return cv_text, meta

    def _gen_seniority_mismatch(
        self, row: pd.Series, job_skills_full: List[str]
    ) -> Tuple[str, dict]:
        """Seniority mismatch: skill match tapi level beda jauh."""
        ratio = self.rng.uniform(self.config.pos_overlap_min, self.config.pos_overlap_max)
        n = max(1, int(len(job_skills_full) * ratio))
        picked = self.rng.choice(job_skills_full, size=n, replace=False).tolist()

        # Flip seniority
        job_sen = int(row["_seniority"])
        cv_sen = 1 if job_sen >= 3 else 4

        meta = {
            "cv_skills_set": {s.lower() for s in picked},
            "cv_seniority": cv_sen,
            "cv_industry": row["industry_name"],
            "cv_role": row["_role_simple"],
        }
        years = max(1, cv_sen * 2 + int(self.rng.integers(-1, 2)))
        cv_title = self._adjust_title_seniority(str(row["title"]), cv_sen)
        cv_text = self._build_cv(cv_title, picked, years, cv_sen, row["industry_name"])
        return cv_text, meta

    def _gen_same_domain_diff_role(
        self, row: pd.Series, job_skills_full: List[str]
    ) -> Optional[Tuple[str, dict]]:
        """Same industry, different role category."""
        industry = row["industry_name"]
        roles_in_ind = self.industry_role_idx.get(industry, {})
        candidate_roles = [
            r for r in roles_in_ind
            if r != row["_role_simple"] and roles_in_ind[r]
        ]
        if not candidate_roles:
            return None

        other_role = str(self.rng.choice(candidate_roles))
        other_idx = int(self.rng.choice(roles_in_ind[other_role]))
        other_row = self.df.loc[other_idx]

        other_skills = (
            other_row["required_skills"]
            if isinstance(other_row["required_skills"], list)
            else []
        )
        other_kw = self._extract_keywords(
            other_row["job_description"], self.config.topn_keywords_extra
        )
        cv_skills = list(set(other_skills + other_kw))
        if not cv_skills:
            return None

        meta = {
            "cv_skills_set": {s.lower() for s in cv_skills},
            "cv_seniority": int(other_row["_seniority"]),
            "cv_industry": industry,
            "cv_role": other_role,
        }
        years = max(1, int(other_row["_seniority"]) * 2)
        cv_text = self._build_cv(
            str(other_row["title"]), cv_skills, years, int(other_row["_seniority"]), industry
        )
        return cv_text, meta

    def _gen_cross_domain(
        self, row: pd.Series, job_skills_full: List[str]
    ) -> Optional[Tuple[str, dict]]:
        """Cross-domain: CV dari industri yang berbeda."""
        other_inds = [i for i in self.valid_industries if i != row["industry_name"]]
        if not other_inds:
            return None
        other_ind = str(self.rng.choice(other_inds))
        other_idx = int(self.rng.choice(self.industry_to_indices[other_ind]))
        other_row = self.df.loc[other_idx]

        other_skills = (
            other_row["required_skills"]
            if isinstance(other_row["required_skills"], list)
            else []
        )
        other_kw = self._extract_keywords(
            other_row["job_description"], self.config.topn_keywords_extra
        )
        cv_skills = list(set(other_skills + other_kw))
        if not cv_skills:
            return None

        meta = {
            "cv_skills_set": {s.lower() for s in cv_skills},
            "cv_seniority": int(other_row["_seniority"]),
            "cv_industry": other_ind,
            "cv_role": other_row["_role_simple"],
        }
        years = max(1, int(other_row["_seniority"]) * 2)
        cv_text = self._build_cv(
            str(other_row["title"]), cv_skills, years, int(other_row["_seniority"]), other_ind
        )
        return cv_text, meta

    # ---------- Main entry ----------

    def synthesize(self) -> Tuple[List[str], List[str], np.ndarray]:
        """
        Hasilkan training pairs dengan soft labels.

        Returns:
            cv_texts:  List CV strings
            job_texts: List job description strings
            labels:    np.ndarray[float32] di range [0.0, 1.0]
        """
        cv_texts: List[str] = []
        job_texts: List[str] = []
        labels: List[float] = []
        skipped = 0

        # Probabilitas mode (akan dinormalisasi)
        c = self.config
        modes = [
            "high_match", "partial", "seniority_mismatch",
            "same_domain_diff_role", "cross_domain",
        ]
        probs = np.array([
            c.p_high_match, c.p_partial, c.p_seniority_mismatch,
            c.p_same_domain_diff_role, c.p_cross_domain,
        ])
        probs = probs / probs.sum()

        mode_to_gen = {
            "high_match": self._gen_high_match,
            "partial": self._gen_partial,
            "seniority_mismatch": self._gen_seniority_mismatch,
            "same_domain_diff_role": self._gen_same_domain_diff_role,
            "cross_domain": self._gen_cross_domain,
        }

        mode_counts: Dict[str, int] = {m: 0 for m in modes}

        for idx, row in self.df.iterrows():
            job_desc = str(row["job_description"])[:2000]
            req_skills = (
                row["required_skills"]
                if isinstance(row["required_skills"], list)
                else []
            )
            job_kw = self._extract_keywords(job_desc, c.topn_keywords_main)
            job_skills_full = list(set(req_skills + job_kw))

            if len(job_skills_full) < c.min_job_skills:
                skipped += 1
                continue

            job_skills_set = {s.lower() for s in job_skills_full}

            mode = str(self.rng.choice(modes, p=probs))
            result = mode_to_gen[mode](row, job_skills_full)

            if result is None:
                skipped += 1
                continue

            cv_text, meta = result
            mode_counts[mode] += 1

            label = self._compute_soft_label(
                cv_skills=meta["cv_skills_set"],
                job_skills=job_skills_set,
                cv_seniority=meta["cv_seniority"],
                job_seniority=int(row["_seniority"]),
                cv_industry=meta["cv_industry"],
                job_industry=row["industry_name"],
                cv_role=meta["cv_role"],
                job_role=row["_role_simple"],
            )

            cv_texts.append(cv_text)
            job_texts.append(job_desc)
            labels.append(label)

        labels_np = np.array(labels, dtype=np.float32)
        logger.info(
            "Synthesized %d pairs | skipped=%d | mean=%.3f std=%.3f | modes=%s",
            len(labels_np), skipped, labels_np.mean(), labels_np.std(), mode_counts,
        )
        return cv_texts, job_texts, labels_np


# =============================================================================
# Sanity check / demo
# =============================================================================

def quick_sanity_check(synth: "CVJobPairSynthesizer", n_samples: int = 5) -> None:
    """Cetak beberapa contoh pair untuk inspeksi visual."""
    cv_texts, job_texts, labels = synth.synthesize()
    print(f"\n=== Sanity check: {len(labels)} pairs ===")
    print(f"Label distribution: mean={labels.mean():.3f}, "
          f"std={labels.std():.3f}, min={labels.min():.3f}, max={labels.max():.3f}")
    # Histogram bins
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    counts, _ = np.histogram(labels, bins=bins)
    for lo, hi, ct in zip(bins[:-1], bins[1:], counts):
        print(f"  [{lo:.1f}, {hi:.1f}): {ct} ({ct/len(labels)*100:.1f}%)")

    print(f"\n=== Sample {n_samples} pairs ===")
    indices = np.random.choice(len(cv_texts), size=min(n_samples, len(cv_texts)), replace=False)
    for i in indices:
        print(f"\n[label={labels[i]:.3f}]")
        print(f"  CV : {cv_texts[i][:200]}...")
        print(f"  Job: {job_texts[i][:200]}...")