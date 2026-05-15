"""
CV Profile Extractor untuk SkillAlign AI.

Mengekstrak informasi profil kandidat dari teks CV:
- current_role : jabatan/posisi saat ini, termasuk seniority
                 (misal "Senior Software Engineer", "Lead Data Scientist")
- experience   : total tahun pengalaman (misal "4 Years")
- education    : gelar pendidikan tertinggi (misal "B.S. Computer Science")

Strategi seniority:
  Seniority (Senior/Junior/Lead/dll) dideteksi TERPISAH dari base role,
  lalu digabungkan secara dinamis. Ini menghindari keharusan mendaftarkan
  semua kombinasi (Senior Frontend, Junior Frontend, dll.) di role_config.py.

Menggunakan ROLE_PATTERNS + SENIORITY_LEVELS dari hybrid_scorer agar
konsisten dengan pipeline scoring.
"""

import re
import logging
from datetime import date
from dataclasses import dataclass
from typing import Optional, List, Tuple

from src.inference.role_config import ROLE_DISPLAY

logger = logging.getLogger(__name__)


# =============================================================================
# Seniority prefix map
# Mapping dari seniority label (hybrid_scorer) → prefix tampilan.
# Hanya level yang relevan sebagai prefix; executive & mid_to_senior
# biasanya sudah menjadi bagian dari title itu sendiri.
# =============================================================================

_SENIORITY_PREFIX: dict = {
    "junior":       "Junior",
    "senior":       "Senior",
    "mid_to_senior": "Lead",   # architect/manager → "Lead ..."
    # "executive" sengaja tidak dimap — Director/VP/CTO sudah terbaca
    # sebagai bagian dari role key via ROLE_PATTERNS
}


# =============================================================================
# Degree priority map (semakin tinggi nilai = semakin tinggi gelar)
# =============================================================================

_DEGREE_PATTERNS: List[Tuple[int, str, str]] = [
    # (priority, label, regex_pattern)
    (4, "Ph.D.",    r"\b(ph\.?d\.?|doctor(?:ate)?|d\.phil\.?)\b"),
    (3, "Master",   r"\b(m\.?s\.?c?\.?|master(?:\'?s)?|m\.?eng\.?|m\.?tech\.?|m\.?res\.?)\b"),
    (3, "MBA",      r"\b(m\.?b\.?a\.?)\b"),
    (2, "Bachelor", r"\b(b\.?s\.?c?\.?|b\.?a\.?|bachelor(?:\'?s)?|b\.?eng\.?|b\.?tech\.?)\b"),
    (1, "Diploma",  r"\b(diploma|associate(?:\'?s)?|a\.?a\.?s?\.?|a\.?s\.?)\b"),
]

# Bidang studi yang sering muncul setelah gelar (untuk disambiguasi)
_FIELD_PATTERN = (
    r"(?:\s+(?:of\s+|in\s+)?)"
    r"((?:[A-Z][a-zA-Z]+\s*){1,5})"
)

# =============================================================================
# Regex pola pengalaman kerja
# =============================================================================

# Pola eksplisit "X years (of) experience"
_EXPERIENCE_PATTERNS: List[str] = [
    r"\b(\d+)\+?\s*years?\s+of\s+(?:professional\s+|work\s+|industry\s+|relevant\s+)?experience\b",
    r"\bexperience\s+of\s+(?:over\s+|more\s+than\s+)?(\d+)\+?\s*years?\b",
    r"\b(\d+)\+?\s*years?\s+(?:professional\s+)?(?:working\s+)?experience\b",
    r"\bover\s+(\d+)\s*years?\s+(?:of\s+)?experience\b",
    r"\b(\d+)\+?\s*(?:yr|yrs)\.?\s+(?:of\s+)?experience\b",
    # Pola generik: "X years in ...", "X years as ..."
    r"\b(\d+)\+?\s*years?\s+(?:in\s+(?:the\s+)?|as\s+(?:a\s+)?)[a-z]",
    # Experienced/Experience with X years
    r"\bwith\s+(\d+)\+?\s*years?\b",
]

# Pola rentang tahun: "2020 - 2024" atau "2020 – Present"
_DATE_RANGE_PATTERN = r"\b(\d{4})\s*[-\u2013\u2014]\s*(\d{4}|present|current|now|sekarang)\b"

# Label fallback untuk mahasiswa / fresh graduate
_STUDENT_PATTERNS: List[str] = [
    r"\b(student|mahasiswa|undergraduate|fresh\s+graduate|entry[\s\-]?level)\b",
]

# =============================================================================
# Section header patterns — untuk section-aware parsing
# =============================================================================

# Pola header yang HANYA menandai seksi KERJA (bukan organisasi/volunteer)
# Memerlukan prefix: work/professional/relevant/employment/career
_WORK_SECTION_HEADER_PAT = re.compile(
    r"(?im)"
    r"^[\s\*\-\u2022]*"   # opsional: bullet / whitespace di awal
    r"(?:"
    r"(?:work|professional|relevant|related|employment|career)"  # prefix wajib
    r"[\s&/\-]*"
    r"(?:experience|history|background|summary)?"
    r"|experience"        # atau "experience" berdiri sendiri (tanpa prefix)
    r")"
    r"[\s]*[:\-]?\s*$"   # akhir baris
)

# Pola header section APAPUN (untuk mendeteksi akhir seksi kerja)
_ANY_SECTION_HEADER_PAT = re.compile(
    r"(?im)"
    r"^[\s\*\-\u2022]*"
    r"(?:"
    # Bahasa Inggris
    r"education(?:al)?|academic|qualification|skills?|technical\s+skills?"
    r"|training|certific|project|portfolio|volunteer|award|reference"
    r"|summary|objective|language|interest|publication|honor"
    r"|organizational|leadership|activities|internship"
    # Bahasa Indonesia
    r"|pendidikan|keahlian|sertif|proyek|pelatihan|organisasi|portofolio"
    r"|ringkasan|penghargaan"
    r")"
    r"[\w\s&/\-]*[:\-]?\s*$"
)


@dataclass
class CvProfile:
    """Hasil ekstraksi profil kandidat dari teks CV."""
    current_role: Optional[str] = None
    experience:   Optional[str] = None
    education:    Optional[str] = None


class CvProfileExtractor:
    """
    Mengekstrak profil kandidat dari teks CV menggunakan regex.

    Menggunakan ROLE_PATTERNS dari hybrid_scorer untuk konsistensi
    deteksi role dengan pipeline scoring.

    Example:
        >>> extractor = CvProfileExtractor()
        >>> profile = extractor.extract(cv_text)
        >>> print(profile.current_role)   # "Frontend Developer"
        >>> print(profile.experience)     # "4 Years"
        >>> print(profile.education)      # "B.S. Computer Science"
    """

    def __init__(self) -> None:
        # Import saat instantiasi untuk hindari circular import
        from src.inference.hybrid_scorer import ROLE_PATTERNS, SENIORITY_LEVELS
        self._role_patterns: List[Tuple[str, str]] = ROLE_PATTERNS
        self._seniority_levels: List[Tuple[int, str, str]] = SENIORITY_LEVELS

    # ------------------------------------------------------------------
    # Role extraction
    # ------------------------------------------------------------------

    def extract_current_role(self, cv_text: str) -> Optional[str]:
        """
        Deteksi jabatan/posisi saat ini dari teks CV, termasuk seniority.

        Strategi:
        1. Cari di 5 baris pertama (header CV biasanya berisi title)
        2. Cari setelah label eksplisit: "Position:", "Role:", dst.
        3. Scan seluruh teks vs ROLE_PATTERNS, ambil match pertama

        Setelah base role ditemukan, deteksi seniority dari teks yang sama
        (Senior/Junior/Lead/dll) lalu gabungkan: "Senior Software Engineer".
        """
        if not cv_text or not isinstance(cv_text, str):
            return None

        t = cv_text.strip()

        # Prioritas 1: baris pertama CV (sering berisi nama + title)
        header = "\n".join(t.splitlines()[:5])
        base_role = self._detect_base_role(header)
        if base_role:
            prefix = self._detect_seniority_prefix(header)
            return f"{prefix} {base_role}".strip() if prefix else base_role

        # Prioritas 2: label eksplisit
        label_pattern = (
            r"(?:current\s+)?(?:position|role|title|designation)"
            r"\s*[:\-–]\s*([^\n,;]{3,50})"
        )
        match = re.search(label_pattern, t, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            base_role = self._detect_base_role(candidate)
            if base_role:
                prefix = self._detect_seniority_prefix(candidate)
                return f"{prefix} {base_role}".strip() if prefix else base_role

        # Prioritas 3: scan seluruh teks
        base_role = self._detect_base_role(t)
        if base_role:
            prefix = self._detect_seniority_prefix(t)
            return f"{prefix} {base_role}".strip() if prefix else base_role

        return None

    def _detect_base_role(self, text: str) -> Optional[str]:
        """Cocokkan teks dengan ROLE_PATTERNS, kembalikan display name base role."""
        t_lower = text.lower()
        for role_key, pattern in self._role_patterns:
            if re.search(pattern, t_lower):
                return ROLE_DISPLAY.get(role_key, role_key.replace("_", " ").title())
        return None

    def _detect_seniority_prefix(self, text: str) -> Optional[str]:
        """
        Deteksi prefix seniority dari teks (Senior / Junior / Lead / dll).

        Menggunakan SENIORITY_LEVELS dari hybrid_scorer (reuse, tidak duplikat).
        Hanya kembalikan prefix yang bermakna sebagai awalan title;
        level 'executive' tidak dimap karena title-nya sudah spesifik
        (Director, VP, CTO, dsb.) dan ditangkap oleh ROLE_PATTERNS.

        Returns:
            Prefix string ("Senior", "Junior", "Lead") atau None.
        """
        t_lower = text.lower()
        for _level, label, pattern in self._seniority_levels:
            if label not in _SENIORITY_PREFIX:
                continue   # skip 'executive' — sudah tertangani di role key
            if re.search(pattern, t_lower):
                return _SENIORITY_PREFIX[label]
        return None

    def _extract_work_section_text(self, cv_text: str) -> Optional[str]:
        """
        Ekstrak teks dari seksi KERJA di CV (Work Experience, Employment, dll.).

        Mencari header seksi kerja, lalu mengambil isi hingga header seksi
        berikutnya. Mengembalikan None jika tidak ada seksi kerja ditemukan.

        Ini memastikan date-range inference HANYA berjalan pada tanggal
        yang berasal dari riwayat kerja, bukan dari EDUCATION atau PROJECTS.
        """
        # Cari awal seksi kerja
        work_match = _WORK_SECTION_HEADER_PAT.search(cv_text)
        if not work_match:
            return None

        text_after_header = cv_text[work_match.end():]

        # Cari header seksi berikutnya → batas akhir seksi kerja
        next_section = _ANY_SECTION_HEADER_PAT.search(text_after_header)
        if next_section:
            return text_after_header[:next_section.start()]

        return text_after_header

    # ------------------------------------------------------------------
    # Experience extraction
    # ------------------------------------------------------------------

    def extract_experience(self, cv_text: str) -> Optional[str]:
        """
        Deteksi total pengalaman KERJA dari teks CV.

        Strategi (berurutan):
        1. Pola eksplisit "X years of experience" di seluruh teks
           → paling presisi, tidak perlu tahu seksi mana
        2. Section-aware date-range inference
           → isolasi seksi kerja dulu, BARU hitung rentang tahun
           → hindari salah hitung tanggal dari seksi EDUCATION/PROJECTS
        3. Fallback label → "Fresh Graduate" jika student/mahasiswa
           dan tidak ada seksi kerja

        Output: "X Years", "X+ Years", "Fresh Graduate", atau None.
        """
        if not cv_text or not isinstance(cv_text, str):
            return None

        t = cv_text.lower()
        found_years: List[int] = []
        has_plus: bool = False

        # -----------------------------------------------------------
        # Strategi 1: pola eksplisit
        # -----------------------------------------------------------
        for pattern in _EXPERIENCE_PATTERNS:
            for match in re.finditer(pattern, t, re.IGNORECASE):
                try:
                    val = int(match.group(1))
                    if 0 < val <= 50:
                        found_years.append(val)
                        if "+" in match.group(0):
                            has_plus = True
                except (IndexError, ValueError):
                    continue

        if found_years:
            suffix = "+ Years" if has_plus else " Years"
            return f"{max(found_years)}{suffix}"

        # -----------------------------------------------------------
        # Strategi 2: section-aware date-range inference
        # Hanya scan rentang tahun dari seksi KERJA, bukan EDUCATION
        # -----------------------------------------------------------
        work_text = self._extract_work_section_text(cv_text)
        if work_text:
            current_year = date.today().year
            total_months = 0
            has_current = False

            for m in re.finditer(_DATE_RANGE_PATTERN, work_text, re.IGNORECASE):
                start_yr = int(m.group(1))
                end_raw  = m.group(2).strip()
                if re.match(r"present|current|now|sekarang", end_raw, re.IGNORECASE):
                    end_yr = current_year
                    has_current = True
                else:
                    try:
                        end_yr = int(end_raw)
                    except ValueError:
                        continue
                if 1980 <= start_yr <= current_year and start_yr <= end_yr:
                    total_months += (end_yr - start_yr) * 12

            if total_months > 0:
                total_yr = max(1, total_months // 12)
                suffix = "+ Years" if has_current else " Years"
                return f"{total_yr}{suffix}"

        # -----------------------------------------------------------
        # Strategi 3: fallback untuk mahasiswa / fresh graduate
        # Hanya aktif jika tidak ada seksi kerja yang terdeteksi
        # -----------------------------------------------------------
        if work_text is None:   # tidak ada seksi kerja → kemungkinan student
            for pattern in _STUDENT_PATTERNS:
                if re.search(pattern, t, re.IGNORECASE):
                    return "Fresh Graduate"

        return None

    # ------------------------------------------------------------------
    # Education extraction
    # ------------------------------------------------------------------

    def extract_education(self, cv_text: str) -> Optional[str]:
        """
        Deteksi gelar pendidikan tertinggi dari teks CV.

        Prioritas: PhD > Master/MBA > Bachelor > Diploma
        Mencoba menyertakan nama bidang studi setelah gelar.
        """
        if not cv_text or not isinstance(cv_text, str):
            return None

        best_priority = -1
        best_label: Optional[str] = None

        for priority, label, degree_re in _DEGREE_PATTERNS:
            if priority <= best_priority:
                continue  # sudah ada yang lebih tinggi
            match = re.search(degree_re, cv_text, re.IGNORECASE)
            if not match:
                continue

            # Coba ambil bidang studi setelah gelar
            # Fix: lookahead untuk terminator agar field bisa diambil
            # dengan benar walau diikuti (tahun), koma, titik, newline, dll.
            end_pos = match.end()
            tail = cv_text[end_pos: end_pos + 80]
            field_match = re.match(
                r"[\s\.,\-]*(?:in\s+|of\s+)?([A-Za-z][A-Za-z\s&]{2,40}?)"
                r"(?=\s*[\(,;\.\n\d]|\s*$)",
                tail
            )

            if field_match:
                field = field_match.group(1).strip()
                # Hapus noise words di akhir
                field = re.sub(
                    r"\b(?:university|college|institute|school|from|at|the)\b.*$",
                    "", field, flags=re.IGNORECASE
                ).strip().rstrip(",. ")
                if len(field) > 2:
                    best_label = f"{label} {field}"
                else:
                    best_label = label
            else:
                best_label = label

            best_priority = priority

        return best_label

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, cv_text: str) -> CvProfile:
        """
        Ekstrak semua atribut profil dari satu teks CV.

        Args:
            cv_text: Teks CV mentah (raw text dari PDF atau input user).

        Returns:
            CvProfile dengan current_role, experience, dan education.
            Field yang gagal dideteksi akan bernilai None.
        """
        return CvProfile(
            current_role=self.extract_current_role(cv_text),
            experience=self.extract_experience(cv_text),
            education=self.extract_education(cv_text),
        )
