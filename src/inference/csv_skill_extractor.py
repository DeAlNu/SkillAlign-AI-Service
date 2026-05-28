import csv
import os
import re
import logging
from typing import Dict, Set, Tuple, List

logger = logging.getLogger(__name__)

class CsvSkillExtractor:
    """
    Ekstraktor skill berbasis aturan menggunakan database pekerjaan industri
    dari file Dataset/job_skill_map.csv.
    
    Menyediakan ekstraksi skill yang sangat cepat (sub-50ms) dan hemat memori
    karena tidak bergantung pada model deep learning atau spaCy.
    """
    
    def __init__(self, csv_path: str = None) -> None:
        if csv_path is None:
            # Dapatkan path relatif terhadap file ini
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # current_dir = project_root/src/inference/
            # naik 2 level ke project_root
            project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
            csv_path = os.path.join(project_root, "src/database/", "job_skill_map.csv")
            
        self.csv_path = csv_path
        self.skills_db: Set[str] = set()
        self.skills_list: List[str] = []
        self.compiled_skills: List[Tuple[str, re.Pattern]] = []
        
        self.load_skills()
        
    def load_skills(self) -> None:
        if not os.path.exists(self.csv_path):
            logger.error(f"Dataset CSV tidak ditemukan di: {self.csv_path}")
            return
            
        logger.info(f"Loading skills database dari {self.csv_path}...")
        try:
            with open(self.csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Ambil dari keempat kolom skill
                    for col in ["skills_core", "skills_common", "skills_optional", "soft_skills"]:
                        if col in row and row[col]:
                            skills = row[col].split("|")
                            for s in skills:
                                s_clean = s.strip().lower()
                                if s_clean:
                                    self.skills_db.add(s_clean)
                                    
            # Urutkan berdasarkan panjang karakter (descending)
            # Penting untuk overlap-prevention agar kata yang lebih panjang dicocokkan terlebih dahulu
            self.skills_list = sorted(list(self.skills_db), key=len, reverse=True)
            
            # Pre-compile regex patterns dengan boundary aman untuk karakter IT (misal C#, .NET)
            self.compiled_skills = []
            for s in self.skills_list:
                s_escaped = re.escape(s)
                # Gunakan pattern custom word boundary seperti di hybrid_scorer.py
                pattern = re.compile(rf"(?:^|[^a-z0-9_]){s_escaped}(?:$|[^a-z0-9_])", re.IGNORECASE)
                self.compiled_skills.append((s, pattern))
                
            logger.info(f"Berhasil memuat {len(self.skills_list)} skill unik dari CSV.")
            
        except Exception as e:
            logger.error(f"Gagal memuat skills dari CSV: {str(e)}", exc_info=True)

    def extract_skills(self, text: str) -> Dict[str, Tuple[str, float]]:
        """
        Ekstrak skill dari teks berdasarkan kamus CSV.
        
        Returns:
            Dict[skill_id, (skill_name, confidence)]
            Di mana skill_id adalah lowercase nama skill dengan spasi diganti underscore.
            Confidence selalu 1.0 (exact dictionary match).
        """
        if not text or len(text.strip()) < 10:
            return {}
            
        text_lower = text.lower()
        extracted: Dict[str, Tuple[str, float]] = {}
        matched_intervals: List[Tuple[int, int]] = []
        
        for s, pattern in self.compiled_skills:
            for match in pattern.finditer(text_lower):
                start, end = match.span()
                matched_str = match.group(0)
                
                # Cari index sebenarnya dari nama skill di dalam teks kecocokan regex
                offset = matched_str.lower().find(s)
                if offset != -1:
                    actual_start = start + offset
                    actual_end = actual_start + len(s)
                else:
                    actual_start, actual_end = start, end
                    
                # Cek overlap dengan interval yang sudah terdaftar
                is_overlap = False
                for ms, me in matched_intervals:
                    if not (actual_end <= ms or actual_start >= me):
                        is_overlap = True
                        break
                        
                if not is_overlap:
                    matched_intervals.append((actual_start, actual_end))
                    sid = s.replace(" ", "_")
                    extracted[sid] = (s, 1.0)
                    
        return extracted
