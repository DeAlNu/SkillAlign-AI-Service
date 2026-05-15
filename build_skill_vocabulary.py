"""
Build Skill Vocabulary dari skills_desc column di job_posting.csv.

Perbaikan dari versi sebelumnya:
- Menggunakan kolom skills_desc (bukan job_description penuh)
  → skills_desc berisi daftar skill yang dibutuhkan, bukan deskripsi umum
- Sort berdasarkan FREKUENSI (bukan IDF) — skill yang sering diminta = skill penting
- min_df yang lebih ketat untuk filter company names dan noise yang jarang muncul
- Filter panjang minimum dan noise words yang lebih komprehensif

Catatan limitasi:
- Vocabulary ini mencerminkan industri dalam dataset (LinkedIn US, IT-heavy)
- Untuk industri logistik Indonesia atau healthcare, skill domain-specific
  mungkin tidak tercakup jika tidak ada dalam training data

Output: preprocessors/skill_vocabulary.csv

Usage:
    python build_skill_vocabulary.py
"""

import os
import re
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Config
# ============================================================
JOB_POSTING_PATH = "Dataset/database_design/job_posting.csv"
SKILLS_CSV_PATH  = "Dataset/database_design/skills.csv"
OUTPUT_PATH      = "preprocessors/skill_vocabulary.csv"
MAX_VOCAB_SIZE   = 2000
# Muncul di minimal 0.5% job postings (~620 jobs dari 123k)
# Ini memfilter company names yang hanya muncul di 1-5 posting
MIN_DOC_FREQ     = 0.005
# Tidak muncul di lebih dari 25% posting (terlalu generik)
MAX_DOC_FREQ     = 0.25

# Noise terms yang bukan skill
_NOISE_TERMS = {
    # Kata kerja dan deskriptif
    "work", "working", "team", "role", "position", "candidate",
    "required", "include", "including", "responsible", "responsibilities",
    "strong", "knowledge", "ability", "skills", "skill", "looking",
    "seeking", "proficient", "familiar", "understanding", "high", "level",
    "good", "excellent", "great", "minimum", "plus", "related",
    "based", "degree", "bachelor", "master", "university", "education",
    "job", "company", "business", "organization", "department",
    "support", "manage", "develop", "build", "create", "ensure",
    "provide", "working", "preferred", "proven", "ability",
    # Kata legal/HR
    "equal", "opportunity", "employer", "discrimination", "disability",
    "veteran", "protected", "color", "age", "religion", "gender",
    "race", "national", "origin", "status", "applicable",
    # Kata filler
    "new", "old", "big", "small", "best", "plus", "key", "end",
    "like", "make", "take", "need", "help", "use", "using",
    "years", "year", "experience", "background", "knowledge",
    # Single characters
    "a", "b", "c", "d", "e", "f", "g", "h",
}


def clean_skills_text(text: str) -> str:
    """Bersihkan skills_desc text."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    # Ganti pemisah skill (koma, titik koma, pipa, newline) dengan spasi
    text = re.sub(r'[,;|/\n\r]', ' ', text)
    # Pertahankan tanda khusus untuk tech terms: ., +, #
    text = re.sub(r'[^a-z0-9\s\.\+\#\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_valid_skill_term(term: str) -> bool:
    """Cek apakah term valid sebagai skill name."""
    words = term.split()

    # Minimal 2 karakter per kata
    if any(len(w) < 2 for w in words):
        return False

    # Tidak boleh seluruhnya noise words
    if all(w in _NOISE_TERMS for w in words):
        return False

    # Tidak boleh pure angka
    if term.replace(' ', '').replace('.', '').isdigit():
        return False

    # Bigram: minimal satu kata yang bukan noise
    meaningful_words = [w for w in words if w not in _NOISE_TERMS]
    if not meaningful_words:
        return False

    return True


def extract_from_skills_desc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ekstrak skill vocabulary dari kolom skills_desc.

    Menggunakan CountVectorizer dengan sort berdasarkan FREKUENSI DOKUMEN
    (bukan IDF) — skill yang paling sering diminta = paling penting.
    """
    if "skills_desc" not in df.columns:
        logger.warning("Kolom 'skills_desc' tidak ditemukan. Coba pakai 'job_description'.")
        col = "job_description"
    else:
        col = "skills_desc"
        logger.info(f"Menggunakan kolom: '{col}'")

    texts = df[col].dropna().apply(clean_skills_text)
    texts = texts[texts.str.len() > 5]
    logger.info(f"Memproses {len(texts):,} teks dari kolom '{col}'...")

    vec = CountVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=MIN_DOC_FREQ,
        max_df=MAX_DOC_FREQ,
        max_features=MAX_VOCAB_SIZE * 3,
    )

    logger.info("Fitting CountVectorizer...")
    X = vec.fit_transform(texts)

    feature_names = vec.get_feature_names_out()
    # Hitung document frequency (berapa banyak posting yang menyebut skill ini)
    doc_freq = np.asarray((X > 0).sum(axis=0)).flatten()
    total_docs = X.shape[0]

    logger.info(f"Raw features: {len(feature_names)}")

    records = []
    for term, freq in zip(feature_names, doc_freq):
        if not is_valid_skill_term(term):
            continue
        doc_freq_pct = freq / total_docs
        records.append({
            "skill": term,
            "doc_frequency": int(freq),
            "doc_freq_pct": round(float(doc_freq_pct), 5),
        })

    result_df = pd.DataFrame(records)
    # Sort: frekuensi tinggi = skill yang sering diminta = lebih penting
    result_df = result_df.sort_values("doc_frequency", ascending=False)
    result_df = result_df.head(MAX_VOCAB_SIZE).reset_index(drop=True)

    logger.info(f"Final vocabulary size: {len(result_df)}")
    return result_df


def load_linkedin_skills(skills_path: str) -> list:
    """Load dari skills.csv (LinkedIn taxonomy)."""
    if not os.path.exists(skills_path):
        return []
    try:
        df = pd.read_csv(skills_path)
        col = "skill_name" if "skill_name" in df.columns else df.columns[1]
        skills = df[col].dropna().str.lower().str.strip().tolist()
        logger.info(f"LinkedIn skills loaded: {len(skills)}")
        return skills
    except Exception as e:
        logger.warning(f"Tidak bisa baca skills.csv: {e}")
        return []


def main():
    os.makedirs("preprocessors", exist_ok=True)

    if not os.path.exists(JOB_POSTING_PATH):
        logger.error(f"File tidak ditemukan: {JOB_POSTING_PATH}")
        return

    logger.info(f"Loading: {JOB_POSTING_PATH}")

    # Baca kolom yang diperlukan saja untuk efisiensi
    sample = pd.read_csv(JOB_POSTING_PATH, nrows=1)
    available_cols = sample.columns.tolist()
    logger.info(f"Available columns: {available_cols}")

    use_cols = [c for c in ["skills_desc", "job_description"] if c in available_cols]
    df = pd.read_csv(JOB_POSTING_PATH, usecols=use_cols)
    logger.info(f"Loaded {len(df):,} rows.")

    # Ekstrak dari skills_desc (lebih bersih) atau fallback ke job_description
    vocab_df = extract_from_skills_desc(df)

    # Tambahkan LinkedIn skills
    linkedin_skills = load_linkedin_skills(SKILLS_CSV_PATH)
    if linkedin_skills:
        extra_df = pd.DataFrame({
            "skill": linkedin_skills,
            "doc_frequency": 0,
            "doc_freq_pct": 0.0,
        })
        # Gabungkan, buang duplikat, pertahankan yang dari dataset
        vocab_df = pd.concat([vocab_df, extra_df]).drop_duplicates(
            subset="skill", keep="first"
        ).reset_index(drop=True)
        logger.info(f"Total setelah gabung LinkedIn: {len(vocab_df)}")

    vocab_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"\nSaved to: {OUTPUT_PATH}")
    logger.info(f"Top 30 skills (most frequently required):")
    print(vocab_df[["skill", "doc_frequency", "doc_freq_pct"]].head(30).to_string(index=False))

    logger.info("\n--- Limitasi ---")
    logger.info("Vocabulary ini mencerminkan job postings LinkedIn US (IT-heavy).")
    logger.info("Skill domain spesifik (logistik Indonesia, healthcare, dsb.)")
    logger.info("mungkin tidak tercakup jika tidak ada dalam training data.")


if __name__ == "__main__":
    main()
