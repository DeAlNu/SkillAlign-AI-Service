"""
02_model_development.py - Semantic Matching & Information Extraction Pivot

Arsitektur Baru:
1. Data Preparation: Menggunakan teks asli (real-world dataset).
2. Information Extraction: Menggunakan NLP sederhana/Dictionary untuk mengekstrak poin penting (Skill).
3. Semantic Embedding: Menggunakan Sentence-Transformers (all-MiniLM-L6-v2).
4. Matcher: Menggabungkan Cosine Similarity dari embedding dan Jaccard Similarity dari extracted skills.
5. Evaluasi: Uji coba langsung menggunakan inference_test_cases.csv untuk menghindari overfitting.
"""

import pandas as pd
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, util
import re
import os

print("=== Modul Semantic Matching & IE Initialized ===")

# ---------------------------------------------------------
# 1. DATA PREPARATION (Real Data)
# ---------------------------------------------------------
def load_datasets(job_posting_path, skill_dictionary_path):
    """
    Muat dataset dunia nyata. Tidak lagi menggunakan generate template sintetis.
    """
    print(f"Membaca dataset dari {job_posting_path}...")
    try:
        df_posting = pd.read_csv(job_posting_path, usecols=['job_posting_id', 'title', 'job_description', 'skills_desc'])
        df_skills = pd.read_csv(skill_dictionary_path)
        print("Dataset berhasil dimuat!")
        return df_posting, df_skills
    except Exception as e:
        print(f"Gagal memuat dataset: {e}")
        return None, None

# ---------------------------------------------------------
# 2. INFORMATION EXTRACTION MODULE
# ---------------------------------------------------------
class SkillExtractor:
    """Modul untuk mengekstrak poin-poin skill dari teks mentah."""
    def __init__(self, skill_dict=None):
        # Memuat model dasar SpaCy untuk tokenisasi ringan (pastikan spacy download en_core_web_sm sudah dijalankan)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Peringatan: SpaCy model 'en_core_web_sm' belum diunduh. Silakan jalankan 'python -m spacy download en_core_web_sm'")
            self.nlp = None

        # Memuat dictionary skill jika diberikan
        self.skill_dictionary = set()
        if skill_dict is not None and 'skill_name' in skill_dict.columns:
            self.skill_dictionary = set(skill_dict['skill_name'].dropna().str.lower().tolist())

    def extract_skills(self, text):
        """Mengekstrak kata kunci/skills dari teks."""
        if not isinstance(text, str):
            return set()
            
        text_lower = text.lower()
        extracted = set()
        
        # Jika kita memiliki kamus skill (dari skills.csv), kita bisa mencari irisannya secara eksplisit (Exact Match)
        if self.skill_dictionary:
            # Tokenisasi sederhana
            words = set(re.findall(r'\b\w+\b', text_lower))
            extracted = words.intersection(self.skill_dictionary)
        else:
            # Fallback jika tidak ada dictionary: Ekstrak noun phrases menggunakan Spacy
            if self.nlp:
                doc = self.nlp(text_lower)
                extracted = set(chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 2)
                
        return extracted

# ---------------------------------------------------------
# 3. SEMANTIC EMBEDDING MODEL (Transformer)
# ---------------------------------------------------------
class SemanticEncoder:
    """Modul untuk mengonversi teks panjang menjadi Vektor Semantik 384-Dimensi."""
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading SentenceTransformer: {model_name}...")
        # Model ini sudah pre-trained pada jutaan pasang teks untuk tugas semantic similarity
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts):
        """Mengubah daftar teks menjadi vector embeddings"""
        return self.model.encode(texts, convert_to_tensor=True)

# ---------------------------------------------------------
# 4. HYBRID MATCHER (Scoring System)
# ---------------------------------------------------------
class HybridMatcher:
    """Modul pembanding utama yang mengombinasikan Opsi A dan Opsi B."""
    def __init__(self, extractor: SkillExtractor, encoder: SemanticEncoder):
        self.extractor = extractor
        self.encoder = encoder
        
    def calculate_jaccard_similarity(self, set_a, set_b):
        """Hitung irisan skill yang sama."""
        if not set_a and not set_b: return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return float(intersection) / union if union > 0 else 0.0

    def predict_match(self, cv_text, job_desc):
        """Kalkulasi kecocokan CV vs Pekerjaan"""
        # A. Semantic Score (Pemahaman makna bahasa natural keseluruhan)
        emb_cv = self.encoder.encode([cv_text])
        emb_job = self.encoder.encode([job_desc])
        semantic_score = util.cos_sim(emb_cv, emb_job).item()
        
        # Pastikan skor tidak negatif
        semantic_score = max(0.0, semantic_score)
        
        # B. Extraction Score (Pencocokan kata kunci mutlak)
        cv_skills = self.extractor.extract_skills(cv_text)
        job_skills = self.extractor.extract_skills(job_desc)
        extraction_score = self.calculate_jaccard_similarity(cv_skills, job_skills)
        
        # C. Gabungkan Skor: 70% Makna Semantik Konteks + 30% Kata Kunci Skill Eksplisit
        # (Bobot dapat di-fine-tune sesuai kebutuhan domain)
        final_score = (0.7 * semantic_score) + (0.3 * extraction_score)
        
        # Penentuan Confidence & Recommendation
        confidence = "High" if final_score > 0.75 else "Medium" if final_score > 0.45 else "Low"
        rec = "Highly Recommended" if final_score > 0.75 else "Recommended" if final_score > 0.45 else "Not Recommended"
        
        return {
            "score": final_score,
            "semantic_score": semantic_score,
            "extraction_score": extraction_score,
            "cv_extracted_skills": list(cv_skills)[:10],  # Tampilkan max 10 untuk log
            "job_extracted_skills": list(job_skills)[:10],
            "confidence": confidence,
            "recommendation": rec
        }

# ---------------------------------------------------------
# 5. INFERENCE & EVALUATION (Zero-Shot pada Real Data)
# ---------------------------------------------------------
def run_evaluation(test_csv_path, skill_dict_path=None):
    """
    Mengevaluasi pipeline baru secara langsung (Zero-Shot) pada data test.
    Kita akan buktikan bahwa masalah over-predicting di CNN sudah teratasi.
    """
    print(f"\n=== Memulai Evaluasi Zero-Shot pada Test Cases ===")
    try:
        df_test = pd.read_csv(test_csv_path)
    except FileNotFoundError:
        print(f"[Error] File {test_csv_path} tidak ditemukan.")
        return
        
    df_skills = None
    if skill_dict_path and os.path.exists(skill_dict_path):
        df_skills = pd.read_csv(skill_dict_path)
    
    # Inisialisasi komponen Pipeline
    extractor = SkillExtractor(skill_dict=df_skills)
    encoder = SemanticEncoder()
    matcher = HybridMatcher(extractor, encoder)
    
    print(f"Menguji {len(df_test)} data inference...")
    print("-" * 60)
    
    for idx, row in df_test.iterrows():
        cv_text = row.get('cv_text', '')
        job_text = row.get('job_description', '')
        expected_label = row.get('label', 'N/A')  # Ground truth label jika ada
        
        # Melakukan proses inference Hybrid!
        result = matcher.predict_match(cv_text, job_text)
        
        print(f"[Test Case {idx+1}]")
        print(f"  CV Preview    : {str(cv_text)[:80]}...")
        print(f"  Job Preview   : {str(job_text)[:80]}...")
        print(f"  Expected      : {'MATCH' if expected_label in [1, '1', 1.0] else 'NOT MATCH'}")
        print(f"  ----------------")
        print(f"  Final Score   : {result['score']:.4f}")
        print(f"  Semantic Part : {result['semantic_score']:.4f}")
        print(f"  Extract Part  : {result['extraction_score']:.4f}")
        print(f"  CV Skills Ext : {result['cv_extracted_skills']}")
        print(f"  Status        : {result['recommendation']} ({result['confidence']} Confidence)\n")

if __name__ == "__main__":
    # Jalur absolut dari repository agar aman dieksekusi di mana saja
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Path test cases (File yang digunakan pengguna untuk uji coba gagalnya CNN sebelumnya)
    test_file_path = os.path.join(base_dir, 'tests', 'test_data', 'inference_test_cases.csv')
    
    # Kamus skill opsional untuk Information Extraction (Opsi A)
    skill_dict_path = os.path.join(base_dir, 'Dataset', 'database_design', 'skills.csv')
    
    run_evaluation(test_file_path, skill_dict_path)
