# SkillAlign AI Service

> **CV-Job Matching menggunakan Deep Learning & NLP**  
> Capstone Project — DBS Foundation Coding Camp 2026  
> Tim ID: CC26-PSU318

## 📋 Overview

SkillAlign adalah AI Service berbasis **Deep Learning** untuk **CV-Job Matching**. Model menerima teks CV dan teks job description, lalu menghasilkan skor kecocokan (0.0–1.0). Service ini berperan sebagai *scoring engine* yang dikonsumsi oleh Backend (Express.js) dalam alur **Two-Stage Retrieval + Re-Ranking**.

### Fitur Utama

| Fitur | Deskripsi |
|---|---|
| **CV-Job Matching** | Skor kecocokan 0.0–1.0 berbasis neural network |
| **Batch Endpoint** | Scoring 1 CV terhadap hingga 50 job sekaligus |
| **Skill Gap Analysis** | Deteksi skill yang dimiliki vs yang dibutuhkan job (SkillNer EMSI ~6k skills) |
| **Analyze CV** | Ekstraksi profil kandidat + saran job title |
| **Recommend Jobs** | Ranking job postings + industry skill readiness analysis |
| **Learning Path** | Rekomendasi kursus Coursera/YouTube via Gemini + caching Supabase |
| **HybridScorer** | Gabungan neural score + structured feature score |

---

## 🏗️ Arsitektur Model

### Model v4 — SkillAlignMatcherV4 *(Latest, in notebook)*

```
cv_input (300)                    job_input (300)
      │                                  │
      └──── Shared Embedding (15k vocab × 128 dim, pre-trained Word2Vec) ────┘
                    │                              │
         ┌──────────┼──────────┐       ┌──────────┼──────────┐
    Conv1D(256)  Conv1D(256)  Conv1D(256)  (same 3 branches for job)
    kernel=2     kernel=3     kernel=5
    SpatialDropout1D(0.3) + L2(1e-4)
         └──────────┼──────────┘
              GlobalMaxPool1D × 3
              Concat → 192-dim each branch
              Total CV repr: 576-dim
                    │
         ┌──────────┴──────────┐
      CV repr (576)      Job repr (576)
         └── CustomAttentionLayer (cross-attention) ──┘
                            │
                     Dense(256) → Dense(128) → Dense(64)
                            │
                     Linear → matching_score (0.0–1.0)
```

**Loss**: Huber Loss (δ=0.1) · **LR**: Cosine Annealing (η_max=1.374e-3, T_max=80)  
**OPTIMAL_THRESHOLD**: 0.44 · **F1-Score**: 0.9036 · **Accuracy**: 88.65%

**HybridScorer** (inference):
```
final_score = α × neural_score + (1 − α) × structured_score
```
- α = 0.40 jika structured_score tersedia, 0.15 jika tidak

---

### Model v3 — Deployed di Cloud Run *(Saat ini aktif)*

Arsitektur sama dengan v2 (Dual-Input CNN + Custom Attention), dengan perbaikan **data synthesis**:
- 5-mode pair synthesizer (vs 2-mode v2)
- Soft continuous labels (regression, bukan binary)
- Huber regression loss
- Hard negative pairs: cross-domain, same-domain diff-role, seniority mismatch

**Metrics v3** (regression):  
Val MAE: 0.144 · Pseudo-Accuracy @0.5: 81.1%

---

## 📁 Struktur Proyek

```
SkillAlign-AI-Service/
├── src/
│   ├── database/
│   │   └── supabase_client.py          # Singleton Supabase client + cache helpers
│   ├── inference/
│   │   ├── predict.py                  # SkillAlignPredictor + HybridScorer
│   │   ├── api_service.py              # FastAPI router (predict, skill-gap, analyze-cv, recommend)
│   │   ├── skill_gap.py                # SkillGapAnalyzer (SkillNer EMSI database)
│   │   ├── learning_path_router.py     # Learning path endpoints (Gemini + YouTube)
│   │   ├── course_finder.py            # CourseFinder via Gemini Search Grounding
│   │   ├── cv_profile_extractor.py     # Ekstraksi profil CV (role, exp, education)
│   │   ├── job_title_suggester.py      # Saran job title dari skill CV
│   │   ├── industry_skill_analyzer.py  # Analisis kesiapan skill vs industri
│   │   ├── hybrid_scorer.py            # HybridScorer (neural + structured)
│   │   └── role_config.py              # Konfigurasi role & skill mapping
│   ├── models/
│   │   ├── model_architecture.py       # SkillAlignMatcherV4 (Multi-Scale CNN)
│   │   ├── custom_layers.py            # CustomAttentionLayer
│   │   ├── custom_loss.py              # Huber + auxiliary losses
│   │   └── custom_callbacks.py         # F1-sweep callback, LR scheduler
│   ├── preprocessing/
│   │   ├── nlp_preprocessor.py         # Tokenizer + Lemmatizer
│   │   ├── feature_engineering.py      # TF-IDF, structured features
│   │   ├── embeddings.py               # Word2Vec manager
│   │   └── pair_synthesizer.py         # 5-mode data synthesis
│   ├── training/
│   │   ├── train.py                    # Training pipeline
│   │   ├── custom_training_loop.py     # tf.GradientTape demo
│   │   └── hyperparameter_tuning.py    # Keras Tuner
│   └── utils/
│       ├── metrics.py
│       ├── error_handling.py
│       ├── validation.py
│       └── visualization.py
├── models/
│   ├── skillalign_matcher_v4.keras     # Model v4 — latest (local & Cloud Run next deploy)
│   ├── model_config_v4.json
│   ├── skillalign_matcher_v3.keras     # Model v3 — saat ini deployed di Cloud Run
│   └── model_config_v3.json
├── preprocessors/
│   ├── nlp_preprocessor_v4.pkl
│   ├── embedding_manager_v4.pkl
│   ├── nlp_preprocessor_v3.pkl
│   └── embedding_manager_v3.pkl
├── notebooks/
│   ├── 02U_model_development.ipynb     # Training pipeline v4 (Google Colab)
│   └── plots_v4/                       # Visualisasi training v4 (CM, pred_vs_actual, dll)
├── scripts/
│   └── supabase_migrations.sql         # DDL untuk tabel skill_courses & learning_path_sessions
├── logs/
│   └── training_v4/                    # TensorBoard event files
├── Dockerfile
├── .dockerignore
├── .env                                # Local env vars (di-gitignore)
├── .env.example                        # Template env vars
├── main.py                             # FastAPI entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Instalasi

> **AI Engineer / Data Scientist**: ikuti semua langkah (1–5).  
> **Software Engineer / Backend**: ikuti langkah 1, 2, 4, 5 saja (skip langkah 3).

### 1. Virtual Environment

```bash
cd SkillAlign-AI-Service

# Buat virtual environment BARU (jangan copy dari project lain)
python -m venv venv

# Aktifkan (Windows)
.\venv\Scripts\activate

# Aktifkan (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Windows — gunakan python -m pip, BUKAN pip langsung
# (menghindari masalah launcher .exe jika venv pernah dipindah/dicopy)
python -m pip install -r requirements.txt

# Download spaCy model (~400MB, wajib untuk /skill-gap dan /extract-cv-skills)
python -m spacy download en_core_web_lg
```

### 3. Setup Environment Variables

Buat file `.env` di root project (lihat `.env.example` sebagai template):

```env
# Model (v3 = deployed, v4 = latest dari notebook)
MODEL_PATH=models/skillalign_matcher_v3.keras
PREPROCESSOR_PATH=preprocessors/nlp_preprocessor_v3.pkl
CONFIG_PATH=models/model_config_v3.json

# Gemini API (Google AI Studio — FREE tier cukup)
GEMINI_API_KEY=your_gemini_api_key_here

# YouTube Data API v3
YOUTUBE_API_KEY=your_youtube_api_key_here

# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here
```

> ⚠️ **Jangan commit `.env` ke git.** File ini sudah di-gitignore.  
> ⚠️ **GEMINI_API_KEY**: Gunakan key dari **Google AI Studio** (bukan GCP Vertex AI) agar tetap di FREE tier. Gemini 2.5 Flash **thinking tokens** dikenakan biaya — pastikan project GCP tidak terhubung ke billing account untuk penggunaan capstone.

### 4. Setup Supabase Tables *(wajib untuk Learning Path)*

Jalankan SQL berikut di **Supabase Dashboard → SQL Editor**:

```sql
-- Lihat file lengkap di: scripts/supabase_migrations.sql
CREATE TABLE IF NOT EXISTS skill_courses ( ... );
CREATE TABLE IF NOT EXISTS learning_path_sessions ( ... );
```

File lengkap ada di `scripts/supabase_migrations.sql`.

### 5. Jalankan API Server

```bash
# Windows — WAJIB gunakan python -m uvicorn, bukan uvicorn langsung
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Linux/Mac — bisa pakai uvicorn langsung
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server berhasil jika muncul log:
```
INFO - ✅ Model v4 loaded | threshold=0.44 | hybrid=ON
INFO - Application startup complete.
```

> **Catatan**: Jika model belum ada, server tetap berjalan tapi `/predict` return HTTP 503.

### 6. Akses Swagger UI

Buka browser: **http://localhost:8000/docs**

---

## 🐳 Docker & Deployment

### Build & Run Lokal

```bash
docker build -t skillalign-ai .
docker run -p 8000:8000 --env-file .env skillalign-ai
```

### Deploy ke Cloud Run (via Google Colab / gcloud CLI)

```bash
IMAGE="asia-southeast1-docker.pkg.dev/skillalign-496406/skillalign-repo/skillalign-ai:v3"

# Build + push ke Artifact Registry (1 command)
gcloud builds submit --tag $IMAGE .

# Deploy ke Cloud Run
gcloud run deploy skillaign-ai \
  --image $IMAGE \
  --region asia-southeast1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 2 \
  --set-env-vars "MODEL_PATH=models/skillalign_matcher_v3.keras" \
  --set-env-vars "PREPROCESSOR_PATH=preprocessors/nlp_preprocessor_v3.pkl" \
  --set-env-vars "CONFIG_PATH=models/model_config_v3.json" \
  --set-env-vars "OPTIMAL_THRESHOLD=0.44" \
  --set-env-vars "USE_HYBRID=true" \
  --set-env-vars "GEMINI_API_KEY=..." \
  --set-env-vars "YOUTUBE_API_KEY=..." \
  --set-env-vars "SUPABASE_URL=..." \
  --set-env-vars "SUPABASE_SERVICE_ROLE_KEY=..." \
  --allow-unauthenticated
```

> **Catatan Cloud Run**: `$PORT` di-inject otomatis (8080). Dockerfile sudah menggunakan `${PORT:-8000}` sehingga tidak perlu set manual.

---

## 🧪 Testing API

### Cara 1 — Swagger UI

Buka **http://localhost:8000/docs** → klik endpoint → "Try it out" → isi body → "Execute".

---

### Cara 2 — curl

**Health check:**
```bash
curl http://localhost:8000/health
```
```json
{ "status": "healthy", "model_loaded": true, "model_version": "v4", "optimal_threshold": 0.44 }
```

**Single prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cv_text": "Experienced Data Scientist with 5 years in Python TensorFlow machine learning deep learning. Deployed 10+ production models.",
    "job_description": "Looking for a Data Scientist with Python skills, ML frameworks, and data analysis experience."
  }'
```
```json
{
  "matching_score": 0.78,
  "confidence": "High",
  "recommendation": "Highly Recommended"
}
```

**Batch prediction:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "cv_text": "Data Scientist with 5 years in Python TensorFlow machine learning.",
    "job_descriptions": [
      "Data Scientist role requiring Python and ML skills.",
      "Marketing Manager for digital campaigns and SEO.",
      "Frontend Developer with React and JavaScript."
    ]
  }'
```
```json
{
  "results": [
    { "rank": 1, "job_index": 0, "matching_score": 0.78, "confidence": "High", "recommendation": "Highly Recommended", "inference_time_ms": 51.2 },
    { "rank": 2, "job_index": 2, "matching_score": 0.31, "confidence": "Low",  "recommendation": "Not Recommended",    "inference_time_ms": 48.7 },
    { "rank": 3, "job_index": 1, "matching_score": 0.25, "confidence": "Low",  "recommendation": "Not Recommended",    "inference_time_ms": 49.1 }
  ],
  "total_items": 3,
  "total_time_ms": 312.4
}
```

> **Cara baca `job_index`**: posisi job di array input (0-based). Backend gunakan ini untuk lookup data lengkap job dari database.

**Skill gap analysis:**
```bash
curl -X POST http://localhost:8000/api/v1/skill-gap \
  -H "Content-Type: application/json" \
  -d '{
    "cv_text": "Data Analyst with 3 years experience. Skilled in SQL, Excel, and PowerBI. Basic Python knowledge.",
    "job_description": "Data Scientist position requiring Python, machine learning, TensorFlow, and statistical modeling."
  }'
```
```json
{
  "skill_gap_score": 0.25,
  "skill_coverage_percent": "25%",
  "top_priority_skill": "machine learn",
  "present_skills": [
    { "skill": "python", "skill_id": "KS125LS6N7WP4S6SFTCK", "match_score": 1.0, "priority": 0 },
    { "skill": "sql",    "skill_id": "KS440W865GC4VRBW6LJP", "match_score": 1.0, "priority": 0 }
  ],
  "missing_skills": [
    { "skill": "machine learn", "skill_id": "KS1261Z68KSKR1X31KS3", "match_score": 0.0, "priority": 1 },
    { "skill": "tensorflow",    "skill_id": "KS120B874P2P6BSVTU0F", "match_score": 0.0, "priority": 2 }
  ],
  "recommendation_summary": "Kesesuaian skill: 25% (perlu peningkatan). Prioritaskan mempelajari: machine learn, tensorflow.",
  "analysis_time_ms": 6500.0
}
```

> ⏱️ **Request pertama ~5–15 detik** (SkillNer loading model `en_core_web_lg` ~400MB ke memory). Request berikutnya ~500ms.

---

## 📡 API Endpoints

| Method | Endpoint | Deskripsi | Butuh Model? |
|---|---|---|---|
| GET | `/` | Service info & status | — |
| GET | `/health` | Health check | — |
| POST | `/predict` | Single CV vs 1 Job | ✅ |
| POST | `/api/v1/predict` | Single CV vs 1 Job (versioned) | ✅ |
| POST | `/api/v1/predict/batch` | 1 CV vs ≤50 Jobs, diranking | ✅ |
| POST | `/api/v1/skill-gap` | Analisis skill gap CV vs Job (SkillNer) | ❌ |
| POST | `/api/v1/extract-cv-skills` | Ekstrak skill dari CV saja | ❌ |
| POST | `/api/v1/analyze-cv` | Profil CV + saran job title | ❌ |
| POST | `/api/v1/recommend` | Ranking job + industry skill analysis | ✅ |
| POST | `/api/v1/learning-path/analyze` | Rencana belajar per skill (Gemini + YouTube) | ❌ |
| POST | `/api/v1/learning-path/refresh` | Refresh cache kursus untuk skill tertentu | ❌ |
| GET | `/api/v1/learning-path/courses/{skill}` | Ambil kursus dari cache Supabase | ❌ |

---

## 📊 Request / Response Schema

### Single & Batch Predict

**Request (single):**
```json
{
  "cv_text": "string (min 50, max 10.000 char)",
  "job_description": "string (min 30, max 10.000 char)",
  "user_id": "string (opsional)"
}
```

**Request (batch):**
```json
{
  "cv_text": "string",
  "job_descriptions": ["string", "..."],
  "user_id": "string (opsional)"
}
```
> Maksimum 50 job descriptions per request.

### Skill Gap

**Request:**
```json
{
  "cv_text": "string (min 50 char)",
  "job_description": "string (min 30 char)"
}
```

**Response fields:**

| Field | Tipe | Keterangan |
|---|---|---|
| `skill_gap_score` | float 0–1 | Skor kesesuaian skill |
| `skill_coverage_percent` | string | Persentase skill requirement yang terpenuhi |
| `top_priority_skill` | string | Skill paling penting untuk dipelajari |
| `present_skills[]` | array | Skill yang ada di CV sesuai job requirement |
| `missing_skills[]` | array | Skill yang kurang, urut by prioritas |
| `present_skills[].skill_id` | string | EMSI canonical skill ID |
| `present_skills[].match_score` | float | Confidence SkillNer (1.0 = exact match) |
| `recommendation_summary` | string | Ringkasan rekomendasi bahasa natural |
| `analysis_time_ms` | float | Waktu analisis (ms) |

### Confidence & Recommendation Mapping

| Score | Confidence | Recommendation |
|---|---|---|
| ≥ 0.70 | High | Highly Recommended |
| 0.44 – 0.69 | Medium | Consider |
| < 0.44 | Low | Not Recommended |

> **Threshold = 0.44** dikalibrasi via F1-sweep pada validation set (v4).

---

## 🔧 Environment Variables

| Variable | Wajib | Deskripsi |
|---|---|---|
| `MODEL_PATH` | ✅ | Path ke file `.keras` (default: `models/skillalign_matcher_v4.keras`) |
| `PREPROCESSOR_PATH` | ✅ | Path ke file `.pkl` preprocessor |
| `CONFIG_PATH` | — | Path ke `model_config_*.json` |
| `OPTIMAL_THRESHOLD` | — | Threshold klasifikasi (default: 0.44) |
| `USE_HYBRID` | — | Aktifkan HybridScorer (default: true) |
| `GEMINI_API_KEY` | ✅* | Wajib untuk Learning Path endpoint |
| `YOUTUBE_API_KEY` | ✅* | Wajib untuk Learning Path (YouTube resources) |
| `SUPABASE_URL` | ✅* | Wajib untuk Learning Path caching |
| `SUPABASE_SERVICE_ROLE_KEY` | ✅* | Wajib untuk Learning Path caching |

> *Opsional jika tidak menggunakan endpoint Learning Path.

---

## 📊 TensorBoard

```bash
tensorboard --logdir=logs/training_v4
```
Buka browser: **http://localhost:6006**

---

## 📈 Performance Model

### v4 — SkillAlignMatcherV4 *(Latest)*

| Metric | Nilai |
|---|---|
| Accuracy (threshold=0.44) | **88.65%** |
| F1-Score (threshold=0.44) | **0.9036** |
| Precision | 0.8792 |
| Recall | 0.9293 |
| MAE (regression) | 0.1105 |
| RMSE | 0.1783 |
| Correlation | 0.772 |
| Best Val MAE (epoch 70/80) | 0.10766 |
| Optimal Threshold | **0.44** |
| Inference Time (predict) | ~50ms |
| Inference Time (skill-gap, cold) | ~5–15s |
| Inference Time (skill-gap, warm) | ~500ms |

### v3 — Deployed *(Cloud Run)*

| Metric | Nilai |
|---|---|
| Val MAE | 0.144 |
| Pseudo-Accuracy @threshold 0.5 | 81.1% |
| Epochs | 32 |

> **Catatan MAE**: Model v3 dan v4 menggunakan **regression** (continuous label 0.0–1.0, Huber Loss), bukan binary classification. MAE 0.144 berarti rata-rata prediksi meleset ~0.14 dari label. Untuk keputusan binary (cocok/tidak), gunakan threshold 0.44 yang dikalibrasi dari F1-sweep.

---

## ⚠️ Known Limitations

| Aspek | Keterbatasan |
|---|---|
| **Geografis** | Dataset LinkedIn US — kurang akurat untuk konteks Indonesia |
| **Industri** | Dominasi IT, Healthcare, Finance — logistik/manufaktur kurang terwakili |
| **Bahasa** | Input Bahasa Inggris saja — input Bahasa Indonesia → banyak OOV token |
| **Skill NER** | SkillNer terkadang truncate nama skill multi-kata (misal: "machine learning" → "machine learn") |
| **Cold Start** | SkillNer butuh ~5–15s loading di request pertama |
| **Gemini Billing** | Gemini 2.5 Flash thinking tokens dikenakan biaya jika GCP project terhubung billing |

---

## 👥 Tim

- **Zahri Ramadhani** — AI Engineer
- **Destian Aldi Nugraha** — AI Engineer

## 📄 Lisensi

Capstone Project — DBS Foundation Coding Camp 2026
