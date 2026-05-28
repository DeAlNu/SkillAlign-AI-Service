# SkillAlign AI Service

<div align="center">

**CV-Job Matching Engine berbasis Deep Learning & NLP**

*Capstone Project — DBS Foundation Coding Camp 2026 · Tim CC26-PSU318*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-3.13-D00000?logo=keras&logoColor=white)](https://keras.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Cloud Run](https://img.shields.io/badge/Cloud%20Run-Deployed-4285F4?logo=google-cloud&logoColor=white)](https://cloud.google.com/run)
[![Supabase](https://img.shields.io/badge/Supabase-Cache-3ECF8E?logo=supabase&logoColor=white)](https://supabase.com)

[![F1 Score](https://img.shields.io/badge/F1--Score-0.9036-brightgreen)](notebooks/plots_v4/threshold_calibration.png)
[![Accuracy](https://img.shields.io/badge/Accuracy-88.65%25-brightgreen)](notebooks/plots_v4/cm_v4.png)
[![Inference](https://img.shields.io/badge/Inference-~50ms-blue)](README.md)

</div>

---

---

## Daftar Isi

- [Overview](#-overview)
- [Live Service](#-live-service)
- [Arsitektur Model](#-arsitektur-model)
- [Arsitektur Sistem](#-arsitektur-sistem)
- [Struktur Proyek](#-struktur-proyek)
- [Setup & Instalasi](#-setup--instalasi)
- [Environment Variables](#-environment-variables)
- [Menjalankan Service](#-menjalankan-service)
- [Docker & Deployment](#-docker--deployment)
- [API Reference](#-api-reference)
- [Contoh Request & Response](#-contoh-request--response)
- [Performance Metrics](#-performance-metrics)
  - [Training Curves](#training-curves)
  - [Threshold Calibration](#threshold-calibration)
  - [Score Distribution](#score-distribution)
  - [Confusion Matrix](#confusion-matrix--prediction-quality)
- [Training Pipeline](#-training-pipeline)
- [Supabase Setup](#-supabase-setup)
- [Known Limitations](#-known-limitations)
- [Tim](#-tim)

---

## 📋 Overview

SkillAlign AI Service adalah **scoring engine** berbasis Deep Learning untuk **CV-Job Matching**. Service ini menerima teks CV dan teks job description, lalu menghasilkan skor kecocokan 0.0–1.0 beserta analisis skill gap, profil kandidat, dan rekomendasi learning path.

Service berperan sebagai **AI backend** yang dikonsumsi oleh Backend (Express.js) dalam alur **Two-Stage Retrieval + Re-Ranking**:

```
User Request → Backend (Express.js)
                    │
          ┌─────────┴────────────┐
          │  Stage 1: Retrieve   │  ← PostgreSQL full-text / vector search
          │  (top-K candidates)  │
          └─────────┬────────────┘
                    │
          ┌─────────┴────────────┐
          │  Stage 2: Re-Rank    │  ← SkillAlign AI Service (this repo)
          │  (neural scoring)    │
          └─────────┬────────────┘
                    │
              Final Ranked Results
```

### Fitur Utama

| Fitur | Deskripsi | Endpoint |
|---|---|---|
| **CV-Job Matching** | Skor kecocokan 0.0–1.0 (neural + structured) | `/predict`, `/api/v1/predict` |
| **Batch Scoring** | 1 CV vs hingga 50 job sekaligus, diranking | `/api/v1/predict/batch` |
| **Skill Gap Analysis** | Deteksi skill yang ada vs yang kurang dari job | `/api/v1/skill-gap` |
| **Extract CV Skills** | Ekstrak daftar skill dari teks CV saja | `/api/v1/extract-cv-skills` |
| **Analyze CV** | Profil kandidat (role, pengalaman, pendidikan) + saran job title | `/api/v1/analyze-cv` |
| **Recommend Jobs** | Ranking job postings + analisis kesiapan skill per industri | `/api/v1/recommend` |
| **Learning Path** | Rencana belajar per skill (Gemini 2.5 Flash + YouTube + cache Supabase) | `/api/v1/learning-path/...` |

### Tech Stack

| Layer | Teknologi |
|---|---|
| **API Framework** | FastAPI 0.104 + Uvicorn |
| **ML Framework** | TensorFlow 2.15 / Keras 3.13 |
| **NLP** | NLTK, Gensim Word2Vec, KeyBERT, SentenceTransformers |
| **Skill Extraction** | CsvSkillExtractor (rule-based, ~685 job types, ~3000+ skills) |
| **AI / LLM** | Gemini 2.5 Flash via REST (Search Grounding) |
| **Video API** | YouTube Data API v3 |
| **Cache / DB** | Supabase (PostgreSQL) |
| **Containerization** | Docker, Google Cloud Build |
| **Hosting** | Google Cloud Run (asia-southeast2 / Jakarta) |

---

## 🌐 Live Service

Service di-deploy di **Google Cloud Run** oleh masing-masing anggota tim. Setiap deployment berjalan secara mandiri dengan konfigurasi env vars sendiri.

| Anggota | Region | Health Check |
|---|---|---|
| Destian | asia-southeast2 (Jakarta) | `GET <URL_DESTIAN>/health` |
| Zahri | asia-southeast1 (Singapore) | `GET <URL_ZAHRI>/health` |

> URL production masing-masing deployment bisa dilihat di Google Cloud Console → Cloud Run, atau via `gcloud run services describe skillalign-ai --region <region>`.

Health check response:
```json
{ "status": "healthy", "model_loaded": true, "model_version": "v4", "optimal_threshold": 0.44 }
```

---

## 🏗️ Arsitektur Model

### Model v4 — SkillAlignMatcherV4 (Aktif di Production)

Model utama menggunakan arsitektur **Multi-Scale CNN + Cross-Attention**:

```
cv_input (seq_len=300)                   job_input (seq_len=300)
        │                                         │
        └──── Shared Embedding Layer ─────────────┘
              (vocab=15.000, dim=128, pre-trained Word2Vec)
                       │                       │
          ┌────────────┼────────────┐           │  (sama untuk job)
     Conv1D(256)  Conv1D(256)  Conv1D(256)      │
     kernel=2     kernel=3     kernel=5         │
     SpatialDropout1D(0.3)                      │
     L2 regularization (1e-4)                   │
          └────────────┼────────────┘           │
            GlobalMaxPool1D × 3 branches        │
            Concat → 768-dim repr              768-dim repr
                       │                         │
                       └──── CrossAttentionLayer ─┘
                                    │
                             Dense(256, relu)
                             Dropout(0.4)
                             Dense(128, relu)
                             Dropout(0.3)
                             Dense(64, relu)
                             Dense(1, sigmoid)
                                    │
                           matching_score (0.0–1.0)
```

**Hyperparameter & Training:**

| Parameter | Nilai |
|---|---|
| Vocab size | 15.000 token |
| Embedding dim | 128 (pre-trained Word2Vec) |
| Sequence length | 300 token |
| CNN kernels | k=2, k=3, k=5 (256 filter each) |
| Loss function | Huber Loss (δ=0.1) |
| Learning rate | Cosine Annealing (η_max=1.374e-3, T_max=80) |
| Epochs | 80 |
| Batch size | 64 |
| Optimizer | Adam + weight decay |
| Optimal threshold | **0.44** (dikalibrasi via F1-sweep) |

### HybridScorer

Inference menggabungkan neural score dengan structured features:

```
final_score = α × neural_score + (1 − α) × structured_score

α = 0.40  →  jika structured_score tersedia (CV dan Job punya data terstruktur)
α = 0.15  →  jika hanya neural_score (teks bebas tanpa structured data)
```

Structured features mencakup: TF-IDF cosine similarity, skill keyword overlap, pendidikan matching, dan pengalaman level matching.

---

## 🔧 Arsitektur Sistem

```
                         ┌─────────────────────────────────────────┐
                         │          SkillAlign AI Service           │
                         │              (FastAPI)                   │
                         │                                          │
  HTTP Request  ────────►│  /predict          → SkillAlignPredictor │
                         │  /api/v1/predict        + HybridScorer   │
                         │  /api/v1/predict/batch                   │
                         │                                          │
                         │  /api/v1/skill-gap  → CsvSkillExtractor  │
                         │  /api/v1/extract-cv-skills  (rule-based) │
                         │                                          │
                         │  /api/v1/analyze-cv → CvProfileExtractor │
                         │  /api/v1/recommend  → IndustryAnalyzer   │
                         │                                          │
                         │  /api/v1/learning-path/* → CourseFinder  │
                         │                            (Gemini REST)  │
                         └───────────┬──────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
               ┌────▼────┐    ┌──────▼──────┐  ┌─────▼──────┐
               │ TF/Keras │    │  Gemini 2.5  │  │  Supabase  │
               │ Model v4 │    │  Flash REST  │  │  (Cache)   │
               └──────────┘    └─────────────┘  └────────────┘
```

### Skill Extraction (v7 — CSV Rule-Based)

`CsvSkillExtractor` menggantikan SkillNer sepenuhnya:

```
Input text
    │
    ▼
Load job_skill_map.csv (~685 job types)
    │  ← skills_core | skills_common | skills_optional | soft_skills
    ▼
Deduplicate semua skill → sorted by length DESC (longer match first)
    │
    ▼
Pre-compile regex patterns (custom word boundary: handle C#, .NET, C++)
    │
    ▼
Scan text → greedy match → overlap prevention
    │
    ▼
Dict[skill_id → (skill_name, confidence=1.0)]
```

Keunggulan vs SkillNer:
- Tidak ada dependency spaCy (~800MB lebih ringan)
- Startup instan, sub-65ms per request
- Tidak ada cold start lag
- Tidak ada risiko segfault dari Cython concurrency

---

## 📁 Struktur Proyek

```
SkillAlign-AI-Service/
├── src/
│   ├── database/
│   │   ├── supabase_client.py          # Singleton Supabase client + cache helpers
│   │   └── job_skill_map.csv           # ~685 job types, ~3000+ skills (4 columns)
│   ├── inference/
│   │   ├── predict.py                  # SkillAlignPredictor + HybridScorer
│   │   ├── api_service.py              # FastAPI router: predict, skill-gap, analyze-cv, recommend
│   │   ├── skill_gap.py                # SkillGapAnalyzer v7 (wraps CsvSkillExtractor)
│   │   ├── csv_skill_extractor.py      # CsvSkillExtractor (rule-based, regex)
│   │   ├── learning_path_router.py     # Learning path endpoints (Gemini + YouTube)
│   │   ├── course_finder.py            # CourseFinder via Gemini 2.5 Flash REST API
│   │   ├── cv_profile_extractor.py     # Ekstraksi profil CV (role, exp, education)
│   │   ├── job_title_suggester.py      # Saran job title dari skill CV
│   │   ├── industry_skill_analyzer.py  # Analisis kesiapan skill vs industri
│   │   ├── hybrid_scorer.py            # HybridScorer (neural + structured)
│   │   └── role_config.py              # Konfigurasi role & skill mapping
│   ├── models/
│   │   ├── model_architecture.py       # SkillAlignMatcherV4 (Multi-Scale CNN + CrossAttention)
│   │   ├── custom_layers.py            # CrossAttentionLayer
│   │   ├── custom_loss.py              # Huber + auxiliary losses
│   │   └── custom_callbacks.py         # F1-sweep callback, LR scheduler
│   ├── preprocessing/
│   │   ├── nlp_preprocessor.py         # Tokenizer + Lemmatizer (NLTK-based)
│   │   ├── feature_engineering.py      # TF-IDF, structured features
│   │   ├── embeddings.py               # Word2Vec embedding manager (Gensim)
│   │   └── pair_synthesizer.py         # 5-mode data synthesis untuk training
│   ├── training/
│   │   ├── train.py                    # Training pipeline (main entry)
│   │   ├── custom_training_loop.py     # tf.GradientTape training loop
│   │   └── hyperparameter_tuning.py    # Keras Tuner integration
│   └── utils/
│       ├── metrics.py                  # Custom metrics (F1-sweep)
│       ├── error_handling.py           # Exception handlers
│       ├── validation.py               # Input validation helpers
│       └── visualization.py            # Confusion matrix, training curves
├── models/
│   ├── skillalign_matcher_v4.keras     # Model v4 — aktif di production
│   └── model_config_v4.json            # Konfigurasi model v4
├── preprocessors/
│   ├── nlp_preprocessor_v4.pkl         # Tokenizer + vocab v4
│   └── embedding_manager_v4.pkl        # Word2Vec weights v4
├── notebooks/
│   ├── 02U_model_development.ipynb     # Training pipeline v4 (Google Colab)
│   └── plots_v4/                       # Visualisasi: CM, pred_vs_actual, calibration
├── scripts/
│   ├── supabase_migrations.sql         # DDL tabel skill_courses & learning_path_sessions
│   └── train_v4.py                     # Script training v4 (CLI)
├── logs/
│   └── training_v4/                    # TensorBoard event files
├── Dockerfile                          # Multi-stage build, ~2.5GB final image
├── .gcloudignore                       # Reduce build context: ~60MB (dari 573MB)
├── .env.example                        # Template env vars
├── main.py                             # FastAPI entry point + lifespan
└── requirements.txt                    # Python dependencies
```

---

## ⚙️ Setup & Instalasi

> **AI Engineer / Data Scientist**: ikuti semua langkah.  
> **Software Engineer / Backend**: ikuti langkah 1–4 saja (skip training di langkah 5).

### 1. Prasyarat

- Python 3.11 (direkomendasikan; 3.10 bisa tapi belum diuji)
- Git
- (Opsional) Docker jika ingin run via container
- (Opsional) `gcloud` CLI jika ingin deploy ke Cloud Run

### 2. Clone & Virtual Environment

```bash
git clone https://github.com/<org>/SkillAlign-AI-Service.git
cd SkillAlign-AI-Service

# Buat virtual environment baru
python -m venv venv

# Aktifkan (Windows)
.\venv\Scripts\activate

# Aktifkan (Linux/Mac)
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Windows — gunakan python -m pip untuk menghindari masalah PATH
python -m pip install -r requirements.txt

# Linux/Mac
pip install -r requirements.txt
```

> **Catatan**: Tidak perlu download model spaCy. `CsvSkillExtractor` (digunakan sejak v7) adalah pure-Python, tidak bergantung spaCy/SkillNer.

NLTK data didownload otomatis saat pertama kali menjalankan service. Jika ingin pre-download manual:

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### 4. Setup Environment Variables

Salin file template dan isi dengan nilai yang sesuai:

```bash
cp .env.example .env
```

Edit file `.env`:

```env
# ── Model ───────────────────────────────────────────────────────────────────
MODEL_PATH=models/skillalign_matcher_v4.keras
PREPROCESSOR_PATH=preprocessors/nlp_preprocessor_v4.pkl
CONFIG_PATH=models/model_config_v4.json
OPTIMAL_THRESHOLD=0.44
USE_HYBRID=true

# ── Gemini API (Google AI Studio — https://aistudio.google.com) ─────────────
# Wajib untuk /api/v1/learning-path/*
GEMINI_API_KEY=your_gemini_api_key_here

# ── YouTube Data API v3 (Google Cloud Console) ───────────────────────────────
# Wajib untuk /api/v1/learning-path/* (opsional jika tidak butuh video)
YOUTUBE_API_KEY=your_youtube_api_key_here

# ── Supabase ─────────────────────────────────────────────────────────────────
# Wajib untuk caching learning path
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here
```

> ⚠️ **Jangan commit `.env`** — sudah di-gitignore.  
> ⚠️ **GEMINI_API_KEY**: Gunakan key dari **Google AI Studio** (bukan GCP Vertex AI) agar tetap di FREE tier.

---

## 🔧 Environment Variables

| Variable | Wajib | Default | Deskripsi |
|---|---|---|---|
| `MODEL_PATH` | ✅ | — | Path ke file `.keras` model v4 |
| `PREPROCESSOR_PATH` | ✅ | — | Path ke file `.pkl` NLP preprocessor |
| `CONFIG_PATH` | — | — | Path ke `model_config_v4.json` |
| `OPTIMAL_THRESHOLD` | — | `0.44` | Threshold klasifikasi binary (matching/not) |
| `USE_HYBRID` | — | `true` | Aktifkan HybridScorer (neural + structured) |
| `GEMINI_API_KEY` | ✅* | — | API key Gemini 2.5 Flash (Google AI Studio) |
| `YOUTUBE_API_KEY` | ✅* | — | YouTube Data API v3 key |
| `SUPABASE_URL` | ✅* | — | URL project Supabase |
| `SUPABASE_SERVICE_ROLE_KEY` | ✅* | — | Service role key Supabase (bukan anon key) |

> *Opsional jika tidak menggunakan endpoint `/api/v1/learning-path/*`.

---

## 🚀 Menjalankan Service

### Development (dengan hot-reload)

```bash
# Windows
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Linux/Mac
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server siap jika log menampilkan:

```
INFO:     ✅ Model v4 loaded | threshold=0.44 | hybrid=ON
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Swagger UI

Buka browser: **http://localhost:8000/docs**

### TensorBoard (monitoring training)

```bash
tensorboard --logdir=logs/training_v4
# Buka: http://localhost:6006
```

---

## 🐳 Docker & Deployment

### Build & Run Lokal

```bash
# Build image
docker build -t skillalign-ai .

# Run dengan env vars
docker run -p 8000:8000 \
  --env-file .env \
  skillalign-ai

# Atau pass env vars manual
docker run -p 8000:8000 \
  -e MODEL_PATH=models/skillalign_matcher_v4.keras \
  -e PREPROCESSOR_PATH=preprocessors/nlp_preprocessor_v4.pkl \
  -e GEMINI_API_KEY=your_key \
  skillalign-ai
```

Health check setelah container running:

```bash
curl http://localhost:8000/health
```

### Deploy ke Google Cloud Run

Pastikan sudah login ke gcloud dan project di-set:

```bash
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT_ID
```

#### Step 1 — Build image via Cloud Build

Ganti variabel sesuai konfigurasi GCP masing-masing:

```bash
PROJECT_ID="your-gcp-project-id"
REGION="asia-southeast1"   # sesuaikan: asia-southeast1, asia-southeast2, dst.
REPO="your-artifact-registry-repo"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/skillalign-ai:latest"

# Submit build ke Cloud Build (~15-20 menit build pertama, ~8-10 menit berikutnya)
gcloud builds submit \
  --tag $IMAGE \
  --timeout=30m \
  --project=$PROJECT_ID \
  .
```

> **Tip**: File `.gcloudignore` mengecualikan dataset, notebook, old model versions, dan file dev lainnya — mereduksi upload dari ~573MB menjadi ~60MB.

#### Step 2 — Deploy revision baru

```bash
gcloud run deploy skillalign-ai \
  --image $IMAGE \
  --region $REGION \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 3 \
  --set-env-vars "MODEL_PATH=models/skillalign_matcher_v4.keras" \
  --set-env-vars "PREPROCESSOR_PATH=preprocessors/nlp_preprocessor_v4.pkl" \
  --set-env-vars "CONFIG_PATH=models/model_config_v4.json" \
  --set-env-vars "OPTIMAL_THRESHOLD=0.44" \
  --set-env-vars "USE_HYBRID=true" \
  --set-env-vars "GEMINI_API_KEY=YOUR_GEMINI_KEY" \
  --set-env-vars "YOUTUBE_API_KEY=YOUR_YOUTUBE_KEY" \
  --set-env-vars "SUPABASE_URL=YOUR_SUPABASE_URL" \
  --set-env-vars "SUPABASE_SERVICE_ROLE_KEY=YOUR_SUPABASE_KEY" \
  --allow-unauthenticated \
  --project $PROJECT_ID
```

> `$PORT` di-inject otomatis oleh Cloud Run (8080). Dockerfile sudah menggunakan `${PORT:-8000}` sehingga tidak perlu set manual.

#### Cek deployment

```bash
gcloud run services describe skillalign-ai \
  --region $REGION \
  --project $PROJECT_ID \
  --format "value(status.url)"
```

---

## 📡 API Reference

### Ringkasan Endpoint

| Method | Endpoint | Deskripsi | Butuh Model? |
|---|---|---|---|
| GET | `/` | Service info, versi, status model | — |
| GET | `/health` | Health check | — |
| POST | `/predict` | Single CV vs 1 Job (shorthand) | ✅ |
| POST | `/api/v1/predict` | Single CV vs 1 Job (versioned) | ✅ |
| POST | `/api/v1/predict/batch` | 1 CV vs ≤50 Jobs, diranking | ✅ |
| POST | `/api/v1/skill-gap` | Analisis skill gap CV vs Job | ❌ |
| POST | `/api/v1/extract-cv-skills` | Ekstrak skill dari teks CV saja | ❌ |
| POST | `/api/v1/analyze-cv` | Profil kandidat + saran job title | ❌ |
| POST | `/api/v1/recommend` | Ranking jobs + industry skill analysis | ✅ |
| POST | `/api/v1/learning-path/analyze` | Rencana belajar per skill (Gemini + YouTube) | ❌ |
| POST | `/api/v1/learning-path/refresh` | Refresh cache kursus untuk skill tertentu | ❌ |
| GET | `/api/v1/learning-path/courses/{skill}` | Ambil kursus dari cache Supabase | ❌ |

---

### Schema Request / Response

#### `POST /predict` dan `POST /api/v1/predict`

**Request:**
```json
{
  "cv_text": "string (min 50, max 10.000 karakter)",
  "job_description": "string (min 30, max 10.000 karakter)",
  "user_id": "string (opsional)"
}
```

**Response:**
```json
{
  "matching_score": 0.78,
  "confidence": "High",
  "recommendation": "Highly Recommended",
  "inference_time_ms": 51.2
}
```

**Confidence & Recommendation Mapping:**

| Score | Confidence | Recommendation |
|---|---|---|
| ≥ 0.70 | High | Highly Recommended |
| 0.44 – 0.69 | Medium | Consider |
| < 0.44 | Low | Not Recommended |

---

#### `POST /api/v1/predict/batch`

**Request:**
```json
{
  "cv_text": "string",
  "job_descriptions": [
    "string — teks job description 1",
    "string — teks job description 2",
    "..."
  ],
  "user_id": "string (opsional)"
}
```

> `job_descriptions` adalah **array of strings** (plain text), bukan array of objects.  
> Maksimum **50 job descriptions** per request.

**Response:**
```json
{
  "results": [
    {
      "rank": 1,
      "job_index": 0,
      "matching_score": 0.78,
      "confidence": "High",
      "recommendation": "Highly Recommended",
      "inference_time_ms": 51.2
    },
    {
      "rank": 2,
      "job_index": 2,
      "matching_score": 0.31,
      "confidence": "Low",
      "recommendation": "Not Recommended",
      "inference_time_ms": 48.7
    }
  ],
  "total_items": 3,
  "total_time_ms": 312.4
}
```

> **`job_index`**: posisi job di array input (0-based). Backend gunakan ini untuk lookup data lengkap job dari database.

---

#### `POST /api/v1/skill-gap`

Menganalisis skill gap antara CV kandidat dan job description menggunakan **CsvSkillExtractor** (rule-based, ~685 job types, ~3000+ skill entries).

**Request:**
```json
{
  "cv_text": "string (min 50 karakter)",
  "job_description": "string (min 30 karakter)"
}
```

**Response:**
```json
{
  "skill_gap_score": 0.5,
  "skill_coverage_percent": "50%",
  "top_priority_skill": "machine learning",
  "present_skills": [
    { "skill": "python",     "skill_id": "python",     "match_score": 1.0, "priority": 0 },
    { "skill": "sql",        "skill_id": "sql",         "match_score": 1.0, "priority": 0 },
    { "skill": "pandas",     "skill_id": "pandas",      "match_score": 1.0, "priority": 0 }
  ],
  "missing_skills": [
    { "skill": "machine learning", "skill_id": "machine_learning", "match_score": 0.0, "priority": 1 },
    { "skill": "tensorflow",       "skill_id": "tensorflow",       "match_score": 0.0, "priority": 2 },
    { "skill": "deep learning",    "skill_id": "deep_learning",    "match_score": 0.0, "priority": 3 }
  ],
  "recommendation_summary": "Kesesuaian skill: 50% (cukup baik). Skill yang sudah dimiliki: python, sql, pandas. Pertimbangkan untuk mempelajari: machine learning, tensorflow, deep learning.",
  "analysis_time_ms": 45.0
}
```

> **`skill_id`**: format snake_case dari nama skill (bukan EMSI ID). Gunakan untuk de-duplikasi di sisi client.

**Response fields detail:**

| Field | Tipe | Keterangan |
|---|---|---|
| `skill_gap_score` | float 0–1 | Proporsi skill job yang terpenuhi CV |
| `skill_coverage_percent` | string | Format persen, misal `"50%"` |
| `top_priority_skill` | string | Skill missing pertama (prioritas tertinggi) |
| `present_skills` | array | Skill di CV yang cocok dengan job requirement |
| `missing_skills` | array | Skill job yang tidak ditemukan di CV, urut by prioritas |
| `recommendation_summary` | string | Ringkasan rekomendasi dalam Bahasa Indonesia |
| `analysis_time_ms` | float | Waktu analisis dalam millisecond |

---

#### `POST /api/v1/extract-cv-skills`

Ekstrak daftar skill dari teks CV saja (tanpa job description).

**Request:**
```json
{
  "cv_text": "string (min 50 karakter)"
}
```

**Response:**
```json
{
  "skills": ["python", "sql", "pandas", "data analysis", "machine learning"],
  "skill_count": 5,
  "extraction_time_ms": 12.3
}
```

---

#### `POST /api/v1/analyze-cv`

Ekstrak profil lengkap kandidat dari teks CV.

**Request:**
```json
{
  "cv_text": "string (min 100 karakter)"
}
```

**Response:**
```json
{
  "predicted_role": "Data Scientist",
  "experience_level": "Mid-level (3-5 years)",
  "education_level": "Bachelor's Degree",
  "top_skills": ["python", "machine learning", "tensorflow", "sql", "data analysis"],
  "suggested_job_titles": [
    "Data Scientist",
    "Machine Learning Engineer",
    "AI/ML Researcher",
    "Data Analyst",
    "NLP Engineer"
  ],
  "profile_summary": "Kandidat dengan latar belakang Data Science...",
  "analysis_time_ms": 87.4
}
```

---

#### `POST /api/v1/recommend`

Ranking beberapa job postings berdasarkan kecocokan dengan CV, disertai analisis kesiapan skill per industri.

**Request:**
```json
{
  "cv_text": "string",
  "job_postings": [
    {
      "job_id": "job_001",
      "job_title": "Data Scientist",
      "job_description": "string"
    },
    {
      "job_id": "job_002",
      "job_title": "ML Engineer",
      "job_description": "string"
    }
  ]
}
```

**Response:**
```json
{
  "ranked_jobs": [
    {
      "job_id": "job_001",
      "job_title": "Data Scientist",
      "matching_score": 0.82,
      "confidence": "High",
      "recommendation": "Highly Recommended",
      "rank": 1
    },
    {
      "job_id": "job_002",
      "job_title": "ML Engineer",
      "matching_score": 0.71,
      "confidence": "High",
      "recommendation": "Highly Recommended",
      "rank": 2
    }
  ],
  "industry_analysis": {
    "Technology": { "readiness_score": 0.85, "matched_skills": 12, "total_skills": 15 }
  },
  "total_time_ms": 420.5
}
```

---

#### `POST /api/v1/learning-path/analyze`

Generate rencana belajar untuk skill yang belum dimiliki kandidat. Menggunakan Gemini 2.5 Flash untuk menemukan kursus Coursera dan YouTube Data API untuk video tambahan. Hasil di-cache di Supabase (TTL 30 hari).

**Request:**
```json
{
  "user_id": "user_123",
  "target_skills": ["machine learning", "tensorflow", "deep learning"],
  "current_skills": ["python", "sql", "pandas"],
  "experience_level": "beginner"
}
```

**Response:**
```json
{
  "user_id": "user_123",
  "learning_paths": [
    {
      "skill": "machine learning",
      "priority": 1,
      "estimated_duration": "3-4 bulan",
      "courses": [
        {
          "title": "Machine Learning Specialization",
          "provider": "Coursera",
          "instructor": "Andrew Ng",
          "url": "https://coursera.org/...",
          "duration": "3 bulan",
          "level": "Beginner",
          "rating": 4.9
        }
      ],
      "youtube_resources": [
        {
          "title": "Machine Learning Course for Beginners",
          "channel": "freeCodeCamp",
          "url": "https://youtube.com/...",
          "duration": "9:52:19"
        }
      ],
      "cached": false
    }
  ],
  "total_skills": 3,
  "cached_skills": 1,
  "total_time_ms": 2340.5
}
```

---

#### `POST /api/v1/learning-path/refresh`

Paksa refresh cache Supabase untuk skill tertentu (hapus cache lama, generate baru).

**Request:**
```json
{
  "skill": "machine learning"
}
```

**Response:**
```json
{
  "skill": "machine learning",
  "status": "refreshed",
  "new_course_count": 5,
  "time_ms": 1820.3
}
```

---

#### `GET /api/v1/learning-path/courses/{skill}`

Ambil kursus dari cache Supabase untuk skill tertentu.

**Request:** URL parameter `skill` (lowercase, spasi diganti `%20` atau `-`)

**Response:**
```json
{
  "skill": "machine learning",
  "courses": [
    {
      "title": "Machine Learning Specialization",
      "provider": "Coursera",
      "url": "https://coursera.org/...",
      "rating": 4.9,
      "cached_at": "2026-05-20T08:31:12Z"
    }
  ],
  "count": 5,
  "cache_age_days": 8
}
```

---

## 📊 Contoh Request & Response (curl)

### Health Check

```bash
curl https://your-service-url.run.app/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v4",
  "optimal_threshold": 0.44,
  "hybrid_scoring": true
}
```

### Single Predict

```bash
curl -X POST https://your-service-url.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cv_text": "Experienced Data Scientist with 5 years experience in Python, TensorFlow, machine learning, and deep learning. Deployed 10+ production ML models. Strong background in statistical modeling and data analysis.",
    "job_description": "We are looking for a Data Scientist proficient in Python, machine learning frameworks (TensorFlow, PyTorch), statistical modeling, and SQL. Experience deploying models to production preferred."
  }'
```

```json
{
  "matching_score": 0.81,
  "confidence": "High",
  "recommendation": "Highly Recommended",
  "inference_time_ms": 53.4
}
```

### Batch Predict

```bash
curl -X POST https://your-service-url.run.app/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "cv_text": "Data Scientist with 5 years in Python TensorFlow machine learning deep learning NLP.",
    "job_descriptions": [
      "Senior Data Scientist — Python, TensorFlow, machine learning, production deployment required.",
      "Digital Marketing Manager — SEO, Google Ads, content strategy, social media campaigns.",
      "Frontend Developer — React.js, TypeScript, CSS, responsive design, REST APIs."
    ]
  }'
```

```json
{
  "results": [
    { "rank": 1, "job_index": 0, "matching_score": 0.83, "confidence": "High", "recommendation": "Highly Recommended", "inference_time_ms": 52.1 },
    { "rank": 2, "job_index": 2, "matching_score": 0.22, "confidence": "Low",  "recommendation": "Not Recommended",    "inference_time_ms": 50.3 },
    { "rank": 3, "job_index": 1, "matching_score": 0.17, "confidence": "Low",  "recommendation": "Not Recommended",    "inference_time_ms": 49.8 }
  ],
  "total_items": 3,
  "total_time_ms": 298.7
}
```

### Skill Gap Analysis

```bash
curl -X POST https://your-service-url.run.app/api/v1/skill-gap \
  -H "Content-Type: application/json" \
  -d '{
    "cv_text": "Data Analyst with 3 years experience. Skilled in SQL, Excel, PowerBI, and basic Python. Experience in data cleaning and visualization.",
    "job_description": "Data Scientist position requiring Python, machine learning, TensorFlow, deep learning, statistical modeling, and SQL. Must have experience with scikit-learn and model deployment."
  }'
```

```json
{
  "skill_gap_score": 0.33,
  "skill_coverage_percent": "33%",
  "top_priority_skill": "machine learning",
  "present_skills": [
    { "skill": "python",       "skill_id": "python",       "match_score": 1.0, "priority": 0 },
    { "skill": "sql",          "skill_id": "sql",           "match_score": 1.0, "priority": 0 },
    { "skill": "data analysis","skill_id": "data_analysis", "match_score": 1.0, "priority": 0 }
  ],
  "missing_skills": [
    { "skill": "machine learning",    "skill_id": "machine_learning",    "match_score": 0.0, "priority": 1 },
    { "skill": "tensorflow",          "skill_id": "tensorflow",           "match_score": 0.0, "priority": 2 },
    { "skill": "deep learning",       "skill_id": "deep_learning",        "match_score": 0.0, "priority": 3 },
    { "skill": "statistical modeling","skill_id": "statistical_modeling", "match_score": 0.0, "priority": 4 },
    { "skill": "scikit-learn",        "skill_id": "scikit-learn",         "match_score": 0.0, "priority": 5 },
    { "skill": "model deployment",    "skill_id": "model_deployment",     "match_score": 0.0, "priority": 6 }
  ],
  "recommendation_summary": "Kesesuaian skill: 33% (perlu peningkatan). Skill yang sudah dimiliki: python, sql, data analysis. Prioritaskan mempelajari: machine learning, tensorflow, deep learning.",
  "analysis_time_ms": 41.2
}
```

### Learning Path

```bash
curl -X POST https://your-service-url.run.app/api/v1/learning-path/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_demo",
    "target_skills": ["machine learning", "tensorflow"],
    "current_skills": ["python", "sql"],
    "experience_level": "beginner"
  }'
```

```json
{
  "user_id": "user_demo",
  "learning_paths": [
    {
      "skill": "machine learning",
      "priority": 1,
      "courses": [
        {
          "title": "Machine Learning Specialization",
          "provider": "Coursera",
          "instructor": "Andrew Ng",
          "url": "https://www.coursera.org/specializations/machine-learning-introduction"
        }
      ],
      "youtube_resources": [
        {
          "title": "Machine Learning Course – Full Beginner Tutorial",
          "channel": "freeCodeCamp.org"
        }
      ],
      "cached": false
    }
  ],
  "total_skills": 2,
  "cached_skills": 0,
  "total_time_ms": 2100.4
}
```

---

## 📈 Performance Metrics

### Model v4 (Aktif di Production)

| Metric | Nilai |
|---|---|
| **F1-Score** (threshold=0.44) | **0.9036** |
| **Accuracy** (threshold=0.44) | **88.65%** |
| Precision | 0.8792 |
| Recall | 0.9293 |
| MAE (regression) | 0.1105 |
| RMSE | 0.1783 |
| Pearson Correlation | 0.772 |
| Best Val MAE (epoch 70/80) | 0.10766 |
| Optimal threshold | **0.44** |

---

### Training Curves

> Huber Loss dan MAE konvergen stabil dalam 80 epoch tanpa overfitting — train/val loss berjalan berdekatan sepanjang training.

![Training History v4](notebooks/plots_v4/training_history_v4.png)

---

### Threshold Calibration

> Threshold 0.44 dipilih berdasarkan F1-sweep pada validation set — bukan hardcoded 0.5. Grafik menunjukkan titik F1 maksimum (garis biru vertikal) vs default 0.5 (garis kuning).

![Threshold Calibration](notebooks/plots_v4/threshold_calibration.png)

---

### Score Distribution

> Distribusi bimodal yang jelas antara pasangan "match" (hijau, skor tinggi) dan "tidak match" (merah, skor rendah) — bukti model belajar memisahkan kedua kelas dengan baik.

![Score Distribution v4](notebooks/plots_v4/score_dist_v4.png)

---

### Confusion Matrix & Prediction Quality

<div align="center">

| Confusion Matrix (threshold=0.5) | Prediction vs Actual |
|:---:|:---:|
| ![Confusion Matrix v4](notebooks/plots_v4/cm_v4.png) | ![Pred vs Actual v4](notebooks/plots_v4/pred_vs_actual_v4.png) |

</div>

> **Confusion Matrix**: dari ~16.000 sampel test, false negative (829) jauh lebih sedikit dari true positive (8319) — model tidak banyak melewatkan kandidat yang seharusnya match.  
> **Pred vs Actual**: titik-titik mengikuti garis diagonal (perfect prediction) dengan korelasi 0.772 — model belajar distribusi label kontinu, bukan sekadar klasifikasi binary.

---

### Inference Time (Cloud Run, warm instance)

| Endpoint | Latency |
|---|---|
| `/predict` (single) | ~50–80ms |
| `/api/v1/predict/batch` (10 jobs) | ~300–400ms |
| `/api/v1/skill-gap` | **~40–65ms** (CSV-based, no cold start) |
| `/api/v1/extract-cv-skills` | ~10–20ms |
| `/api/v1/analyze-cv` | ~80–120ms |
| `/api/v1/learning-path/analyze` (cache hit) | ~800–1200ms |
| `/api/v1/learning-path/analyze` (cache miss) | ~3–8 detik (Gemini API call) |

> Cold start (dari 0 instance): ~8–15 detik untuk load model TensorFlow ke memori. Gunakan `--min-instances 1` untuk menghindari cold start di production.

---

## 🎯 Training Pipeline

### Data

- Dataset: LinkedIn job postings + resume matching pairs (US market)
- Format: pasangan (CV, JD) dengan label continuous 0.0–1.0
- Data synthesis: 5-mode pair synthesizer untuk augmentasi

```
Raw LinkedIn Data
       │
       ▼
NLP Preprocessing (NLTK: tokenize, lemmatize, stopword removal)
       │
       ▼
Word2Vec Training (Gensim, dim=128, window=5, min_count=2)
       │
       ▼
Pair Synthesis (5 mode: positive, negative hard/soft, cross-domain, seniority)
       │
       ▼
Model Training (SkillAlignMatcherV4, 80 epochs, Huber Loss)
       │
       ▼
F1-Sweep Calibration → optimal_threshold = 0.44
       │
       ▼
Export: .keras + .pkl (preprocessor) + .json (config)
```

### Menjalankan Training (Google Colab / lokal)

```bash
# Training pipeline lengkap
python src/training/train.py \
  --model-version v4 \
  --epochs 80 \
  --batch-size 64 \
  --output-dir models/

# Atau buka notebook di Google Colab:
# notebooks/02U_model_development.ipynb
```

---

## 🗄️ Supabase Setup

Learning Path menggunakan Supabase untuk cache kursus (TTL 30 hari).

### DDL Tables

```sql
-- Jalankan di Supabase Dashboard → SQL Editor
-- Atau jalankan file lengkap: scripts/supabase_migrations.sql

CREATE TABLE IF NOT EXISTS skill_courses (
  id           BIGSERIAL PRIMARY KEY,
  skill_name   TEXT NOT NULL,
  course_data  JSONB NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_skill_courses_skill_name ON skill_courses(skill_name);
CREATE INDEX IF NOT EXISTS idx_skill_courses_created_at ON skill_courses(created_at);

CREATE TABLE IF NOT EXISTS learning_path_sessions (
  id           BIGSERIAL PRIMARY KEY,
  user_id      TEXT NOT NULL,
  session_data JSONB NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_lp_sessions_user_id ON learning_path_sessions(user_id);
```

### Cache Strategy

- **Cache hit**: skill sudah ada di `skill_courses` dan `created_at` < 30 hari → langsung return dari Supabase (~50ms)
- **Cache miss**: call Gemini API → simpan hasil ke Supabase → return (~3-8 detik)
- **Refresh**: endpoint `/learning-path/refresh` menghapus cache lama dan generate ulang

---

## ⚠️ Known Limitations

| Aspek | Keterbatasan |
|---|---|
| **Geografis** | Dataset LinkedIn US — kurang akurat untuk konteks Indonesia |
| **Industri** | Dominasi IT, Healthcare, Finance — logistik/manufaktur/pertanian kurang terwakili |
| **Bahasa** | Input harus Bahasa Inggris — input Bahasa Indonesia menghasilkan banyak OOV token dan menurunkan akurasi |
| **Skill Coverage** | `CsvSkillExtractor` hanya mengenali ~3000+ skill yang terdaftar di `job_skill_map.csv` — skill sangat niche atau baru mungkin tidak terdeteksi |
| **Cold Start** | Instance pertama setelah idle perlu ~8–15 detik untuk load model ke memori — gunakan `min-instances 1` untuk mitigasi |
| **Gemini Quota** | Gemini 2.5 Flash memiliki rate limit di tier gratis — hindari burst request banyak skill sekaligus |
| **YouTube API** | YouTube Data API v3 memiliki kuota harian 10.000 unit — monitor penggunaan di GCP Console |
| **Batch Limit** | `/api/v1/predict/batch` dibatasi 50 job descriptions per request |

---

## 👥 Tim

| Nama | Role |
|---|---|
| **Zahri Ramadhani** | AI Engineer |
| **Destian Aldi Nugraha** | AI Engineer |

**Capstone Project** — DBS Foundation Coding Camp 2026  
**Tim ID**: CC26-PSU318

---

## 📄 Lisensi

Capstone Project — DBS Foundation Coding Camp 2026.  
Tidak untuk distribusi komersial.
