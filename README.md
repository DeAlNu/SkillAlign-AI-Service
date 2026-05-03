# SkillAlign AI Service

> **CV-Job Matching menggunakan Deep Learning & NLP**  
> Capstone Project — DBS Foundation Coding Camp 2026  
> Tim ID: CC26-PSU318

## 📋 Overview

SkillAlign adalah model Deep Learning untuk **CV-Job Matching** menggunakan NLP. Model menerima teks CV dan teks job description, lalu menghasilkan skor kecocokan (0.0–1.0). AI Service ini berperan sebagai *scoring engine* yang dikonsumsi oleh Backend (Express.js) dalam alur integrasi Two-Stage Retrieval.

### Fitur Utama
- **Matching Score**: Skor kecocokan CV-Job (0.0–1.0)
- **Batch Endpoint**: Scoring 1 CV terhadap hingga 50 job sekaligus
- **Custom Attention Layer**: Cross-attention mechanism CV ↔ Job
- **Focal Loss**: Penanganan class imbalance
- **F1-Score Callback**: Early stopping berbasis F1-score
- **REST API**: FastAPI service dengan Swagger UI otomatis

---

## 🏗️ Arsitektur Model (v2)

```
cv_input (300)    job_input (300)
    │                  │
    └──── Shared Embedding (15k vocab × 128 dim) ────┘
         │                          │
   CNN Branch CV             CNN Branch Job
   Conv1D(128)→BN            Conv1D(128)→BN
   Conv1D(64) →BN            Conv1D(64) →BN
         │                          │
         └──── Custom Attention ────┘
                     │
              GlobalMaxPool
                     │
              Dense(256)→Dense(128)→Dense(64)
                     │
              Sigmoid → matching_score (0–1)
```

---

## 📁 Struktur Proyek

```
SkillAlign-AI/
├── src/
│   ├── models/
│   │   ├── model_architecture.py      # Dual-Input CNN + Attention
│   │   ├── custom_layers.py           # Custom Attention Layer
│   │   ├── custom_loss.py             # Focal Loss
│   │   └── custom_callbacks.py        # F1-Score Callback
│   ├── preprocessing/
│   │   ├── nlp_preprocessor.py        # Tokenizer + Lemmatizer
│   │   ├── feature_engineering.py     # TF-IDF, skill matching
│   │   └── embeddings.py              # Word2Vec manager
│   ├── training/
│   │   ├── train.py                   # Training pipeline
│   │   ├── custom_training_loop.py    # tf.GradientTape
│   │   └── hyperparameter_tuning.py   # Keras Tuner
│   ├── inference/
│   │   ├── predict.py                 # SkillAlignPredictor class
│   │   └── api_service.py             # FastAPI router (2 endpoints)
│   └── utils/
│       ├── metrics.py
│       ├── error_handling.py
│       ├── validation.py
│       └── visualization.py
├── Dataset/                           # Raw dataset (LinkedIn US)
├── models/
│   ├── skillalign_matcher_v2.keras    # Model utama (v2)
│   └── model_config_v2.json
├── preprocessors/
│   ├── nlp_preprocessor_v2.pkl        # Tokenizer (vocab 15k)
│   └── embedding_manager_v2.pkl       # Word2Vec (128-dim)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_development.ipynb
│   ├── 02U_model_development.ipynb    # Training pipeline v2 (gunakan ini)
│   └── 03_hyperparameter_tuning.ipynb
├── tests/
│   ├── test_inference_csv.py          # Batch inference testing
│   └── test_data/inference_test_cases.csv
├── logs/                              # TensorBoard logs
├── main.py                            # FastAPI entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Instalasi

> **AI Engineer / Data Scientist**: ikuti semua langkah (1–5).  
> **Software Engineer / Backend**: ikuti langkah 1, 2, 4, 5 saja (skip langkah 3).

### 1. Virtual Environment

```bash
# Masuk ke folder project
cd SkillAlign-AI

# Buat virtual environment
python -m venv venv

# Aktifkan (Windows)
.\venv\Scripts\activate

# Aktifkan (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Training Model (AI Engineer)

Jalankan notebook secara berurutan menggunakan kernel **SkillAlign-AI**:

```bash
# Register kernel Jupyter
python -m ipykernel install --user --name=venv --display-name="SkillAlign-AI"

# Buka Jupyter
jupyter notebook
```

Urutan notebook:
1. `notebooks/01_eda.ipynb` — Exploratory Data Analysis
2. `notebooks/02U_model_development.ipynb` — **Training pipeline v2** (gunakan ini)
3. `notebooks/03_hyperparameter_tuning.ipynb` — Hyperparameter tuning (opsional)

Hasil training akan tersimpan di:
- `models/skillalign_matcher_v2.keras`
- `preprocessors/nlp_preprocessor_v2.pkl`
- `preprocessors/embedding_manager_v2.pkl`

### 4. Jalankan API Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server berhasil jika muncul log:
```
INFO - Model loaded successfully on startup.
INFO - Application startup complete.
```

> **Catatan**: Jika model belum ditraining, server tetap berjalan tapi endpoint `/predict` akan return HTTP 503.

### 5. Akses Swagger UI

Buka browser: **http://localhost:8000/docs**

---

## 🧪 Testing API

Ada **3 cara** untuk mengetes API setelah server berjalan:

---

### Cara 1 — Swagger UI (Paling Mudah)

1. Buka **http://localhost:8000/docs**
2. Klik endpoint yang ingin ditest
3. Klik **"Try it out"**
4. Isi request body → klik **"Execute"**

---

### Cara 2 — curl (Terminal)

**Test health check:**
```bash
curl http://localhost:8000/health
```
Expected response:
```json
{ "status": "healthy", "model_loaded": true }
```

**Test single prediction (`/predict`):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{
    \"cv_text\": \"Experienced Data Scientist with 5 years in Python TensorFlow machine learning deep learning and statistical analysis. Deployed 10+ production models.\",
    \"job_description\": \"Looking for a Data Scientist with strong Python skills, experience in ML frameworks TensorFlow and data analysis.\"
  }"
```

Expected response:
```json
{
  "matching_score": 0.78,
  "confidence": "High",
  "recommendation": "Highly Recommended"
}
```

**Test batch prediction (`/api/v1/predict/batch`):**
```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d "{
    \"cv_text\": \"Experienced Data Scientist with 5 years in Python TensorFlow machine learning.\",
    \"job_descriptions\": [
      \"Looking for a Data Scientist with Python and ML skills.\",
      \"Marketing Manager needed for digital campaigns and SEO strategy.\",
      \"Frontend Developer with React and JavaScript experience.\"
    ]
  }"
```

Expected response — **sudah diurutkan dari skor tertinggi**, dilengkapi `rank` dan `job_index`:
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
    },
    {
      "rank": 3,
      "job_index": 1,
      "matching_score": 0.25,
      "confidence": "Low",
      "recommendation": "Not Recommended",
      "inference_time_ms": 49.1
    }
  ],
  "total_items": 3,
  "total_time_ms": 312.4
}
```

> **Cara membaca response:**
> - `rank` → peringkat kesesuaian (1 = paling cocok)
> - `job_index` → posisi job di array input yang kamu kirim (0-based)
>
> Pada contoh di atas: job ke-0 dari input mendapat rank 1, job ke-2 mendapat rank 2, dst.
> Backend gunakan `job_index` untuk ambil data lengkap job (judul, perusahaan, dll.) dari database.

---

### Cara 3 — Script Testing CSV (Batch Evaluation)

Untuk mengevaluasi model terhadap 40 test cases sekaligus:

```bash
# Dari folder SkillAlign-AI/
cd tests
python test_inference_csv.py
```

Hasil evaluasi tersimpan di: `tests/results/inference_test_results.csv`

---

## 📊 Confidence & Recommendation Mapping

| Score | Confidence | Recommendation |
|---|---|---|
| ≥ 0.70 | High | Highly Recommended |
| 0.40 – 0.69 | Medium | Consider |
| < 0.40 | Low | Not Recommended |

---

## 📡 API Endpoints Summary

| Method | Endpoint | Deskripsi | Use Case |
|---|---|---|---|
| GET | `/` | Service info | Cek status |
| GET | `/health` | Health check + model status | Load balancer |
| POST | `/predict` | Single CV vs 1 Job | Testing & demo |
| POST | `/api/v1/predict` | Single CV vs 1 Job (versioned) | Testing & demo |
| POST | `/api/v1/predict/batch` | 1 CV vs ≤50 Jobs | **Integrasi Backend (ranking)** |
| POST | `/api/v1/skill-gap` | Analisis skill gap CV vs Job | **Rekomendasi skill** |

### Request Schema

**Single (`/predict` atau `/api/v1/predict`):**
```json
{
  "cv_text": "string (min 50 char, max 10.000 char)",
  "job_description": "string (min 30 char, max 10.000 char)",
  "user_id": "string (opsional)"
}
```

**Batch (`/api/v1/predict/batch`) — Request:**
```json
{
  "cv_text": "string (min 50 char, max 10.000 char)",
  "job_descriptions": ["string", "string", "..."],
  "user_id": "string (opsional)"
}
```
> Maksimum 50 job descriptions per request.

**Batch — Response (diurutkan dari skor tertinggi):**
```json
{
  "results": [
    {
      "rank": 1,           
      "job_index": 2,      
      "matching_score": 0.85,
      "confidence": "High",
      "recommendation": "Highly Recommended",
      "inference_time_ms": 51.2
    }
  ],
  "total_items": 3,
  "total_time_ms": 312.4
}
```

| Field | Tipe | Keterangan |
|---|---|---|
| `rank` | int | Peringkat kesesuaian (1 = terbaik) |
| `job_index` | int | Posisi job di array input (0-based) — gunakan ini untuk lookup data job di Backend |
| `matching_score` | float | Skor 0.0–1.0 |
| `confidence` | string | High / Medium / Low |
| `recommendation` | string | Highly Recommended / Consider / Not Recommended |
| `inference_time_ms` | float | Waktu inferensi (ms) |

**Contoh penggunaan `job_index` di Backend (Express.js):**
```javascript
const jobs = await db.query('SELECT * FROM jobs WHERE ... LIMIT 50');
const jobDescriptions = jobs.map(j => j.description);

const aiResponse = await axios.post('/api/v1/predict/batch', {
  cv_text: cvText,
  job_descriptions: jobDescriptions
});

// Gabungkan skor dengan data job asli menggunakan job_index
const ranked = aiResponse.data.results.map(item => ({
  rank:           item.rank,
  job:            jobs[item.job_index],   // lookup ke data job asli
  score:          item.matching_score,
  recommendation: item.recommendation
}));
// ranked[0] = job terbaik untuk CV ini
```

---

### Skill Gap (`/api/v1/skill-gap`)

**Request:**
```json
{
  "cv_text": "string (min 50 char)",
  "job_description": "string (min 30 char)"
}
```

**Response:**
```json
{
  "skill_gap_score": 0.20,
  "skill_coverage_percent": "20%",
  "top_priority_skill": "tensorflow",
  "present_skills": [
    { "skill": "python", "weight": 0.45, "priority": 0 }
  ],
  "missing_skills": [
    { "skill": "tensorflow", "weight": 0.38, "priority": 1 },
    { "skill": "mlops",      "weight": 0.31, "priority": 2 },
    { "skill": "docker",     "weight": 0.28, "priority": 3 }
  ],
  "recommendation_summary": "Kesesuaian skill: 20% (perlu peningkatan). Prioritaskan mempelajari: tensorflow, mlops, docker.",
  "analysis_time_ms": 12.4
}
```

| Field | Keterangan |
|---|---|
| `skill_gap_score` | 0.0–1.0, skor kesesuaian skill |
| `skill_coverage_percent` | Persentase skill requirement yang sudah terpenuhi |
| `top_priority_skill` | Skill paling penting untuk dipelajari duluan |
| `present_skills` | Skill yang sudah ada di CV sesuai requirement |
| `missing_skills` | Skill yang kurang, diurutkan berdasarkan prioritas |
| `recommendation_summary` | Ringkasan rekomendasi natural language |

**Test curl:**
```bash
curl -X POST http://localhost:8000/api/v1/skill-gap \
  -H "Content-Type: application/json" \
  -d "{
    \"cv_text\": \"Data Analyst with 3 years experience. Skilled in SQL, Excel, and PowerBI. Basic Python knowledge for data cleaning and visualization.\",
    \"job_description\": \"Data Scientist position requiring Python, machine learning, TensorFlow, statistical modeling, and A/B testing experience.\"
  }"

---

## 🔧 Environment Variables (Opsional)

Untuk mengganti model path tanpa edit kode:

```bash
# Windows (PowerShell)
$env:MODEL_PATH = "models/skillalign_matcher_v2.keras"
$env:PREPROCESSOR_PATH = "preprocessors/nlp_preprocessor_v2.pkl"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Linux/Mac
MODEL_PATH=models/skillalign_matcher_v2.keras \
PREPROCESSOR_PATH=preprocessors/nlp_preprocessor_v2.pkl \
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📊 TensorBoard

```bash
tensorboard --logdir=logs/training
```
Buka browser: **http://localhost:6006**

---

## ⚠️ Known Limitations

| Aspek | Keterbatasan |
|---|---|
| **Geografis** | Dataset LinkedIn US — kurang akurat untuk konteks Indonesia |
| **Industri** | Dominasi IT, Healthcare, Finance — logistik/manufaktur kurang terwakili |
| **Bahasa** | Bahasa Inggris saja — input Bahasa Indonesia → banyak OOV token |
| **Skills** | Skill khusus Indonesia (SIO, K3, PPJK) tidak dikenali model |

---

## 🎯 Performance Model v2

| Metric | Target | Hasil (Internal Test) |
|---|---|---|
| Accuracy | ≥ 85% | **90.03%** ✅ |
| F1-Score | ≥ 0.75 | **0.8972** ✅ |
| AUC | — | **0.9678** |
| Inference Time | < 500ms | **~50ms** ✅ |

> **Catatan**: Internal test menggunakan data synthetic. Pass rate pada test cases logistik Indonesia lebih rendah karena domain mismatch (lihat Known Limitations).

---

## 👥 Tim

- **Zahri Ramadhani** — AI Engineer
- **Destian Aldi Nugraha** — AI Engineer

## 📄 Lisensi

Capstone Project — DBS Foundation Coding Camp 2026
