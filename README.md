# SkillAlign AI Service

> **CV-Job Matching menggunakan Deep Learning & NLP**  
> Capstone Project — Coding Camp 2026  
> Tim ID: CC26-PSU318

## 📋 Overview

SkillAlign adalah model Deep Learning untuk CV-Job Matching yang menggunakan NLP (Natural Language Processing). Model ini menganalisis teks CV pengguna dan deskripsi lowongan kerja untuk menghasilkan skor kecocokan (matching score), serta merekomendasikan skill yang perlu ditingkatkan.

### Fitur Utama
- **Matching Score**: Skor kecocokan CV-Job (0-1)
- **Custom Attention Layer**: Cross-attention mechanism untuk CV-Job similarity
- **Focal Loss**: Penanganan class imbalance
- **F1-Score Callback**: Early stopping berbasis F1-score
- **REST API**: FastAPI service untuk inference
- **TensorBoard Integration**: Monitoring training metrics

## 🏗️ Arsitektur Model

```
Input_CV (seq_len) ──→ Embedding ──→ Conv1D ──┐
                                               ├──→ Custom Attention ──→ Concat ──→ Dense ──→ Sigmoid
Input_Job (seq_len) ──→ Embedding ──→ Conv1D ──┘                          ↑
                                               └──→ GlobalMaxPooling ─────┘
```

- **Dual-Input CNN** dengan Shared Embedding Layer
- **Custom Attention Layer** untuk cross-attention CV ↔ Job
- **TensorFlow Functional API**

## 📁 Struktur Proyek

```
SkillAlign-AI/
├── src/
│   ├── models/
│   │   ├── model_architecture.py      # Model definition (Functional API)
│   │   ├── custom_layers.py           # Custom Attention Layer
│   │   ├── custom_loss.py             # Focal Loss
│   │   └── custom_callbacks.py        # F1-Score Callback
│   │
│   ├── preprocessing/
│   │   ├── nlp_preprocessor.py        # Text preprocessing pipeline
│   │   ├── feature_engineering.py     # TF-IDF, skill matching
│   │   └── embeddings.py             # Word2Vec/FastText manager
│   │
│   ├── training/
│   │   ├── train.py                   # Training pipeline
│   │   ├── custom_training_loop.py    # tf.GradientTape (Side Quest)
│   │   └── hyperparameter_tuning.py   # Keras Tuner
│   │
│   ├── inference/
│   │   ├── predict.py                 # Inference functions
│   │   └── api_service.py            # FastAPI router
│   │
│   └── utils/
│       ├── metrics.py                 # Custom metrics
│       ├── error_handling.py          # Custom exceptions
│       ├── validation.py              # Input validation
│       └── visualization.py           # TensorBoard, plots
│
├── Dataset/                           # Training dataset
├── models/                            # Saved models
├── preprocessors/                     # Saved preprocessors
├── notebooks/                         # Jupyter notebooks
├── tests/                             # Unit tests
├── logs/                              # TensorBoard logs
├── main.py                            # FastAPI entry point
├── requirements.txt                   # Dependencies
└── README.md
```

## ⚙️ Setup & Instalasi 
### If you as ai engineer or data scientist follow step 1 until 4, and if you as software engineer follow step except step 3.

#### 1. Clone & Virtual Environment

```bash
cd SkillAlign-AI
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Initialization kernel 
```bash
python -m ipykernel install --user --name=venv --display-name="SkillAlign-AI"
```

Switch all ipynb kernel to "SkillAlign-AI" for run all ipynb files in order. Run file in notebooks.

#### 4. Run API Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs tersedia di: `http://localhost:8000/docs`

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📊 TensorBoard

```bash
tensorboard --logdir=logs/training
```

## 🎯 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Accuracy | ≥ 85% | ⏳ |
| MAE | ≤ 0.02 | ⏳ |
| Inference Time | < 500ms | ⏳ |

## 📝 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/predict` | Single prediction |
| POST | `/api/v1/predict` | Single prediction (versioned) |
| POST | `/api/v1/predict/batch` | Batch prediction |

## 👥 Tim

- **Zahri Ramadhani** — AI Engineer
- **Destian Aldi Nugraha** — AI Engineer

## 📄 Lisensi

Capstone Project — DBS Foundation Coding Camp 2026
