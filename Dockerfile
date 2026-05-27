FROM python:3.10-slim

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────
# gcc/g++ untuk compile ekstensi native (TensorFlow, gensim, scikit-learn)
# curl untuk health check di local testing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────
# Salin requirements dulu agar layer ini di-cache jika kode berubah tapi deps tidak
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── spaCy language model ───────────────────────────────────────────────────
# en_core_web_lg (~800MB) dibutuhkan oleh SkillNer untuk /skill-gap & /extract-cv-skills
RUN python -m spacy download en_core_web_lg

# ── SentenceTransformer model ──────────────────────────────────────────────
# all-MiniLM-L6-v2 (~90MB) dibutuhkan oleh semantic fallback di skill_gap.py
# Di-pre-download ke image agar tidak download saat runtime (cold start Cloud Run)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Copy source code ───────────────────────────────────────────────────────
COPY . .

# ── Create directories for artifacts ──────────────────────────────────────
# models/ dan preprocessors/ sudah ada via .gitkeep atau model files
# Buat juga logs/ agar tidak error saat startup
RUN mkdir -p models preprocessors logs

# ── Expose port ────────────────────────────────────────────────────────────
# Cloud Run & Railway inject $PORT — default 8000 untuk local docker run
EXPOSE 8000

# ── Default start command ──────────────────────────────────────────────────
# Menggunakan sh -c agar $PORT env var bisa di-expand:
#   - Cloud Run : PORT=8080 (inject otomatis)
#   - Railway   : PORT=xxxx (inject via railway.toml / env var)
#   - Local     : PORT tidak di-set → fallback ke 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
