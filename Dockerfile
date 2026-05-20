FROM python:3.12-slim

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
RUN pip install --no-cache-dir -r requirements.txt

# ── spaCy language model ───────────────────────────────────────────────────
# en_core_web_lg (~800MB) dibutuhkan oleh SkillNer untuk /skill-gap & /extract-cv-skills
RUN python -m spacy download en_core_web_lg

# ── Copy source code ───────────────────────────────────────────────────────
COPY . .

# ── Create directories for artifacts ──────────────────────────────────────
# models/ dan preprocessors/ sudah ada via .gitkeep atau model files
# Buat juga logs/ agar tidak error saat startup
RUN mkdir -p models preprocessors logs

# ── Expose port ────────────────────────────────────────────────────────────
# Railway meng-override ini dengan $PORT env var via railway.toml startCommand
EXPOSE 8000

# ── Default start command ──────────────────────────────────────────────────
# Digunakan saat run lokal: docker run -p 8000:8000 skillaign-ai
# Railway menggunakan startCommand dari railway.toml (dengan $PORT)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
