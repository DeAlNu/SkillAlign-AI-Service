#!/usr/bin/env python3
"""
SkillAlign AI — Model Artifact Downloader

Dipakai saat deployment ke Railway (atau environment lain) di mana model v4
tidak ada di dalam Docker image karena ukurannya besar.

Cara kerja:
  1. Cek apakah setiap artifact sudah ada secara lokal.
  2. Jika tidak ada dan URL sudah dikonfigurasi via env var → download.
  3. Jika tidak ada dan URL tidak dikonfigurasi → log warning, skip.

Environment Variables yang perlu di-set di Railway:
  MODEL_DOWNLOAD_URL         = <public URL ke skillalign_matcher_v4.keras>
  PREPROCESSOR_DOWNLOAD_URL  = <public URL ke nlp_preprocessor_v4.pkl>
  EMB_MANAGER_DOWNLOAD_URL   = <public URL ke embedding_manager_v4.pkl>
  CONFIG_DOWNLOAD_URL        = <public URL ke model_config_v4.json>  (opsional)

Cara mendapatkan URL dari Google Drive:
  1. Buka file di Google Drive
  2. Klik kanan → "Bagikan" → "Siapa saja yang punya link"
  3. Salin link (format: https://drive.google.com/file/d/FILE_ID/view?usp=sharing)
  4. Ubah ke format download: https://drive.google.com/uc?export=download&id=FILE_ID

Alternatif: Upload ke GitHub Releases dan gunakan URL release asset langsung.
"""

import os
import sys
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Artifact specification ────────────────────────────────────────────────────
# Key  : environment variable yang menentukan download URL
# Value: (local_path, wajib)
ARTIFACTS = [
    {
        'name'         : 'Model v4 (.keras)',
        'local_path'   : os.getenv('MODEL_PATH', 'models/skillalign_matcher_v4.keras'),
        'url_env_var'  : 'MODEL_DOWNLOAD_URL',
        'required'     : True,
    },
    {
        'name'         : 'NLP Preprocessor v4 (.pkl)',
        'local_path'   : os.getenv('PREPROCESSOR_PATH', 'preprocessors/nlp_preprocessor_v4.pkl'),
        'url_env_var'  : 'PREPROCESSOR_DOWNLOAD_URL',
        'required'     : True,
    },
    {
        'name'         : 'Embedding Manager v4 (.pkl)',
        'local_path'   : os.getenv('EMB_MANAGER_PATH', 'preprocessors/embedding_manager_v4.pkl'),
        'url_env_var'  : 'EMB_MANAGER_DOWNLOAD_URL',
        'required'     : False,   # inference tidak butuh embedding manager
    },
    {
        'name'         : 'Model Config v4 (.json)',
        'local_path'   : os.getenv('CONFIG_PATH', 'models/model_config_v4.json'),
        'url_env_var'  : 'CONFIG_DOWNLOAD_URL',
        'required'     : False,   # untuk auto-read OPTIMAL_THRESHOLD
    },
]


def download_file(url: str, dest_path: str, chunk_size: int = 1024 * 1024) -> bool:
    """
    Download file dari URL ke local path.

    Mendukung Google Drive direct download links.
    Args:
        url: Download URL (harus direct download, bukan share page)
        dest_path: Path tujuan lokal
        chunk_size: Ukuran chunk per download cycle (default 1MB)
    Returns:
        True jika berhasil, False jika gagal
    """
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"  📥 Downloading: {url[:80]}...")
        with requests.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get('content-length', 0))
            downloaded = 0

            with open(dest, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                     if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            size_mb = downloaded / 1024 / 1024
            logger.info(f"  ✅ Saved → {dest_path} ({size_mb:.1f} MB)")
            return True

    except requests.exceptions.RequestException as e:
        logger.error(f"  ❌ Download gagal: {e}")
        if dest.exists():
            dest.unlink()   # hapus partial file
        return False


def ensure_models() -> bool:
    """
    Pastikan semua artifact v4 tersedia secara lokal.

    Dipanggil dari main.py lifespan sebelum model di-load.

    Returns:
        True jika semua artifact *required* tersedia, False jika ada yang missing.
    """
    all_required_ok = True
    any_downloaded  = False

    for art in ARTIFACTS:
        local_path = art['local_path']
        url_env    = art['url_env_var']
        name       = art['name']
        required   = art['required']

        # Sudah ada? Skip.
        if os.path.exists(local_path):
            size_mb = os.path.getsize(local_path) / 1024 / 1024
            logger.info(f"  ✅ {name}: sudah ada ({size_mb:.1f} MB)")
            continue

        # Belum ada — cek apakah URL dikonfigurasi
        url = os.getenv(url_env)
        if not url:
            msg = (
                f"  ⚠️  {name}: tidak ada lokal & {url_env} belum di-set.\n"
                f"       Set env var {url_env}=<download_url> di Railway\n"
                f"       atau upload file ke {local_path} secara manual."
            )
            if required:
                logger.error(msg)
                all_required_ok = False
            else:
                logger.warning(msg)
            continue

        # Download
        logger.info(f"  🔄 {name}: belum ada, mendownload dari {url_env}...")
        success = download_file(url, local_path)
        if not success and required:
            all_required_ok = False
        elif success:
            any_downloaded = True

    if any_downloaded:
        logger.info("Download selesai.")
    return all_required_ok


# ── CLI usage ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    logger.info("=== SkillAlign Model Downloader ===")
    ok = ensure_models()
    if ok:
        logger.info("✅ Semua artifact required tersedia. Service siap dijalankan.")
        sys.exit(0)
    else:
        logger.error("❌ Beberapa artifact required tidak tersedia.")
        sys.exit(1)
