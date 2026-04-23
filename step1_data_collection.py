"""
==============================================================================
STEP 1: DATA COLLECTION
Title: Transforming User Feedback into Strategic Intelligence:
       An ABSA of balé by BTN Superapp using IndoBERT and E-S-QUAL Framework
==============================================================================
Tujuan  : Mengambil (scrape) ulasan pengguna dari Google Play Store
Output  : data/raw_reviews.csv
Dependensi:
    pip install google-play-scraper pandas tqdm
==============================================================================
"""

import time
import logging
import pandas as pd
from tqdm import tqdm
from google_play_scraper import app, reviews, Sort

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
APP_ID        = "id.co.btn.bale"          # Package name balé by BTN di Play Store
LANG          = "id"                       # Bahasa Indonesia
COUNTRY       = "id"                       # Region Indonesia
TOTAL_TARGET  = 20_000                     # Target minimal ulasan
BATCH_SIZE    = 200                        # Ulasan per request (maks 200)
OUTPUT_PATH   = "data/raw_reviews.csv"
LOG_PATH      = "logs/step1_collection.log"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# FUNGSI UTAMA
# ─────────────────────────────────────────────

def fetch_app_metadata() -> dict:
    """Mengambil metadata aplikasi dari Play Store."""
    logger.info(f"Mengambil metadata aplikasi: {APP_ID}")
    metadata = app(APP_ID, lang=LANG, country=COUNTRY)
    logger.info(
        f"Aplikasi ditemukan: {metadata['title']} | "
        f"Rating: {metadata['score']:.2f} | "
        f"Total Reviews: {metadata['ratings']:,}"
    )
    return metadata


def scrape_reviews(total: int = TOTAL_TARGET) -> pd.DataFrame:
    """
    Melakukan scraping ulasan secara bertahap (pagination) menggunakan
    continuation_token agar tidak terjadi duplikasi.

    Args:
        total: jumlah ulasan yang ingin dikumpulkan

    Returns:
        DataFrame berisi ulasan mentah
    """
    all_reviews = []
    continuation_token = None
    sort_modes = [Sort.NEWEST, Sort.MOST_RELEVANT]  # Multi-sort untuk diversitas

    logger.info(f"Memulai scraping target {total:,} ulasan ...")

    with tqdm(total=total, desc="Scraping ulasan", unit="review") as pbar:
        for sort_mode in sort_modes:
            if len(all_reviews) >= total:
                break

            continuation_token = None   # reset per sort mode
            while len(all_reviews) < total:
                try:
                    result, continuation_token = reviews(
                        APP_ID,
                        lang=LANG,
                        country=COUNTRY,
                        sort=sort_mode,
                        count=BATCH_SIZE,
                        continuation_token=continuation_token
                    )

                    if not result:
                        logger.warning(f"Tidak ada data lagi pada sort={sort_mode}.")
                        break

                    all_reviews.extend(result)
                    pbar.update(len(result))
                    logger.debug(f"Terkumpul: {len(all_reviews):,} ulasan")

                    # Jeda antar-request untuk menghindari rate limiting
                    time.sleep(0.5)

                    if continuation_token is None:
                        break

                except Exception as e:
                    logger.error(f"Error saat scraping: {e}")
                    time.sleep(5)   # tunggu 5 detik sebelum retry
                    break

    logger.info(f"Total ulasan terkumpul (sebelum deduplikasi): {len(all_reviews):,}")
    return pd.DataFrame(all_reviews)


def clean_and_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan dan merestrukturisasi DataFrame hasil scraping.
    Kolom yang dipilih disesuaikan dengan kebutuhan penelitian ABSA.
    """
    # Pilih kolom relevan
    cols_map = {
        "reviewId"      : "review_id",
        "userName"      : "username",
        "userImage"     : "user_image_url",
        "content"       : "review_text",
        "score"         : "rating",
        "thumbsUpCount" : "helpful_count",
        "reviewCreatedVersion": "app_version",
        "at"            : "review_date",
        "replyContent"  : "dev_reply",
        "repliedAt"     : "dev_reply_date",
    }
    df = df.rename(columns={k: v for k, v in cols_map.items() if k in df.columns})

    # Hapus duplikat berdasarkan review_id
    before = len(df)
    df.drop_duplicates(subset=["review_id"], inplace=True)
    logger.info(f"Duplikat dihapus: {before - len(df):,} baris")

    # Hapus baris dengan teks kosong
    df.dropna(subset=["review_text"], inplace=True)
    df = df[df["review_text"].str.strip().str.len() > 5]

    # Tambahkan label rating kategori (berguna untuk analisis awal)
    df["sentiment_label_naive"] = df["rating"].apply(
        lambda r: "Positif" if r >= 4 else ("Negatif" if r <= 2 else "Netral")
    )

    # Reset index
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Dataset final: {len(df):,} ulasan")
    logger.info(f"Distribusi rating:\n{df['rating'].value_counts().sort_index()}")
    logger.info(
        f"Distribusi sentimen naif:\n"
        f"{df['sentiment_label_naive'].value_counts()}"
    )
    return df


def save_output(df: pd.DataFrame, path: str = OUTPUT_PATH) -> None:
    """Menyimpan hasil ke CSV dengan encoding UTF-8."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"Data disimpan ke: {path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 1. Ambil metadata
    metadata = fetch_app_metadata()

    # 2. Scrape ulasan
    raw_df = scrape_reviews(total=TOTAL_TARGET)

    # 3. Bersihkan dan strukturkan
    clean_df = clean_and_structure(raw_df)

    # 4. Simpan
    save_output(clean_df, OUTPUT_PATH)

    print("\n" + "="*60)
    print(f"✅ STEP 1 SELESAI")
    print(f"   Total ulasan tersimpan : {len(clean_df):,}")
    print(f"   Output file            : {OUTPUT_PATH}")
    print("="*60)
