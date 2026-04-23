"""
==============================================================================
STEP 3: ASPECT LABELING (E-S-QUAL Framework)
Title: Transforming User Feedback into Strategic Intelligence:
       An ABSA of balé by BTN Superapp using IndoBERT and E-S-QUAL Framework
==============================================================================
Tujuan  : Memberikan label aspek (E-S-QUAL) pada setiap ulasan menggunakan
          pendekatan hybrid:
          (a) Rule-based keyword matching untuk labeling awal
          (b) Klasterisasi untuk validasi semi-otomatis
          (c) Export ke format CSV untuk anotasi pakar manusia

Dimensi E-S-QUAL yang digunakan:
  1. EFFICIENCY       - Kemudahan dan kecepatan navigasi
  2. SYSTEM_AVAIL     - Ketersediaan dan stabilitas sistem
  3. FULFILLMENT      - Pemenuhan janji layanan
  4. PRIVACY          - Keamanan dan perlindungan data

Input   : data/preprocessed_reviews.csv
Output  :
  - data/labeled_reviews.csv        (label rule-based)
  - data/annotation_sample.csv      (sampel untuk anotasi pakar)
  - data/aspect_distribution.csv    (distribusi aspek)

Dependensi:
    pip install pandas numpy scikit-learn sentence-transformers tqdm
==============================================================================
"""

import re
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/step3_labeling.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

tqdm.pandas()

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
INPUT_PATH      = "data/preprocessed_reviews.csv"
LABELED_PATH    = "data/labeled_reviews.csv"
ANNOTATION_PATH = "data/annotation_sample.csv"
DISTRIB_PATH    = "data/aspect_distribution.csv"
ANNOTATION_N    = 500   # Jumlah sampel untuk anotasi pakar


# ─────────────────────────────────────────────
# KAMUS ASPEK E-S-QUAL (Domain Perbankan Digital Indonesia)
# ─────────────────────────────────────────────
ASPECT_KEYWORDS: Dict[str, List[str]] = {

    "EFFICIENCY": [
        # UX/UI & kemudahan penggunaan
        "mudah", "gampang", "simpel", "simple",
        "praktis", "intuitif", "user friendly",
        "navigasi mudah", "tidak ribet", "nggak ribet",
        "tidak susah", "mudah digunakan",
        "tampilan", "interface", "ui", "ux", "desain",
        "menu", "halaman", "fitur mudah",
        "enak digunakan", "nyaman digunakan",

        # Kecepatan (UX-level, bukan system failure)
        "cepat", "responsif", "lancar", "smooth",
        "loading cepat", "buka cepat",
        "proses cepat", "respon cepat",
    ],

    "SYSTEM_AVAILABILITY": [
        # Error & crash
        "error", "bug", "crash", "force close",
        "keluar sendiri", "aplikasi mati",
        "tidak bisa dibuka", "tidak bisa diakses",

        # Kegagalan fungsi
        "tidak bisa", "gagal", "tidak berfungsi",
        "tidak bekerja", "tidak jalan",

        # Kasus real user (PENTING)
        "tidak bisa login", "gagal login",
        "tidak bisa masuk", "login gagal",
        "tidak bisa transfer", "gagal transfer",
        "tidak bisa top up", "tidak bisa bayar",

        # Loading & performa
        "loading lama", "loading terus",
        "tidak selesai", "stuck", "freeze",
        "hang", "lag", "lemot", "lambat", "lelet",

        # Server & koneksi
        "server down", "server error",
        "maintenance", "gangguan",
        "koneksi", "timeout",

        # Stabilitas
        "sering error", "sering bermasalah",
        "tidak stabil", "update bermasalah",
    ],

    "FULFILLMENT": [
        # Keberhasilan transaksi
        "berhasil", "sukses",
        "transaksi berhasil",
        "transfer berhasil",
        "pembayaran berhasil",

        # Masalah transaksi (REAL DATA CRITICAL)
        "uang tidak masuk",
        "uang belum masuk",
        "saldo tidak masuk",
        "saldo berkurang",
        "saldo terpotong",
        "saldo kepotong",
        "uang hilang",

        # Notifikasi & bukti
        "notifikasi", "pemberitahuan",
        "bukti transaksi", "riwayat transaksi",

        # Akurasi & keandalan
        "sesuai", "akurat", "tepat", "benar",

        # Produk bank
        "kpr", "angsuran", "tagihan", "cicilan",
        "bayar tagihan",

        # Reward
        "promo", "cashback", "reward", "poin",
    ],

    "PRIVACY": [
        # Keamanan umum
        "aman", "keamanan", "privasi",
        "perlindungan data",

        # Risiko
        "data bocor", "kebocoran data",
        "tidak aman", "diretas", "hacked",
        "penipuan", "scam", "phishing",

        # Autentikasi
        "otp", "kode otp",
        "otp tidak masuk", "otp gagal",
        "verifikasi", "autentikasi",

        # Kredensial
        "pin", "password", "kata sandi",
        "ganti pin", "ubah password",

        # Biometrik
        "sidik jari", "fingerprint",
        "face id", "pengenalan wajah",

        # Session issue
        "logout sendiri", "sesi habis",
        "akun terkunci", "akun terblokir",
    ],
}

# Kata-kata sentimen umum bahasa Indonesia
SENTIMENT_POSITIVE = [
    "bagus", "baik", "mantap", "keren",
    "luar biasa", "hebat", "canggih",
    "suka", "senang", "puas", "memuaskan",
    "membantu", "berguna", "bermanfaat",
    "terbaik", "recommended", "rekomen",
    "top", "oke", "ok", "good", "great","excellent",

    # intensifier
    "sangat bagus", "sangat baik",
    "sangat membantu", "sangat puas"
]

SENTIMENT_NEGATIVE = [
    "buruk", "jelek", "parah", "payah",
    "kecewa", "mengecewakan",
    "kesal", "marah", "frustrasi",
    "tidak puas",

    # fungsi gagal
    "tidak bisa", "gagal",
    "error", "bermasalah", "trouble",

    # performa
    "lemot", "lambat", "loading lama",

    # keluhan eksplisit
    "tolong diperbaiki", "segera diperbaiki"
    "mohon diperbaiki",
    "harus diperbaiki",

    # rating
    "bintang 1", "rating jelek",

    # frasa kuat
    "tidak berguna",
    "sangat buruk",
    "sangat lambat", "tidak recommend", "tidak rekomen",
]

# ─────────────────────────────────────────────
# KELAS LABELER ASPEK
# ─────────────────────────────────────────────
class ESQUALAspectLabeler:
    """
    Labeler aspek berbasis aturan menggunakan kamus keyword E-S-QUAL.
    Mendukung multi-label per ulasan (satu ulasan bisa mengandung
    beberapa aspek sekaligus).
    """

    def __init__(
        self,
        aspect_keywords: Dict[str, List[str]] = ASPECT_KEYWORDS,
        sentiment_pos: List[str] = SENTIMENT_POSITIVE,
        sentiment_neg: List[str] = SENTIMENT_NEGATIVE,
    ):
        self.aspect_kw   = aspect_keywords
        self.sent_pos    = sentiment_pos
        self.sent_neg    = sentiment_neg

        # Compile regex untuk setiap aspek dan sentimen
        self._compile_patterns()

    def _compile_patterns(self):
        """Precompile regex patterns untuk efisiensi."""
        self.aspect_patterns = {
            aspect: re.compile(
                r"\b(" + "|".join(re.escape(kw) for kw in keywords) + r")\b",
                re.IGNORECASE
            )
            for aspect, keywords in self.aspect_kw.items()
        }
        self.pos_pattern = re.compile(
            r"\b(" + "|".join(re.escape(w) for w in self.sent_pos) + r")\b",
            re.IGNORECASE
        )
        self.neg_pattern = re.compile(
            r"\b(" + "|".join(re.escape(w) for w in self.sent_neg) + r")\b",
            re.IGNORECASE
        )

    def detect_aspects(self, text: str) -> List[str]:
        """
        Deteksi aspek yang muncul dalam teks.
        Returns: list aspek yang terdeteksi
        """
        detected = []
        for aspect, pattern in self.aspect_patterns.items():
            if pattern.search(text):
                detected.append(aspect)
        return detected if detected else ["GENERAL"]

    def detect_sentiment(self, text: str, rating: int = None) -> str:
        """
        Deteksi sentimen berbasis keyword + rating pengguna.

        Strategi:
        - Prioritas 1: rating eksplisit (5 → Positif, 1-2 → Negatif)
        - Prioritas 2: keyword matching
        - Default: Netral
        """
        pos_count = len(self.pos_pattern.findall(text))
        neg_count = len(self.neg_pattern.findall(text))

        # Jika ada rating, gunakan sebagai sinyal utama
        if rating is not None:
            if rating >= 4:
                return "Positif" if pos_count >= neg_count else "Netral"
            elif rating <= 2:
                return "Negatif" if neg_count >= pos_count else "Netral"

        # Fallback ke keyword
        if pos_count > neg_count:
            return "Positif"
        elif neg_count > pos_count:
            return "Negatif"
        else:
            return "Netral"

    def label_row(self, row: pd.Series) -> pd.Series:
        """
        Memberikan label aspek dan sentimen pada satu baris ulasan.
        Returns pandas Series dengan kolom tambahan.
        """
        text   = str(row.get("clean_text", ""))
        rating = row.get("rating", None)

        aspects   = self.detect_aspects(text)
        sentiment = self.detect_sentiment(text, rating)

        # Skor keyakinan sederhana: berapa keyword yang cocok
        matched_kw = {}
        for aspect in aspects:
            if aspect != "GENERAL":
                matches = self.aspect_patterns[aspect].findall(text)
                matched_kw[aspect] = len(matches)

        return pd.Series({
            "aspects"           : "|".join(aspects),         # Multi-label dipisah "|"
            "aspect_count"      : len(aspects),
            "sentiment_rule"    : sentiment,
            "matched_keywords"  : str(matched_kw),
            "is_multilabel"     : len(aspects) > 1,
        })


# ─────────────────────────────────────────────
# ANALISIS KLASTERISASI (validasi aspek)
# ─────────────────────────────────────────────
def cluster_aspects_with_embeddings(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    n_clusters: int = 8,
    sample_size: int = 5000,
) -> pd.DataFrame:
    """
    Validasi distribusi aspek menggunakan klasterisasi embedding.
    Berguna untuk menemukan pola topik yang tidak tertangkap kamus.

    Args:
        df          : DataFrame berisi ulasan
        text_col    : nama kolom teks yang sudah dibersihkan
        n_clusters  : jumlah klaster
        sample_size : sampel untuk efisiensi komputasi

    Returns:
        DataFrame dengan kolom 'cluster_id' tambahan
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import MiniBatchKMeans

        logger.info("Memuat model SentenceTransformer untuk klasterisasi ...")
        model = SentenceTransformer("firqaaa/indo-sentence-bert-base")

        sample_df = df.sample(min(sample_size, len(df)), random_state=42)
        texts     = sample_df[text_col].tolist()

        logger.info(f"Membuat embedding untuk {len(texts):,} ulasan ...")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

        logger.info(f"Clustering dengan k={n_clusters} ...")
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        sample_df = sample_df.copy()
        sample_df["cluster_id"] = labels

        # Tampilkan kata kunci per klaster
        for i in range(n_clusters):
            cluster_texts = sample_df[sample_df["cluster_id"] == i][text_col]
            words = " ".join(cluster_texts).split()
            from collections import Counter
            top_words = Counter(words).most_common(10)
            logger.info(f"Klaster {i}: {top_words}")

        return sample_df

    except ImportError:
        logger.warning("sentence-transformers tidak tersedia. Klasterisasi dilewati.")
        return df


# ─────────────────────────────────────────────
# EKSPOR UNTUK ANOTASI PAKAR
# ─────────────────────────────────────────────
def prepare_annotation_sample(
    df: pd.DataFrame,
    n: int = ANNOTATION_N,
    output_path: str = ANNOTATION_PATH
) -> None:
    """
    Menyiapkan sampel stratifikasi untuk anotasi pakar manusia.
    Stratifikasi berdasarkan:
    - Rating (1-5) → pastikan kelas minoritas terwakili
    - Aspek rule-based → validasi distribusi

    Format output sesuai standar anotasi dua anotator.
    """
    # Stratified sampling berdasarkan rating
    try:
        sample = df.groupby("rating", group_keys=False).apply(
            lambda x: x.sample(min(len(x), n // 5), random_state=42)
        ).sample(frac=1, random_state=42).head(n)
    except Exception:
        sample = df.sample(min(n, len(df)), random_state=42)

    # Kolom yang dibutuhkan anotator
    annotation_cols = [
        "review_id",
        "review_text",       # Teks asli
        "clean_text",        # Teks yang sudah dibersihkan
        "rating",
        "aspects",           # Label rule-based (untuk referensi)
        "sentiment_rule",    # Sentimen rule-based (untuk referensi)
        # Kolom kosong yang akan diisi anotator
    ]

    available_cols = [c for c in annotation_cols if c in sample.columns]
    annotation_df  = sample[available_cols].copy()

    # Tambahkan kolom anotasi kosong
    annotation_df["aspect_annotator_1"]    = ""   # EFFICIENCY/SYSTEM_AVAIL/FULFILLMENT/PRIVACY/GENERAL
    annotation_df["sentiment_annotator_1"] = ""   # Positif/Negatif/Netral
    annotation_df["aspect_annotator_2"]    = ""
    annotation_df["sentiment_annotator_2"] = ""
    annotation_df["agreement"]             = ""   # Y/N (diisi setelah kedua anotator selesai)
    annotation_df["notes"]                 = ""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    annotation_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Sampel anotasi ({n} baris) disimpan ke: {output_path}")


# ─────────────────────────────────────────────
# PIPELINE UTAMA
# ─────────────────────────────────────────────
def run_labeling_pipeline(
    input_path: str  = INPUT_PATH,
    labeled_path: str = LABELED_PATH,
) -> pd.DataFrame:

    # 1. Load data
    logger.info(f"Membaca data dari: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    logger.info(f"Jumlah baris: {len(df):,}")

    # 2. Labeling aspek dengan rule-based
    logger.info("Melakukan pelabelan aspek E-S-QUAL berbasis aturan ...")
    labeler     = ESQUALAspectLabeler()
    label_df    = df.progress_apply(labeler.label_row, axis=1)
    df          = pd.concat([df, label_df], axis=1)

    # 3. Distribusi aspek
    logger.info("Menghitung distribusi aspek ...")
    aspect_counts = {}
    for aspect in list(ASPECT_KEYWORDS.keys()) + ["GENERAL"]:
        count = df["aspects"].str.contains(aspect).sum()
        aspect_counts[aspect] = count
        logger.info(f"  {aspect:<25}: {count:>6,} ulasan ({count/len(df)*100:.1f}%)")

    distrib_df = pd.DataFrame(
        list(aspect_counts.items()), columns=["Aspect", "Count"]
    )
    distrib_df["Percentage"] = (distrib_df["Count"] / len(df) * 100).round(2)
    Path(DISTRIB_PATH).parent.mkdir(parents=True, exist_ok=True)
    distrib_df.to_csv(DISTRIB_PATH, index=False, encoding="utf-8-sig")

    # 4. Distribusi sentimen per aspek
    logger.info("\nDistribusi sentimen per aspek:")
    for aspect in ASPECT_KEYWORDS.keys():
        subset   = df[df["aspects"].str.contains(aspect)]
        sent_dist = subset["sentiment_rule"].value_counts()
        logger.info(f"\n  {aspect}:\n{sent_dist.to_string()}")

    # 5. Simpan labeled dataset
    Path(labeled_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(labeled_path, index=False, encoding="utf-8-sig")
    logger.info(f"Dataset berlabel disimpan ke: {labeled_path}")

    # 6. Siapkan sampel untuk anotasi pakar
    prepare_annotation_sample(df)

    return df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = run_labeling_pipeline()

    print("\n" + "="*60)
    print("✅ STEP 3 SELESAI")
    print(f"   Total ulasan berlabel  : {len(df):,}")
    print(f"   Output file            : {LABELED_PATH}")
    print(f"   File anotasi pakar     : {ANNOTATION_PATH}")
    print(f"   Distribusi aspek       : {DISTRIB_PATH}")
    print("="*60)

    # Tampilkan contoh
    sample = df[["clean_text", "aspects", "sentiment_rule", "rating"]].sample(5)
    print("\n📋 Contoh Hasil Pelabelan:")
    pd.set_option("display.max_colwidth", 60)
    print(sample.to_string(index=False))
