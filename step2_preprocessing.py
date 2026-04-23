"""
==============================================================================
STEP 2: TEXT PREPROCESSING
Title: Transforming User Feedback into Strategic Intelligence:
       An ABSA of balé by BTN Superapp using IndoBERT and E-S-QUAL Framework
==============================================================================
Tujuan  : Membersihkan teks mentah ulasan dan menyiapkannya untuk pelabelan
          dan fine-tuning model
Input   : data/raw_reviews.csv
Output  : data/preprocessed_reviews.csv
Dependensi:
    pip install pandas numpy nltk Sastrawi emoji regex langdetect tqdm
==============================================================================
"""

import re
import logging
import unicodedata
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
INPUT_PATH  = "data/raw_reviews.csv"
OUTPUT_PATH = "data/preprocessed_reviews.csv"
LOG_PATH    = "logs/step2_preprocessing.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

tqdm.pandas()

# ─────────────────────────────────────────────
# KAMUS NORMALISASI BAHASA INFORMAL (Banking Domain)
# ─────────────────────────────────────────────
NORMALIZATION_DICT = {

    # =========================
    # NEGASI (SANGAT KRITIS)
    # =========================
    "gak": "tidak", "ga": "tidak", "g": "tidak",
    "nggak": "tidak", "ngga": "tidak", "gk": "tidak",
    "tdk": "tidak", "tak": "tidak",
    "engga": "tidak", "enggak": "tidak",
    "kagak": "tidak",
    "gakkk": "tidak", "gakk": "tidak", "gaaa": "tidak",

    # Frasa negasi penting
    "gak bisa": "tidak bisa", "ga bisa": "tidak bisa",
    "nggak bisa": "tidak bisa", "gabisa": "tidak bisa",
    "gbs": "tidak bisa", "gakbisa": "tidak bisa",
    "tdk bisa": "tidak bisa",
    "g bisa": "tidak bisa",

    # =========================
    # KATA UMUM (SINGKATAN)
    # =========================
    "yg": "yang", "dgn": "dengan", "dg": "dengan",
    "utk": "untuk", "krn": "karena", "karna": "karena",
    "jg": "juga",
    "sdh": "sudah", "udh": "sudah", "udah": "sudah",
    "blm": "belum", "belom": "belum",
    "msh": "masih", "masi": "masih",
    "lg": "lagi", "lgi": "lagi",
    "bs": "bisa", "bsa": "bisa",
    "bgt": "banget",
    "bkn": "bukan",
    "hrs": "harus",
    "sm": "sama",
    "pd": "pada",
    "dr": "dari",
    "tp": "tapi", "tpi": "tapi",
    "jd": "jadi",
    "klo": "kalau", "kl": "kalau", "kalo": "kalau",
    "kyk": "seperti", "kayak": "seperti", "kaya": "seperti",
    "aja": "saja", "aj": "saja",
    "biar": "agar",

    # =========================
    # PRONOMINA
    # =========================
    "sy": "saya", "sya": "saya",
    "gw": "saya", "gue": "saya", "gua": "saya",
    "aku": "saya",
    "km": "kamu", "kmu": "kamu", "lu": "kamu", "loe": "kamu",
    "org": "orang",

    # =========================
    # TYPO UMUM
    # =========================
    "aplkasi": "aplikasi", "apliksi": "aplikasi",
    "apl": "aplikasi", "app": "aplikasi", "apps": "aplikasi",
    "bener": "benar", "bnr": "benar",
    "trs": "terus", "trus": "terus",
    "smua": "semua",
    "mngkin": "mungkin",
    "wktu": "waktu",

    # =========================
    # DOMAIN PERBANKAN
    # =========================
    "rek": "rekening",
    "tf": "transfer", "trf": "transfer",
    "trx": "transaksi", "txn": "transaksi",
    "notif": "notifikasi",
    "verif": "verifikasi",
    "login": "login",     # jangan ubah ke "masuk" (biar konsisten domain)
    "logout": "logout",
    "otp": "otp",
    "pin": "pin",
    "qris": "qris",
    "kpr": "kpr",

    # =========================
    # CODE-MIXING (MINIMAL & AMAN)
    # =========================
    "failed": "gagal",
    "success": "berhasil",
    "retry": "coba lagi",
    "error": "error",
    "invalid": "tidak valid",
    "update": "perbarui",
    "upgrade": "perbarui",
    "download": "unduh",

    # =========================
    # MASALAH TEKNIS (JANGAN UBAH MAKNA)
    # =========================
    "crash": "crash",
    "force close": "crash",
    "fc": "crash",
    "freeze": "freeze",
    "hang": "hang",
    "lag": "lambat",
    "lemot": "lambat",
    "lelet": "lambat",
    "loading": "loading",
    "stuck": "stuck",

    # =========================
    # FILLER (AMAN DIHAPUS)
    # =========================
    "deh": "", "sih": "", "loh": "", "nih": "",
    "dong": "", "kok": "",
    "wkwk": "", "wk": "", "haha": "", "hehe": "", "hihi": ""
}

# ─────────────────────────────────────────────
# KELAS PREPROCESSOR
# ─────────────────────────────────────────────
class IndonesianTextPreprocessor:
    """
    Pipeline preprocessing teks bahasa Indonesia untuk domain ulasan
    aplikasi perbankan. Dirancang khusus untuk menangani:
    - Code-mixing (Indonesia-Inggris)
    - Bahasa informal dan slang
    - Noise linguistik (emoji, simbol, typo)
    """

    def __init__(self, normalization_dict: dict = NORMALIZATION_DICT):
        self.norm_dict = normalization_dict
        self._init_stemmer()

    def _init_stemmer(self):
        """Inisialisasi stemmer Sastrawi untuk bahasa Indonesia."""
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
            logger.info("Sastrawi stemmer berhasil diinisialisasi.")
        except ImportError:
            self.stemmer = None
            logger.warning("Sastrawi tidak ditemukan. Stemming dinonaktifkan.")

    # ── Sub-proses individual ──────────────────

    @staticmethod
    def remove_emoji(text: str) -> str:
        """Hapus semua emoji dari teks."""
        try:
            import emoji
            return emoji.replace_emoji(text, replace="")
        except ImportError:
            # Fallback dengan regex Unicode range
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"
                "\U0001F300-\U0001F5FF"
                "\U0001F680-\U0001F9FF"
                "\U00002600-\U000027BF"
                "\U0001F1E0-\U0001F1FF"
                "]+", flags=re.UNICODE
            )
            return emoji_pattern.sub("", text)

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalisasi karakter Unicode ke bentuk NFC."""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def remove_urls(text: str) -> str:
        """Hapus URL."""
        return re.sub(r"https?://\S+|www\.\S+", "", text)

    @staticmethod
    def remove_mentions_hashtags(text: str) -> str:
        """Hapus mention (@) dan hashtag (#)."""
        return re.sub(r"[@#]\w+", "", text)

    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Hapus tag HTML jika ada."""
        return re.sub(r"<[^>]+>", "", text)

    @staticmethod
    def normalize_repeated_chars(text: str) -> str:
        """
        Normalisasi pengulangan karakter berlebih, contoh:
        'bagusssss' → 'bagus', 'loooong' → 'long'
        """
        return re.sub(r"(.)\1{2,}", r"\1\1", text)

    @staticmethod
    def normalize_repeated_punctuation(text: str) -> str:
        """Normalisasi tanda baca berulang: '!!!' → '!'"""
        return re.sub(r"([!?.]){2,}", r"\1", text)

    @staticmethod
    def remove_special_chars(text: str) -> str:
        """
        Hapus karakter khusus kecuali huruf, angka, dan tanda baca dasar.
        Pertahankan tanda baca penting untuk konteks sentimen.
        """
        return re.sub(r"[^\w\s.,!?;:()\-']", " ", text)

    def normalize_slang(self, text: str) -> str:
        """
        Normalisasi kata slang menggunakan kamus normalisasi.
        Proses: tokenisasi sederhana per kata → lookup → rekonstruksi.
        """
        words = text.split()
        normalized = [
            self.norm_dict.get(w.lower(), w) for w in words
        ]
        # Filter token kosong hasil normalisasi filler words
        normalized = [w for w in normalized if w.strip()]
        return " ".join(normalized)

    def stem_text(self, text: str) -> str:
        """
        Stemming menggunakan Sastrawi (opsional).
        CATATAN: Untuk fine-tuning IndoBERT, stemming umumnya TIDAK disarankan
        karena merusak konteks subword tokenization. Aktifkan hanya untuk
        analisis statistik (TF-IDF, frequency analysis).
        """
        if self.stemmer:
            return self.stemmer.stem(text)
        return text

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalisasi whitespace berlebih."""
        return re.sub(r"\s+", " ", text).strip()

    # ── Pipeline lengkap ──────────────────────

    def preprocess(
        self,
        text: str,
        apply_stemming: bool = False,  # Nonaktif untuk IndoBERT
        lowercase: bool = True
    ) -> str:
        """
        Pipeline preprocessing lengkap.

        Args:
            text          : teks mentah
            apply_stemming: aktifkan stemming (gunakan False untuk IndoBERT)
            lowercase     : konversi ke huruf kecil

        Returns:
            Teks yang sudah dibersihkan
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        text = self.normalize_unicode(text)
        text = self.remove_html_tags(text)
        text = self.remove_emoji(text)
        text = self.remove_urls(text)
        text = self.remove_mentions_hashtags(text)
        text = self.normalize_repeated_chars(text)
        text = self.normalize_repeated_punctuation(text)

        if lowercase:
            text = text.lower()

        text = self.normalize_slang(text)
        text = self.remove_special_chars(text)

        if apply_stemming:
            text = self.stem_text(text)

        text = self.normalize_whitespace(text)
        return text

    def preprocess_for_bert(self, text: str) -> str:
        """
        Preprocessing ringan khusus untuk input IndoBERT.
        Hindari stemming dan penghapusan kata-kata fungsional
        karena BERT memerlukan konteks kalimat penuh.
        """
        return self.preprocess(text, apply_stemming=False, lowercase=True)

    def preprocess_for_analysis(self, text: str) -> str:
        """
        Preprocessing lebih agresif untuk analisis statistik dan
        pembangunan kamus aspek.
        """
        return self.preprocess(text, apply_stemming=True, lowercase=True)


# ─────────────────────────────────────────────
# FUNGSI DETEKSI BAHASA
# ─────────────────────────────────────────────
def detect_language_safe(text: str) -> str:
    """Deteksi bahasa dengan penanganan error."""
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────
# FUNGSI STATISTIK TEKS
# ─────────────────────────────────────────────
def compute_text_statistics(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Menambahkan kolom statistik teks untuk analisis eksplorasi."""
    df["char_count"]    = df[text_col].str.len()
    df["word_count"]    = df[text_col].str.split().str.len()
    df["sent_count"]    = df[text_col].str.count(r"[.!?]+")
    return df


# ─────────────────────────────────────────────
# PIPELINE UTAMA
# ─────────────────────────────────────────────
def run_preprocessing_pipeline(
    input_path: str  = INPUT_PATH,
    output_path: str = OUTPUT_PATH
) -> pd.DataFrame:

    # 1. Load data
    logger.info(f"Membaca data dari: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    logger.info(f"Jumlah baris awal: {len(df):,}")

    preprocessor = IndonesianTextPreprocessor()

    # 2. Preprocessing untuk IndoBERT (tidak di-stem)
    logger.info("Preprocessing teks untuk IndoBERT ...")
    df["clean_text"] = df["review_text"].progress_apply(
        preprocessor.preprocess_for_bert
    )

    # 3. Preprocessing agresif untuk analisis statistik
    logger.info("Preprocessing teks untuk analisis statistik ...")
    df["clean_text_stem"] = df["review_text"].progress_apply(
        preprocessor.preprocess_for_analysis
    )

    # 4. Deteksi bahasa
    logger.info("Mendeteksi bahasa ulasan ...")
    df["lang"] = df["clean_text"].progress_apply(detect_language_safe)
    logger.info(f"Distribusi bahasa:\n{df['lang'].value_counts()}")

    # 5. Tambahkan statistik teks
    df = compute_text_statistics(df, "clean_text")

    # 6. Filter: hapus teks terlalu pendek setelah preprocessing
    before = len(df)
    df = df[df["word_count"] >= 3]
    logger.info(f"Dihapus karena teks terlalu pendek: {before - len(df):,} baris")

    # 7. Simpan
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Data preprocessed disimpan ke: {output_path}")

    # 8. Ringkasan
    logger.info(
        f"\n{'='*50}\n"
        f"RINGKASAN PREPROCESSING\n"
        f"{'='*50}\n"
        f"Total ulasan final     : {len(df):,}\n"
        f"Rata-rata kata/ulasan  : {df['word_count'].mean():.1f}\n"
        f"Median kata/ulasan     : {df['word_count'].median():.0f}\n"
        f"{'='*50}"
    )
    return df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = run_preprocessing_pipeline()

    print("\n" + "="*60)
    print("✅ STEP 2 SELESAI")
    print(f"   Total ulasan diproses  : {len(df):,}")
    print(f"   Output file            : {OUTPUT_PATH}")
    print(f"   Kolom yang dihasilkan  : {list(df.columns)}")
    print("="*60)

    # Tampilkan 5 contoh hasil preprocessing
    sample = df[["review_text", "clean_text", "rating", "word_count"]].sample(5)
    print("\n📋 Contoh Hasil Preprocessing:")
    print(sample.to_string(index=False))
