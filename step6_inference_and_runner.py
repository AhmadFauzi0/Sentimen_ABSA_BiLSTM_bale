"""
==============================================================================
STEP 6: INFERENCE PIPELINE & MASTER RUNNER
Title: Transforming User Feedback into Strategic Intelligence:
       An ABSA of balé by BTN Superapp using IndoBERT and E-S-QUAL Framework
==============================================================================
Tujuan  : Pipeline inferensi untuk ulasan baru + skrip master yang menjalankan
          semua step secara berurutan (Step 1 → Step 5)

File ini berisi:
  (a) ABSAPredictor   — kelas inferensi real-time untuk ulasan baru
  (b) run_full_pipeline() — master runner yang memanggil semua step

Dependensi:
    Semua dependensi dari step1–step5
==============================================================================
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/step6_inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
ASPECTS    = ["EFFICIENCY", "SYSTEM_AVAILABILITY", "FULFILLMENT", "PRIVACY"]
SENTIMENTS = {0: "Positif", 1: "Netral", 2: "Negatif"}
MODEL_PATH     = "models/best_model.pt"
TOKENIZER_PATH = "models/tokenizer"


# ─────────────────────────────────────────────
# KELAS INFERENSI
# ─────────────────────────────────────────────
class ABSAPredictor:
    """
    Pipeline inferensi real-time untuk prediksi aspek dan sentimen
    pada ulasan baru menggunakan model IndoBERT yang sudah di-fine-tune.

    Contoh penggunaan:
    >>> predictor = ABSAPredictor()
    >>> results = predictor.predict("Aplikasi ini bagus tapi sering error")
    >>> print(results)
    """

    def __init__(
        self,
        model_path: str     = MODEL_PATH,
        tokenizer_path: str = TOKENIZER_PATH,
        max_length: int     = 128,
        threshold: float    = 0.5,    # Threshold sigmoid untuk deteksi aspek
        device: Optional[str] = None
    ):
        self.max_length = max_length
        self.threshold  = threshold

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"ABSAPredictor menggunakan device: {self.device}")

        # Load model dan tokenizer
        self._load_model(model_path, tokenizer_path)
        self._load_preprocessor()

    def _load_model(self, model_path: str, tokenizer_path: str) -> None:
        """Load model dan tokenizer dari checkpoint."""
        from transformers import AutoTokenizer
        # Import model dari step4 (pastikan file ada di path yang sama)
        try:
            from step4_indobert_finetuning import IndoBERTABSA, TrainingConfig
            cfg = TrainingConfig()

            logger.info(f"Memuat checkpoint dari: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)

            self.model = IndoBERTABSA(
                model_name=cfg.model_name,
                n_aspects=len(ASPECTS),
            ).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()

            logger.info(f"Memuat tokenizer dari: {tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            logger.info("Model dan tokenizer berhasil dimuat.")

        except (ImportError, FileNotFoundError) as e:
            logger.error(f"Gagal memuat model: {e}")
            logger.warning("Mode fallback: menggunakan rule-based predictor saja.")
            self.model     = None
            self.tokenizer = None

    def _load_preprocessor(self) -> None:
        """Load preprocessor dari step2."""
        try:
            from step2_preprocessing import IndonesianTextPreprocessor
            self.preprocessor = IndonesianTextPreprocessor()
        except ImportError:
            self.preprocessor = None
            logger.warning("Preprocessor tidak tersedia, teks tidak akan dibersihkan.")

    def _preprocess(self, text: str) -> str:
        """Bersihkan teks sebelum inferensi."""
        if self.preprocessor:
            return self.preprocessor.preprocess_for_bert(text)
        return text.lower().strip()

    def predict_single(self, text: str) -> Dict:
        """
        Prediksi aspek dan sentimen untuk satu ulasan.

        Returns:
            Dict berisi:
            - text           : teks asli
            - clean_text     : teks setelah preprocessing
            - aspects        : list aspek yang terdeteksi
            - sentiments     : dict {aspek: sentimen} hanya untuk aspek terdeteksi
            - confidence     : dict {aspek: probabilitas} untuk aspek
            - summary        : ringkasan singkat prediksi
        """
        clean = self._preprocess(text)

        # Jika model tersedia, gunakan neural predictor
        if self.model is not None and self.tokenizer is not None:
            encoding = self.tokenizer(
                clean,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids  = encoding["input_ids"].to(self.device)
            att_mask   = encoding["attention_mask"].to(self.device)
            tok_type   = encoding.get("token_type_ids", torch.zeros_like(input_ids)).to(self.device)

            with torch.no_grad():
                asp_logits, snt_logits = self.model(input_ids, att_mask, tok_type)

            asp_probs = torch.sigmoid(asp_logits).squeeze().cpu().numpy()
            snt_preds = torch.argmax(snt_logits, dim=-1).squeeze().cpu().numpy()

            detected_aspects   = []
            sentiments         = {}
            confidence         = {}

            for j, asp in enumerate(ASPECTS):
                prob = float(asp_probs[j])
                confidence[asp] = round(prob, 4)
                if prob >= self.threshold:
                    detected_aspects.append(asp)
                    snt_idx = int(snt_preds[j]) if snt_preds.ndim > 0 else int(snt_preds)
                    sentiments[asp] = SENTIMENTS.get(snt_idx, "Netral")

            if not detected_aspects:
                detected_aspects = ["GENERAL"]

        else:
            # Fallback: rule-based dari step3
            from step3_aspect_labeling import ESQUALAspectLabeler
            labeler          = ESQUALAspectLabeler()
            detected_aspects = labeler.detect_aspects(clean)
            sentiment_label  = labeler.detect_sentiment(clean)
            sentiments  = {asp: sentiment_label for asp in detected_aspects}
            confidence  = {asp: 0.0 for asp in ASPECTS}

        # Ringkasan
        summary_parts = [f"{asp}={sentiments.get(asp, '-')}" for asp in detected_aspects]
        summary = " | ".join(summary_parts)

        return {
            "text"       : text,
            "clean_text" : clean,
            "aspects"    : detected_aspects,
            "sentiments" : sentiments,
            "confidence" : confidence,
            "summary"    : summary,
        }

    def predict_batch(
        self,
        texts: Union[List[str], pd.Series],
        batch_size: int = 64,
    ) -> pd.DataFrame:
        """
        Prediksi batch untuk banyak ulasan sekaligus.
        Lebih efisien dibanding predict_single dalam loop.

        Returns:
            DataFrame dengan kolom prediksi
        """
        from tqdm import tqdm

        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        results = []
        for text in tqdm(texts, desc="Inferensi batch"):
            results.append(self.predict_single(text))

        return pd.DataFrame(results)


# ─────────────────────────────────────────────
# DEMO INFERENSI
# ─────────────────────────────────────────────
def demo_inference() -> None:
    """Demonstrasi inferensi pada contoh ulasan nyata."""
    sample_reviews = [
        "Aplikasi ini sangat mudah digunakan, navigasinya intuitif banget. KPR simulasinya juga lengkap!",
        "Server sering down waktu mau transfer, nyebelin banget. Udah gitu loading-nya lama.",
        "OTP-nya sering tidak masuk ke HP, bikin takut soal keamanan data saya.",
        "Bayar tagihan berhasil tapi notifikasinya tidak keluar. Jadi tidak tahu berhasil apa tidak.",
        "Aplikasi bale by BTN memang keren untuk urusan KPR, tapi sering crash kalau koneksi lemah.",
        "Wah mantap, transfer cepat dan aman. PIN biometriknya juga berfungsi dengan baik.",
        "Kecewa banget, sudah 3 hari tidak bisa login. CS-nya juga susah dihubungi.",
    ]

    print("\n" + "="*70)
    print("🔍 DEMO INFERENSI ABSA — balé by BTN")
    print("="*70)

    predictor = ABSAPredictor()

    for i, review in enumerate(sample_reviews, 1):
        result = predictor.predict_single(review)
        print(f"\n[{i}] {result['text'][:80]}...")
        print(f"    Aspek    : {', '.join(result['aspects'])}")
        print(f"    Sentimen : {result['summary']}")
        conf_str = " | ".join([f"{k}: {v:.2f}" for k, v in result["confidence"].items()])
        print(f"    Keyakinan: {conf_str}")
    print("\n" + "="*70)


# ─────────────────────────────────────────────
# MASTER PIPELINE RUNNER
# ─────────────────────────────────────────────
def run_full_pipeline(
    skip_scraping: bool = False,   # Set True jika data sudah ada
    skip_training: bool = False,   # Set True jika model sudah ada
) -> None:
    """
    Menjalankan seluruh pipeline penelitian secara berurutan:

    Step 1 → Data Collection (scraping)
    Step 2 → Preprocessing
    Step 3 → Aspect Labeling
    Step 4 → IndoBERT Fine-tuning
    Step 5 → Evaluasi & Visualisasi

    Args:
        skip_scraping : lewati step 1 jika data mentah sudah ada
        skip_training : lewati step 4 jika model sudah ada
    """
    import os

    # Buat direktori yang diperlukan
    for d in ["data", "models", "results", "results/figures", "logs"]:
        os.makedirs(d, exist_ok=True)

    steps = [
        ("STEP 1: Data Collection",     "step1_data_collection.py",   skip_scraping),
        ("STEP 2: Preprocessing",        "step2_preprocessing.py",     False),
        ("STEP 3: Aspect Labeling",      "step3_aspect_labeling.py",   False),
        ("STEP 4: IndoBERT Fine-tuning", "step4_indobert_finetuning.py", skip_training),
        ("STEP 5: Evaluasi & Viz",       "step5_evaluation_visualization.py", False),
    ]

    total_start = time.time()
    results_log = []

    for step_name, script, skip in steps:
        if skip:
            logger.info(f"⏭️  MELEWATI {step_name} (skip=True)")
            results_log.append({"step": step_name, "status": "SKIPPED"})
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"▶️  MEMULAI {step_name}")
        logger.info(f"{'='*60}")

        start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, script],
                capture_output=False,
                check=True,
            )
            elapsed = time.time() - start
            logger.info(f"✅ {step_name} SELESAI dalam {elapsed:.1f} detik")
            results_log.append({
                "step"   : step_name,
                "status" : "SUCCESS",
                "elapsed": f"{elapsed:.1f}s"
            })

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start
            logger.error(f"❌ {step_name} GAGAL: {e}")
            results_log.append({
                "step"   : step_name,
                "status" : "FAILED",
                "elapsed": f"{elapsed:.1f}s",
                "error"  : str(e)
            })
            logger.warning("Melanjutkan ke step berikutnya ...")

    total_elapsed = time.time() - total_start

    # Ringkasan
    print("\n" + "="*60)
    print("📋 RINGKASAN EKSEKUSI PIPELINE")
    print("="*60)
    for item in results_log:
        status_icon = "✅" if item["status"] == "SUCCESS" else (
                      "⏭️" if item["status"] == "SKIPPED" else "❌")
        elapsed_str = item.get("elapsed", "-")
        print(f"  {status_icon} {item['step']:<40} [{elapsed_str}]")

    print(f"\n⏱️  Total waktu eksekusi: {total_elapsed/60:.1f} menit")
    print("="*60)

    # Simpan log
    log_path = Path("results/pipeline_execution_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": results_log,
            "total_elapsed_seconds": round(total_elapsed, 2),
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Log eksekusi disimpan ke: {log_path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ABSA Pipeline — balé by BTN Superapp"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "demo", "inference"],
        default="demo",
        help=(
            "full     : jalankan seluruh pipeline (step 1-5)\n"
            "demo     : demo inferensi pada contoh ulasan\n"
            "inference: inferensi pada file CSV baru"
        )
    )
    parser.add_argument(
        "--input",  type=str, default=None,
        help="Path CSV untuk mode inference (kolom 'review_text' diperlukan)"
    )
    parser.add_argument(
        "--output", type=str, default="results/predictions.csv",
        help="Path output untuk mode inference"
    )
    parser.add_argument(
        "--skip_scraping", action="store_true",
        help="Lewati step scraping (gunakan data yang sudah ada)"
    )
    parser.add_argument(
        "--skip_training", action="store_true",
        help="Lewati step fine-tuning (gunakan model yang sudah ada)"
    )

    args = parser.parse_args()

    if args.mode == "full":
        run_full_pipeline(
            skip_scraping=args.skip_scraping,
            skip_training=args.skip_training,
        )

    elif args.mode == "demo":
        demo_inference()

    elif args.mode == "inference":
        if not args.input:
            print("Error: --input diperlukan untuk mode inference")
            sys.exit(1)

        df       = pd.read_csv(args.input, encoding="utf-8-sig")
        texts    = df["review_text"].fillna("").tolist()
        pred     = ABSAPredictor()
        pred_df  = pred.predict_batch(texts)
        pred_df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"\n✅ Hasil prediksi disimpan ke: {args.output}")
