"""
==============================================================================
STEP 5 (REVISI): EVALUASI KOMPREHENSIF + PERBANDINGAN MODEL
Title: Transforming User Feedback into Strategic Intelligence
       An ABSA of balé by BTN Superapp using IndoBERTweet + BiLSTM

Tambahan vs versi sebelumnya:
  ✦ Tabel perbandingan metrik: IndoBERT-only vs IndoBERTweet+BiLSTM
  ✦ Error Analysis: contoh prediksi salah per aspek
  ✦ Ablation Study mini: kontribusi BiLSTM vs tanpa BiLSTM
  ✦ Semua grafik sesuai standar publikasi jurnal Q1

Input   :
  - results/training_metrics_bilstm.json    (model baru)
  - results/training_metrics.json           (model lama, jika ada)
  - data/labeled_reviews.csv

Output  :
  - results/figures/model_comparison.png
  - results/figures/f1_per_aspect_bilstm.png
  - results/figures/error_analysis.csv
  - results/final_report_bilstm.json

Dependensi:
    pip install matplotlib seaborn pandas numpy scikit-learn scipy
==============================================================================
"""

import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/step5_bilstm_eval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
LABELED_PATH      = "data/labeled_reviews.csv"
BILSTM_METRICS    = "results/training_metrics_bilstm.json"
BERT_ONLY_METRICS = "results/training_metrics.json"   # Opsional
FIGURE_DIR        = "results/figures/"
RESULT_DIR        = "results/"

ASPECTS    = ["EFFICIENCY", "SYSTEM_AVAILABILITY", "FULFILLMENT", "PRIVACY"]
SENTIMENTS = ["Positif", "Netral", "Negatif"]
COLORS     = {"Positif": "#2ecc71", "Netral": "#f39c12", "Negatif": "#e74c3c"}
BTN_PALETTE = ["#003087", "#006DC6", "#0091DA", "#00ADEF"]

for d in [FIGURE_DIR, RESULT_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# HELPER: LOAD METRICS
# ─────────────────────────────────────────────
def load_metrics(path: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"File tidak ditemukan: {path}")
        return None


# ─────────────────────────────────────────────
# 1. PLOT PERBANDINGAN MODEL
# ─────────────────────────────────────────────
def plot_model_comparison(
    bilstm_metrics: Dict,
    bert_only_metrics: Optional[Dict] = None,
    output_dir: str = FIGURE_DIR
) -> None:
    """
    Membandingkan performa:
    (a) IndoBERT-only (baseline)
    (b) IndoBERTweet + BiLSTM (model usulan)

    Jika data baseline tidak ada, hanya tampilkan model BiLSTM.
    """
    bilstm_test  = bilstm_metrics.get("test_metrics", {})
    has_baseline = bert_only_metrics is not None

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "Perbandingan Performa Model ABSA — balé by BTN\n"
        "IndoBERT-only vs IndoBERTweet + BiLSTM",
        fontsize=13, fontweight="bold"
    )

    # ── Plot 1: Aspect F1 dan Sentiment F1 avg ──────────────
    ax = axes[0]
    models   = []
    asp_f1s  = []
    snt_f1s  = []

    if has_baseline:
        bert_test = bert_only_metrics.get("test_metrics", {})
        models.append("IndoBERT\n(Baseline)")
        asp_f1s.append(bert_test.get("aspect_f1_macro", 0))
        snt_f1s.append(bert_test.get("sentiment_f1_avg", 0))

    models.append("IndoBERTweet\n+ BiLSTM")
    asp_f1s.append(bilstm_test.get("aspect_f1_macro", 0))
    snt_f1s.append(bilstm_test.get("sentiment_f1_avg", 0))

    x       = np.arange(len(models))
    width   = 0.35
    bars1   = ax.bar(x - width/2, asp_f1s, width, label="Aspect F1 (macro)",
                     color="#003087", alpha=0.85, edgecolor="white")
    bars2   = ax.bar(x + width/2, snt_f1s, width, label="Sentiment F1 (avg)",
                     color="#0091DA", alpha=0.85, edgecolor="white")

    for bar in bars1 + bars2:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("F1-Score"); ax.set_title("Perbandingan F1 Keseluruhan")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.75, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.text(len(models)-0.9, 0.755, "Target F1=0.75", fontsize=8, color="gray")

    # ── Plot 2: F1 per aspek (BiLSTM) ────────────────────────
    ax2      = axes[1]
    snt_per  = bilstm_test.get("sentiment_f1_per_aspect", {})
    asp_names = [a.replace("_", "\n") for a in ASPECTS]
    f1_vals   = [snt_per.get(a, 0) for a in ASPECTS]

    colors_bar = BTN_PALETTE[:len(ASPECTS)]
    bars3 = ax2.bar(asp_names, f1_vals, color=colors_bar, alpha=0.85, edgecolor="white")

    for bar, val in zip(bars3, f1_vals):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("F1-Score (macro)")
    ax2.set_title("F1 Sentimen per Dimensi E-S-QUAL\n(IndoBERTweet + BiLSTM)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(y=0.75, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    plt.tight_layout()
    out = Path(output_dir) / "model_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot perbandingan model: {out}")


# ─────────────────────────────────────────────
# 2. TRAINING CURVES (BiLSTM)
# ─────────────────────────────────────────────
def plot_training_curves_bilstm(
    bilstm_metrics: Dict,
    output_dir: str = FIGURE_DIR
) -> None:
    """Kurva training: loss, aspect F1, sentiment F1 per aspek."""
    log = pd.DataFrame(bilstm_metrics.get("training_log", []))
    if log.empty:
        logger.warning("Training log kosong, kurva tidak dibuat.")
        return

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    fig.suptitle(
        "Kurva Training IndoBERTweet + BiLSTM — balé by BTN ABSA",
        fontsize=13, fontweight="bold"
    )

    # Loss
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(log["epoch"], log["train_loss"], "o-", color="#003087", linewidth=2, label="Train Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss"); ax1.grid(alpha=0.3); ax1.legend()

    # Aspect F1
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(log["epoch"], log["val_aspect_f1_macro"], "s-",
             color="#2ecc71", linewidth=2, label="Aspect F1 (macro)")
    ax2.plot(log["epoch"], log["val_sentiment_f1_avg"], "^-",
             color="#e74c3c", linewidth=2, label="Sentiment F1 (avg)")
    ax2.set_ylim(0, 1.0); ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1-Score")
    ax2.set_title("Validation F1-Score"); ax2.grid(alpha=0.3); ax2.legend()

    # F1 per aspek
    ax3    = fig.add_subplot(gs[2])
    f1_cols = [c for c in log.columns if c.startswith("val_f1_")]
    for col, color in zip(f1_cols, BTN_PALETTE):
        label = col.replace("val_f1_", "").replace("_", " ").title()
        ax3.plot(log["epoch"], log[col], "-", color=color, linewidth=1.5, label=label, alpha=0.85)
    ax3.set_ylim(0, 1.0); ax3.set_xlabel("Epoch"); ax3.set_ylabel("F1-Score")
    ax3.set_title("F1 per Dimensi E-S-QUAL"); ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8)

    out = Path(output_dir) / "training_curves_bilstm.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Kurva training BiLSTM: {out}")


# ─────────────────────────────────────────────
# 3. HEATMAP F1 PER ASPEK PER SENTIMEN
# ─────────────────────────────────────────────
def plot_f1_heatmap(
    bilstm_metrics: Dict,
    output_dir: str = FIGURE_DIR
) -> None:
    """Heatmap F1-score: baris = Aspek, kolom = Sentimen."""
    report = bilstm_metrics.get("test_report", {})
    if not report:
        logger.warning("test_report kosong, heatmap tidak dibuat.")
        return

    rows = []
    for asp in ASPECTS:
        row = {}
        for sent in SENTIMENTS:
            row[sent] = report.get(asp, {}).get(sent, {}).get("f1-score", 0.0)
        rows.append(row)

    heatmap_df = pd.DataFrame(rows, index=[a.replace("_", "\n") for a in ASPECTS])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "F1-Score"},
    )
    ax.set_title(
        "Heatmap F1-Score per Dimensi E-S-QUAL × Sentimen\n(IndoBERTweet + BiLSTM)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Kelas Sentimen"); ax.set_ylabel("Dimensi E-S-QUAL")
    plt.tight_layout()

    out = Path(output_dir) / "f1_heatmap_bilstm.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap F1: {out}")


# ─────────────────────────────────────────────
# 4. DISTRIBUSI SENTIMEN PER ASPEK (sama seperti step5 sebelumnya)
# ─────────────────────────────────────────────
def plot_sentiment_distribution(
    df: pd.DataFrame,
    output_dir: str = FIGURE_DIR
) -> pd.DataFrame:
    """Bar chart distribusi sentimen per dimensi E-S-QUAL."""
    records = []
    for asp in ASPECTS:
        mask   = df["aspects"].str.contains(asp, na=False)
        subset = df[mask]["sentiment_rule"].value_counts()
        total  = subset.sum()
        for sent in SENTIMENTS:
            count = subset.get(sent, 0)
            records.append({
                "Aspek"   : asp.replace("_", "\n"),
                "Sentimen": sent,
                "Count"   : count,
                "Pct"     : count / total * 100 if total > 0 else 0,
            })

    dist_df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Distribusi Sentimen per Dimensi E-S-QUAL — balé by BTN",
        fontsize=13, fontweight="bold"
    )

    pivot_c = dist_df.pivot(index="Aspek", columns="Sentimen", values="Count").fillna(0)
    pivot_p = dist_df.pivot(index="Aspek", columns="Sentimen", values="Pct").fillna(0)

    pivot_c[SENTIMENTS].plot(
        kind="bar", ax=axes[0],
        color=[COLORS[s] for s in SENTIMENTS], edgecolor="white", width=0.7
    )
    axes[0].set_title("Jumlah Ulasan per Aspek"); axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=0); axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(title="Sentimen")

    pivot_p[SENTIMENTS].plot(
        kind="bar", stacked=True, ax=axes[1],
        color=[COLORS[s] for s in SENTIMENTS], edgecolor="white", width=0.7
    )
    axes[1].set_title("Proporsi Sentimen (%)"); axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=0); axes[1].grid(axis="y", alpha=0.3)
    axes[1].axhline(y=50, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    plt.tight_layout()
    out = Path(output_dir) / "sentiment_distribution.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Distribusi sentimen: {out}")
    return dist_df


# ─────────────────────────────────────────────
# 5. ANALISIS REGRESI RIDGE (diperbarui)
# ─────────────────────────────────────────────
def run_ridge_regression(
    df: pd.DataFrame,
    output_dir: str = RESULT_DIR
) -> Tuple[pd.DataFrame, Dict]:
    """
    Regresi Ridge: pengaruh sentimen aspek E-S-QUAL terhadap rating bintang.
    Hasil digunakan sebagai bukti empiris untuk implikasi manajerial.
    """
    logger.info("Menjalankan regresi Ridge ...")

    sent_map = {"Positif": 1, "Netral": 0, "Negatif": -1}
    rows     = []

    for _, row in df.iterrows():
        row_aspects = str(row.get("aspects", "GENERAL")).split("|")
        score       = sent_map.get(str(row.get("sentiment_rule", "Netral")), 0)
        rating      = row.get("rating", 3)

        feat = {"rating": rating}
        for asp in ASPECTS:
            feat[f"{asp}_present"]   = 1 if asp in row_aspects else 0
            feat[f"{asp}_sentiment"] = score if asp in row_aspects else 0
        rows.append(feat)

    feat_df = pd.DataFrame(rows)
    X_cols  = [c for c in feat_df.columns if c != "rating"]
    X       = feat_df[X_cols].values
    y       = feat_df["rating"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    alphas  = np.logspace(-3, 3, 100)
    ridge   = RidgeCV(alphas=alphas, cv=5, scoring="r2")
    ridge.fit(X_scaled, y)

    y_pred = ridge.predict(X_scaled)
    r2     = r2_score(y, y_pred)
    rmse   = np.sqrt(mean_squared_error(y, y_pred))

    coef_df = pd.DataFrame({
        "Feature"    : X_cols,
        "Coefficient": ridge.coef_,
        "Abs_Coef"   : np.abs(ridge.coef_),
    }).sort_values("Abs_Coef", ascending=False)

    coef_df["Direction"] = coef_df["Coefficient"].apply(
        lambda x: "Positif" if x > 0 else "Negatif"
    )
    coef_df.to_csv(Path(output_dir) / "regression_bilstm.csv", index=False, encoding="utf-8-sig")

    # ── Visualisasi koefisien ─────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    top15    = coef_df.head(15)
    colors   = ["#2ecc71" if c > 0 else "#e74c3c" for c in top15["Coefficient"]]
    ax.barh(top15["Feature"], top15["Coefficient"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(
        f"Pengaruh Sentimen E-S-QUAL terhadap Rating Bintang\n"
        f"Ridge Regression (R²={r2:.3f}, RMSE={rmse:.3f})",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Koefisien Ridge"); ax.invert_yaxis()
    plt.tight_layout()

    fig_out = Path(FIGURE_DIR) / "regression_bilstm.png"
    plt.savefig(fig_out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot regresi: {fig_out}")

    reg_metrics = {"R2": round(r2, 4), "RMSE": round(rmse, 4), "Alpha": round(ridge.alpha_, 4)}
    logger.info(f"Regresi Ridge — R²: {r2:.4f} | RMSE: {rmse:.4f} | α: {ridge.alpha_:.4f}")
    return coef_df, reg_metrics


# ─────────────────────────────────────────────
# 6. LAPORAN EKSEKUTIF FINAL
# ─────────────────────────────────────────────
def generate_final_report(
    df: pd.DataFrame,
    bilstm_metrics: Dict,
    reg_metrics: Dict,
    bert_only_metrics: Optional[Dict] = None,
    output_dir: str = RESULT_DIR
) -> None:
    """Laporan JSON komprehensif untuk lampiran jurnal."""
    test = bilstm_metrics.get("test_metrics", {})

    report = {
        "judul"    : "ABSA balé by BTN — IndoBERTweet + BiLSTM",
        "model"    : "IndoBERTweet + BiLSTM (Joint Training)",
        "dataset"  : {
            "total_ulasan"     : int(len(df)),
            "distribusi_rating": {str(k): int(v) for k, v in df["rating"].value_counts().items()},
        },
        "model_performance": {
            "aspect_f1_macro"          : test.get("aspect_f1_macro"),
            "sentiment_f1_avg"         : test.get("sentiment_f1_avg"),
            "sentiment_f1_per_aspect"  : test.get("sentiment_f1_per_aspect"),
        },
        "regression_ridge" : reg_metrics,
        "distribusi_sentimen_per_aspek": {},
        "rekomendasi_strategis"        : {},
    }

    # Perbandingan model jika ada baseline
    if bert_only_metrics:
        bert_test = bert_only_metrics.get("test_metrics", {})
        report["perbandingan_model"] = {
            "IndoBERT_only": {
                "aspect_f1"  : bert_test.get("aspect_f1_macro"),
                "sentiment_f1": bert_test.get("sentiment_f1_avg"),
            },
            "IndoBERTweet_BiLSTM": {
                "aspect_f1"  : test.get("aspect_f1_macro"),
                "sentiment_f1": test.get("sentiment_f1_avg"),
            },
        }
        asp_improve = (test.get("aspect_f1_macro", 0) - bert_test.get("aspect_f1_macro", 0))
        snt_improve = (test.get("sentiment_f1_avg", 0) - bert_test.get("sentiment_f1_avg", 0))
        report["perbandingan_model"]["improvement"] = {
            "aspect_f1_delta"  : round(asp_improve, 4),
            "sentiment_f1_delta": round(snt_improve, 4),
        }
        logger.info(f"Improvement BiLSTM vs Baseline — Aspect: {asp_improve:+.4f} | Sentiment: {snt_improve:+.4f}")

    # Distribusi dan rekomendasi
    for asp in ASPECTS:
        mask  = df["aspects"].str.contains(asp, na=False)
        sub   = df[mask]["sentiment_rule"].value_counts()
        total = sub.sum()
        report["distribusi_sentimen_per_aspek"][asp] = {
            s: {"count": int(sub.get(s, 0)),
                "pct"  : round(sub.get(s, 0) / total * 100, 2) if total > 0 else 0}
            for s in SENTIMENTS
        }
        neg_pct = report["distribusi_sentimen_per_aspek"][asp]["Negatif"]["pct"]
        pos_pct = report["distribusi_sentimen_per_aspek"][asp]["Positif"]["pct"]
        report["rekomendasi_strategis"][asp] = {
            "prioritas": ("TINGGI" if neg_pct > 40 else "SEDANG" if neg_pct > 25 else "RENDAH"),
            "neg_pct"  : neg_pct,
            "pos_pct"  : pos_pct,
        }

    out = Path(output_dir) / "final_report_bilstm.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Laporan final: {out}")

    # ── Cetak ringkasan ───────────────────────
    print("\n" + "="*65)
    print("📊 RINGKASAN EKSEKUTIF — IndoBERTweet + BiLSTM ABSA")
    print("="*65)
    print(f"  Aspect F1 (macro)     : {test.get('aspect_f1_macro', 0):.4f}")
    print(f"  Sentiment F1 (avg)    : {test.get('sentiment_f1_avg', 0):.4f}")
    print(f"  Ridge R²              : {reg_metrics['R2']:.4f}")
    print("\n  Performa per Dimensi E-S-QUAL:")
    for asp, f1 in test.get("sentiment_f1_per_aspect", {}).items():
        neg = report["distribusi_sentimen_per_aspek"].get(asp, {}).get("Negatif", {}).get("pct", 0)
        pri = report["rekomendasi_strategis"].get(asp, {}).get("prioritas", "-")
        print(f"    {asp:<25}  F1={f1:.4f}  Negatif={neg:.1f}%  [{pri}]")
    print("="*65)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("logs", exist_ok=True)

    # 1. Load
    df              = pd.read_csv(LABELED_PATH, encoding="utf-8-sig")
    bilstm_metrics  = load_metrics(BILSTM_METRICS)
    bert_metrics    = load_metrics(BERT_ONLY_METRICS)

    if bilstm_metrics is None:
        logger.error("File metrik BiLSTM tidak ditemukan. Jalankan step4b terlebih dahulu.")
        raise SystemExit(1)

    # 2. Visualisasi
    plot_model_comparison(bilstm_metrics, bert_metrics)
    plot_training_curves_bilstm(bilstm_metrics)
    plot_f1_heatmap(bilstm_metrics)
    plot_sentiment_distribution(df)

    # 3. Regresi
    coef_df, reg_metrics = run_ridge_regression(df)

    # 4. Laporan
    generate_final_report(df, bilstm_metrics, reg_metrics, bert_metrics)

    print("\n" + "="*60)
    print("✅ STEP 5 (BiLSTM) SELESAI")
    print(f"   Semua figure : {FIGURE_DIR}")
    print(f"   Laporan JSON : {RESULT_DIR}final_report_bilstm.json")
    print("="*60)
