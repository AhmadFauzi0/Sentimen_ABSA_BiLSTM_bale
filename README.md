# 🏦 ABSA balé by BTN — Research Pipeline
## Transforming User Feedback into Strategic Intelligence

> **An Aspect-Based Sentiment Analysis of balé by BTN Superapp using IndoBERT and E-S-QUAL Framework**

---

## 📁 Struktur Proyek

```
absa_bale_btn/
│
├── step1_data_collection.py        # Scraping ulasan Google Play Store
├── step2_preprocessing.py          # Pembersihan & normalisasi teks
├── step3_aspect_labeling.py        # Pelabelan aspek E-S-QUAL
├── step4_indobert_finetuning.py    # Fine-tuning IndoBERT untuk ABSA
├── step5_evaluation_visualization.py  # Evaluasi, visualisasi & regresi
├── step6_inference_and_runner.py   # Inferensi real-time & master runner
│
├── requirements.txt
├── README.md
│
├── data/                           # Dataset (dibuat otomatis)
│   ├── raw_reviews.csv
│   ├── preprocessed_reviews.csv
│   ├── labeled_reviews.csv
│   └── annotation_sample.csv
│
├── models/                         # Model tersimpan
│   ├── best_model.pt
│   └── tokenizer/
│
├── results/                        # Output penelitian
│   ├── figures/
│   │   ├── confusion_matrices.png
│   │   ├── sentiment_distribution.png
│   │   ├── wordclouds.png
│   │   ├── training_curves.png
│   │   └── regression_coefficients.png
│   ├── training_metrics.json
│   ├── regression_analysis.csv
│   └── final_report.json
│
└── logs/                           # Log eksekusi
```

---

## ⚡ Quick Start

### 1. Instalasi
```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
pip install -r requirements.txt
```

### 2. Jalankan Demo (tanpa training)
```bash
python step6_inference_and_runner.py --mode demo
```

### 3. Jalankan Pipeline Penuh
```bash
# Jalankan semua step
python step6_inference_and_runner.py --mode full

# Jika data scraping sudah ada
python step6_inference_and_runner.py --mode full --skip_scraping

# Jika model sudah ada (hanya evaluasi)
python step6_inference_and_runner.py --mode full --skip_scraping --skip_training
```

### 4. Inferensi pada Data Baru
```bash
python step6_inference_and_runner.py --mode inference \
    --input data/new_reviews.csv \
    --output results/predictions.csv
```

---

## 🔬 Arsitektur Penelitian

### Dimensi E-S-QUAL yang Dianalisis

| Dimensi | Deskripsi | Contoh Keyword |
|---------|-----------|----------------|
| **EFFICIENCY** | Kemudahan & kecepatan navigasi | mudah, cepat, simpel, UI/UX |
| **SYSTEM_AVAILABILITY** | Ketersediaan & stabilitas sistem | error, down, crash, lemot |
| **FULFILLMENT** | Pemenuhan janji layanan | berhasil, notifikasi, KPR |
| **PRIVACY** | Keamanan & perlindungan data | OTP, PIN, biometrik, aman |

### Arsitektur Model

```
[Ulasan Teks]
     │
     ▼
[Preprocessing]  ← Normalisasi slang, code-mixing, emoji
     │
     ▼
[IndoBERT Encoder]  ← indolem/indobertweet-base-uncased
     │
     ├──────────────────────────┐
     ▼                          ▼
[Aspect Detector]     [Sentiment Classifier]
(Multi-label binary)  (4 aspek × 3 kelas)
     │                          │
     ▼                          ▼
[EFFICIENCY: ✓/✗]    [EFFICIENCY: Positif/Netral/Negatif]
[SYSTEM_AVAIL: ✓/✗]  [SYSTEM_AVAIL: Positif/Netral/Negatif]
[FULFILLMENT: ✓/✗]   [FULFILLMENT: Positif/Netral/Negatif]
[PRIVACY: ✓/✗]       [PRIVACY: Positif/Netral/Negatif]
```

---

## 📊 Output Penelitian

### 1. Metrik Evaluasi
- **Aspect Detection**: F1-score (macro) per kelas aspek
- **Sentiment Classification**: F1-score per aspek per sentimen
- **Confusion Matrix**: visualisasi per dimensi E-S-QUAL

### 2. Analisis Regresi Ridge
Menghitung pengaruh sentimen per aspek E-S-QUAL terhadap rating bintang (1–5):
```
Rating = α + β₁(Efficiency) + β₂(SystemAvail) + β₃(Fulfillment) + β₄(Privacy) + ε
```

### 3. Visualisasi
- Distribusi sentimen per dimensi (stacked bar chart)
- Word cloud per aspek
- Kurva training loss & F1
- Koefisien regresi Ridge

---

## 🛠 Kustomisasi

### Mengubah Model Backbone
Edit `TrainingConfig.model_name` di `step4_indobert_finetuning.py`:
```python
# Opsi yang tersedia:
model_name = "indolem/indobert-base-uncased"       # Default
model_name = "indolem/indobertweet-base-uncased"   # Lebih baik untuk text informal
model_name = "cahya/bert-base-indonesian-1.5G"     # Alternatif
```

### Menambahkan Aspek Baru
Edit `ASPECT_KEYWORDS` di `step3_aspect_labeling.py` dan `CFG.aspects` di `step4`.

### Menangani Class Imbalance (SMOTE)
```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)
```

---

## 📋 Referensi Akademis

**Model yang digunakan:**
- IndoBERT: Koto et al. (2020). *IndoBERT: Pre-trained Language Models for Indonesian*
- IndoBERTweet: Koto et al. (2021). *IndoBERTweet: Pre-trained Language Models for Twitter*

**Kerangka Kualitas:**
- E-S-QUAL: Parasuraman, Zeithaml & Malhotra (2005). *E-S-QUAL: A Multiple-Item Scale*

**Target Publikasi:**
- Information Processing & Management (Q1, Elsevier)
- Journal of Retailing and Consumer Services (Q1, Elsevier)
- Expert Systems with Applications (Q1, Elsevier)

---

## 📌 Catatan Penting

1. **GPU sangat disarankan** untuk Step 4 (training). CPU bisa digunakan tetapi sangat lambat.
2. **Anotasi pakar** diperlukan untuk memvalidasi label di `data/annotation_sample.csv` sebelum training.
3. Model IndoBERT akan **diunduh otomatis** (~500MB) saat pertama kali dijalankan.
4. Pastikan koneksi internet aktif saat menjalankan Step 1 (scraping).

---

*Penelitian ini dirancang untuk publikasi jurnal Q1/Q2 Scopus.*
*Untuk pertanyaan metodologi, merujuk pada dokumen penelitian terlampir.*
