"""
==============================================================================
STEP 4 (REVISI): FINE-TUNING IndoBERTweet + BiLSTM untuk ABSA
Title: Transforming User Feedback into Strategic Intelligence:
       An ABSA of balé by BTN Superapp using IndoBERTweet + BiLSTM

Perubahan dari versi sebelumnya:
  ✦ Backbone : IndoBERT  → IndoBERTweet (lebih baik untuk teks informal)
  ✦ Encoder  : CLS-only  → BiLSTM di atas seluruh token sequence
  ✦ Pooling  : CLS token → Attention Pooling (weighted sum)
  ✦ Hasil    : Representasi konteks sekuensial yang lebih kaya

Keunggulan BiLSTM di atas IndoBERTweet:
  - BERT sudah memahami konteks secara bidirectional via self-attention
  - BiLSTM menangkap dependensi URUTAN dan TEMPORAL dalam kalimat panjang
  - Attention pooling memilih token paling informatif (bukan hanya [CLS])
  - Lebih robust terhadap ulasan panjang (>64 token) seperti keluhan teknis

Input   : data/labeled_reviews.csv
Output  :
  - models/best_model_bilstm.pt
  - results/training_metrics_bilstm.json

Dependensi:
    pip install transformers torch scikit-learn pandas numpy tqdm
==============================================================================
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_cosine_schedule_with_warmup,  # Cosine lebih stabil dari linear
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/step4_bilstm_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
@dataclass
class BiLSTMConfig:
    # ── Backbone ──────────────────────────────
    # IndoBERTweet: dilatih pada 224M token Twitter Indonesia
    # Lebih baik dari IndoBERT untuk teks informal, slang, code-mixing
    model_name: str = "indolem/indobertweet-base-uncased"

    # ── Path ──────────────────────────────────
    input_path: str = "data/labeled_reviews.csv"
    model_dir: str  = "models/"
    result_dir: str = "results/"

    # ── Label ────────────────────────────────
    aspects: List[str] = field(default_factory=lambda: [
        "EFFICIENCY", "SYSTEM_AVAILABILITY", "FULFILLMENT", "PRIVACY"
    ])
    sentiments: List[str] = field(default_factory=lambda: [
        "Positif", "Netral", "Negatif"
    ])
    sentiment2id: Dict[str, int] = field(default_factory=lambda: {
        "Positif": 0, "Netral": 1, "Negatif": 2
    })

    # ── Tokenizer ────────────────────────────
    max_length: int = 128

    # ── BiLSTM Hyperparameter ─────────────────
    bilstm_hidden: int   = 256    # Hidden size per arah (total = 2×256 = 512)
    bilstm_layers: int   = 2      # Jumlah layer stacked BiLSTM
    bilstm_dropout: float = 0.3   # Dropout antar layer BiLSTM

    # ── Training ─────────────────────────────
    batch_size: int      = 32     # Lebih kecil karena BiLSTM tambah memori
    learning_rate_bert: float = 2e-5   # LR untuk layer BERT (lebih kecil)
    learning_rate_bilstm: float = 5e-4 # LR untuk BiLSTM & head (lebih besar)
    num_epochs: int      = 10      # Lebih banyak epoch karena model lebih kompleks
    warmup_ratio: float  = 0.15
    weight_decay: float  = 0.01
    dropout_rate: float  = 0.3
    gradient_clip: float = 1.0

    # ── Regularisasi ekstra untuk BiLSTM ──────
    recurrent_dropout: float = 0.1
    use_layer_norm: bool = True       # LayerNorm setelah BiLSTM

    # ── Split ────────────────────────────────
    train_ratio: float  = 0.70
    val_ratio: float    = 0.15
    test_ratio: float   = 0.15
    random_seed: int    = 42

    # ── Class Weights ─────────────────────────
    use_class_weights: bool = True
    aspect_loss_weight: float    = 1.0
    sentiment_loss_weight: float = 1.5    # Sentimen lebih sulit → bobot lebih besar


CFG = BiLSTMConfig()


# ─────────────────────────────────────────────
# DATASET (sama seperti sebelumnya)
# ─────────────────────────────────────────────
class ABSADataset(Dataset):
    """Dataset ABSA dengan tokenisasi IndoBERTweet."""

    def __init__(
        self,
        texts: List[str],
        aspect_labels: np.ndarray,
        sentiment_labels: np.ndarray,
        tokenizer,
        max_length: int = CFG.max_length,
    ):
        self.texts            = texts
        self.aspect_labels    = aspect_labels
        self.sentiment_labels = sentiment_labels
        self.tokenizer        = tokenizer
        self.max_length       = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids"       : enc["input_ids"].squeeze(0),
            "attention_mask"  : enc["attention_mask"].squeeze(0),
            "token_type_ids"  : enc.get(
                "token_type_ids",
                torch.zeros(self.max_length, dtype=torch.long)
            ).squeeze(0),
            "aspect_labels"   : torch.tensor(self.aspect_labels[idx],    dtype=torch.float),
            "sentiment_labels": torch.tensor(self.sentiment_labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# ATTENTION POOLING MODULE
# ─────────────────────────────────────────────
class AttentionPooling(nn.Module):
    """
    Menggantikan [CLS]-only pooling dengan attention-weighted pooling.

    Cara kerja:
    1. Hitung skor atensi e_i = tanh(W · h_i) · v (scalar per token)
    2. Normalisasi dengan softmax → α_i
    3. Output: c = Σ α_i · h_i  (weighted sum of all BiLSTM hidden states)

    Keunggulan vs CLS:
    - Tidak bergantung pada satu token [CLS] saja
    - Bisa fokus pada token yang paling informatif untuk setiap aspek
    - Lebih robust untuk kalimat panjang yang relevansinya tersebar
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # (batch, seq_len, hidden_dim)
        attention_mask: torch.Tensor, # (batch, seq_len) — padding mask
    ) -> torch.Tensor:
        """
        Returns:
            pooled : (batch, hidden_dim)
        """
        # Hitung skor atensi
        scores = self.attn(hidden_states).squeeze(-1)     # (batch, seq_len)

        # Mask padding agar tidak ikut softmax
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        # Softmax → bobot atensi
        weights = torch.softmax(scores, dim=-1)           # (batch, seq_len)

        # Weighted sum
        pooled = torch.bmm(
            weights.unsqueeze(1),                          # (batch, 1, seq_len)
            hidden_states                                  # (batch, seq_len, hidden_dim)
        ).squeeze(1)                                       # (batch, hidden_dim)

        return pooled


# ─────────────────────────────────────────────
# MODEL UTAMA: IndoBERTweet + BiLSTM
# ─────────────────────────────────────────────
class IndoBERTweetBiLSTMABSA(nn.Module):
    """
    Arsitektur Hybrid untuk ABSA:

    ┌─────────────────────────────────────────┐
    │  IndoBERTweet Encoder (frozen/fine-tune) │
    │  Output: (batch, seq_len, 768)           │
    └──────────────────┬──────────────────────┘
                       │
    ┌──────────────────▼──────────────────────┐
    │  BiLSTM (2 layers, hidden=256×2)        │
    │  Output: (batch, seq_len, 512)          │
    │  + LayerNorm + Dropout                  │
    └──────────────────┬──────────────────────┘
                       │
    ┌──────────────────▼──────────────────────┐
    │  Attention Pooling                       │
    │  Output: (batch, 512)                   │
    └─────────────┬──────────┬────────────────┘
                  │          │
    ┌─────────────▼─┐   ┌───▼────────────────┐
    │ Aspect Head   │   │ Sentiment Heads    │
    │ (4 binary)    │   │ (4 × 3-class)      │
    └───────────────┘   └────────────────────┘

    Strategi fine-tuning:
    - Layer BERT: lr kecil (2e-5) untuk mencegah catastrophic forgetting
    - BiLSTM + Head: lr lebih besar (5e-4) untuk konvergensi cepat
    """

    def __init__(self, cfg: BiLSTMConfig = CFG):
        super().__init__()
        self.cfg = cfg

        n_aspects    = len(cfg.aspects)
        n_sentiments = len(cfg.sentiments)
        bilstm_out   = cfg.bilstm_hidden * 2   # Bidirectional → 2×

        # ── 1. IndoBERTweet Backbone ──────────────────────────
        self.bert = AutoModel.from_pretrained(cfg.model_name)
        bert_hidden = self.bert.config.hidden_size  # 768

        # ── 2. Projection: 768 → bilstm_hidden*2 (opsional, hemat memori) ──
        self.proj = nn.Sequential(
            nn.Linear(bert_hidden, bilstm_out),
            nn.LayerNorm(bilstm_out),
            nn.Dropout(cfg.dropout_rate),
        )

        # ── 3. BiLSTM ─────────────────────────────────────────
        self.bilstm = nn.LSTM(
            input_size=bilstm_out,
            hidden_size=cfg.bilstm_hidden,
            num_layers=cfg.bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.bilstm_dropout if cfg.bilstm_layers > 1 else 0.0,
        )

        # ── 4. LayerNorm + Dropout setelah BiLSTM ─────────────
        if cfg.use_layer_norm:
            self.lstm_norm = nn.LayerNorm(bilstm_out)
        else:
            self.lstm_norm = nn.Identity()

        self.lstm_dropout = nn.Dropout(cfg.dropout_rate)

        # ── 5. Attention Pooling ───────────────────────────────
        self.attention_pool = AttentionPooling(bilstm_out)

        # ── 6. Aspect Detection Head (multi-label) ────────────
        self.aspect_head = nn.Sequential(
            nn.Linear(bilstm_out, bilstm_out // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(bilstm_out // 2, n_aspects),
            # Sigmoid dipakai di loss (BCEWithLogitsLoss)
        )

        # ── 7. Sentiment Heads (1 per aspek) ──────────────────
        # Masing-masing aspek punya head sendiri untuk spesialisasi
        self.sentiment_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(bilstm_out, bilstm_out // 4),
                nn.GELU(),
                nn.Dropout(cfg.dropout_rate),
                nn.Linear(bilstm_out // 4, n_sentiments),
            )
            for _ in range(n_aspects)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 1. IndoBERTweet encoding
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # bert_out.last_hidden_state: (batch, seq_len, 768)
        seq = bert_out.last_hidden_state

        # 2. Projection
        seq = self.proj(seq)          # (batch, seq_len, 512)

        # 3. BiLSTM
        lstm_out, _ = self.bilstm(seq)   # (batch, seq_len, 512)

        # 4. Normalisasi + dropout
        lstm_out = self.lstm_norm(lstm_out)
        lstm_out = self.lstm_dropout(lstm_out)

        # 5. Attention pooling (menggunakan original attention_mask)
        pooled = self.attention_pool(lstm_out, attention_mask)  # (batch, 512)

        # 6. Aspect head
        aspect_logits = self.aspect_head(pooled)         # (batch, 4)

        # 7. Sentiment heads
        sentiment_logits = torch.stack(
            [head(pooled) for head in self.sentiment_heads],
            dim=1
        )  # (batch, 4, 3)

        return aspect_logits, sentiment_logits

    def get_param_groups(self) -> List[Dict]:
        """
        Differential learning rates:
        - BERT parameters    → lr_bert  (kecil, cegah forgetting)
        - BiLSTM + Head      → lr_bilstm (besar, training dari awal)
        """
        bert_params   = list(self.bert.parameters())
        other_params  = (
            list(self.proj.parameters()) +
            list(self.bilstm.parameters()) +
            list(self.lstm_norm.parameters()) +
            list(self.attention_pool.parameters()) +
            list(self.aspect_head.parameters()) +
            [p for head in self.sentiment_heads for p in head.parameters()]
        )
        return [
            {"params": bert_params,  "lr": self.cfg.learning_rate_bert},
            {"params": other_params, "lr": self.cfg.learning_rate_bilstm},
        ]


# ─────────────────────────────────────────────
# LOSS FUNCTION (sama seperti sebelumnya, bobot diperbarui)
# ─────────────────────────────────────────────
class ABSALoss(nn.Module):
    """
    Combined loss:
    L_total = α · L_aspect (BCE) + β · L_sentiment (CE)

    β > α karena klasifikasi sentimen lebih sulit
    """

    def __init__(
        self,
        alpha: float = CFG.aspect_loss_weight,
        beta: float  = CFG.sentiment_loss_weight,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.bce   = nn.BCEWithLogitsLoss()
        self.ce    = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    def forward(
        self,
        asp_logits: torch.Tensor,
        snt_logits: torch.Tensor,
        asp_targets: torch.Tensor,
        snt_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l_asp = self.bce(asp_logits, asp_targets)

        batch, n_asp, n_snt = snt_logits.shape
        l_snt = self.ce(
            snt_logits.view(batch * n_asp, n_snt),
            snt_targets.view(batch * n_asp),
        )
        total = self.alpha * l_asp + self.beta * l_snt
        return total, l_asp, l_snt


# ─────────────────────────────────────────────
# PERSIAPAN DATA
# ─────────────────────────────────────────────
def prepare_labels(df: pd.DataFrame, cfg: BiLSTMConfig = CFG):
    """Konversi kolom teks label ke matriks numerik."""
    n     = len(df)
    n_asp = len(cfg.aspects)

    aspect_mat   = np.zeros((n, n_asp), dtype=np.float32)
    sentiment_mat = np.full((n, n_asp), -1, dtype=np.int64)

    for i, row in df.iterrows():
        row_aspects  = str(row.get("aspects", "GENERAL")).split("|")
        row_sentiment = cfg.sentiment2id.get(str(row.get("sentiment_rule", "Netral")), 1)

        for j, asp in enumerate(cfg.aspects):
            if asp in row_aspects:
                aspect_mat[i, j]    = 1.0
                sentiment_mat[i, j] = row_sentiment

    return aspect_mat, sentiment_mat


# ─────────────────────────────────────────────
# EVALUASI
# ─────────────────────────────────────────────
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: BiLSTMConfig = CFG,
) -> Dict:
    """Evaluasi komprehensif per aspek."""
    model.eval()
    all_ap, all_at, all_sp, all_st = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tok  = batch["token_type_ids"].to(device)

            asp_logits, snt_logits = model(ids, mask, tok)

            asp_preds = (torch.sigmoid(asp_logits) > 0.5).cpu().numpy().astype(int)
            snt_preds = torch.argmax(snt_logits, dim=-1).cpu().numpy()

            all_ap.append(asp_preds)
            all_at.append(batch["aspect_labels"].cpu().numpy().astype(int))
            all_sp.append(snt_preds)
            all_st.append(batch["sentiment_labels"].cpu().numpy())

    ap = np.vstack(all_ap);  at = np.vstack(all_at)
    sp = np.vstack(all_sp);  st = np.vstack(all_st)

    # Aspect F1
    aspect_f1 = f1_score(at, ap, average="macro", zero_division=0)

    # Sentiment F1 per aspek (abaikan -1)
    sent_f1_per = {}
    full_report  = {}
    for j, asp in enumerate(cfg.aspects):
        mask_valid = st[:, j] != -1
        if mask_valid.sum() > 0:
            f1_macro = f1_score(st[mask_valid, j], sp[mask_valid, j],
                                average="macro", zero_division=0)
            sent_f1_per[asp] = round(f1_macro, 4)
            full_report[asp] = classification_report(
                st[mask_valid, j], sp[mask_valid, j],
                target_names=cfg.sentiments, output_dict=True, zero_division=0
            )

    return {
        "aspect_f1_macro"        : round(aspect_f1, 4),
        "sentiment_f1_per_aspect": sent_f1_per,
        "sentiment_f1_avg"       : round(np.mean(list(sent_f1_per.values())), 4),
        "full_classification_report": full_report,
    }


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train(cfg: BiLSTMConfig = CFG) -> None:
    """Pipeline training IndoBERTweet + BiLSTM."""
    for d in [cfg.model_dir, cfg.result_dir, "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        # Aktifkan TF32 untuk Ampere GPUs (lebih cepat)
        torch.backends.cuda.matmul.allow_tf32 = True

    # 1. Load data
    df = pd.read_csv(cfg.input_path, encoding="utf-8-sig")
    df = df.dropna(subset=["clean_text"]).reset_index(drop=True)
    logger.info(f"Total data: {len(df):,}")

    # 2. Label
    aspect_mat, sentiment_mat = prepare_labels(df, cfg)

    # 3. Split
    texts = df["clean_text"].tolist()
    X_tr, X_tmp, ya_tr, ya_tmp, ys_tr, ys_tmp = train_test_split(
        texts, aspect_mat, sentiment_mat,
        test_size=1 - cfg.train_ratio, random_state=cfg.random_seed
    )
    X_val, X_test, ya_val, ya_test, ys_val, ys_test = train_test_split(
        X_tmp, ya_tmp, ys_tmp,
        test_size=cfg.test_ratio / (cfg.val_ratio + cfg.test_ratio),
        random_state=cfg.random_seed
    )
    logger.info(f"Train: {len(X_tr):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # 4. Tokenizer IndoBERTweet
    logger.info(f"Memuat tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # 5. DataLoader
    def make_loader(X, ya, ys, shuffle):
        ds = ABSADataset(X, ya, ys, tokenizer, cfg.max_length)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=2, pin_memory=(device.type == "cuda"))

    train_dl = make_loader(X_tr,   ya_tr,   ys_tr,   True)
    val_dl   = make_loader(X_val,  ya_val,  ys_val,  False)
    test_dl  = make_loader(X_test, ya_test, ys_test, False)

    # 6. Model
    logger.info("Membangun model IndoBERTweet + BiLSTM ...")
    model = IndoBERTweetBiLSTMABSA(cfg).to(device)

    # Hitung parameter
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameter: {total_params:,} | Trainable: {trainable:,}")

    # 7. Class weights
    class_weights = None
    if cfg.use_class_weights:
        flat   = sentiment_mat[sentiment_mat != -1]
        counts = np.bincount(flat, minlength=len(cfg.sentiments)).astype(float)
        w      = 1.0 / (counts + 1e-9); w /= w.sum()
        class_weights = torch.tensor(w, dtype=torch.float).to(device)
        logger.info(f"Class weights: {class_weights.cpu().numpy().round(4)}")

    # 8. Loss + Optimizer (differential LR)
    criterion = ABSALoss(class_weights=class_weights)
    optimizer = AdamW(
        model.get_param_groups(),
        weight_decay=cfg.weight_decay,
    )

    total_steps  = len(train_dl) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 9. Training loop
    best_val_f1 = 0.0
    log_history = []

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{cfg.num_epochs}")
        for batch in pbar:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tok  = batch["token_type_ids"].to(device)
            ya   = batch["aspect_labels"].to(device)
            ys   = batch["sentiment_labels"].to(device)

            optimizer.zero_grad()
            asp_logits, snt_logits = model(ids, mask, tok)
            loss, l_asp, l_snt     = criterion(asp_logits, snt_logits, ya, ys)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(
                total=f"{loss.item():.4f}",
                asp=f"{l_asp.item():.4f}",
                snt=f"{l_snt.item():.4f}",
            )

        avg_loss    = total_loss / len(train_dl)
        val_metrics = evaluate(model, val_dl, device, cfg)

        logger.info(
            f"Epoch {epoch:02d} | Loss={avg_loss:.4f} | "
            f"Aspect F1={val_metrics['aspect_f1_macro']:.4f} | "
            f"Sentiment F1 Avg={val_metrics['sentiment_f1_avg']:.4f}"
        )
        for asp, f1 in val_metrics["sentiment_f1_per_aspect"].items():
            logger.info(f"   {asp:<25} → F1 = {f1:.4f}")

        log_history.append({
            "epoch"                : epoch,
            "train_loss"           : avg_loss,
            "val_aspect_f1_macro"  : val_metrics["aspect_f1_macro"],
            "val_sentiment_f1_avg" : val_metrics["sentiment_f1_avg"],
            **{f"val_f1_{asp}": v for asp, v in val_metrics["sentiment_f1_per_aspect"].items()},
        })

        # Simpan model terbaik
        combined = (val_metrics["aspect_f1_macro"] + val_metrics["sentiment_f1_avg"]) / 2
        if combined > best_val_f1:
            best_val_f1 = combined
            ckpt_path   = Path(cfg.model_dir) / "best_model_bilstm.pt"
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "val_f1"      : combined,
                "config"      : vars(cfg),
            }, ckpt_path)
            tokenizer.save_pretrained(str(Path(cfg.model_dir) / "tokenizer_bilstm"))
            logger.info(f"  → Checkpoint disimpan (combined F1 = {combined:.4f})")

    # 10. Evaluasi final pada test set
    logger.info("\n" + "="*60)
    logger.info("EVALUASI FINAL — TEST SET")
    ckpt = torch.load(Path(cfg.model_dir) / "best_model_bilstm.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_dl, device, cfg)

    logger.info(f"Test Aspect F1 (macro) : {test_metrics['aspect_f1_macro']:.4f}")
    logger.info(f"Test Sentiment F1 (avg): {test_metrics['sentiment_f1_avg']:.4f}")
    for asp, f1 in test_metrics["sentiment_f1_per_aspect"].items():
        logger.info(f"  {asp:<30}: F1 = {f1:.4f}")

    # Cetak classification report lengkap
    for asp, report in test_metrics["full_classification_report"].items():
        logger.info(f"\nClassification Report — {asp}:")
        for label, scores in report.items():
            if isinstance(scores, dict):
                logger.info(
                    f"  {label:<10} P={scores['precision']:.3f} "
                    f"R={scores['recall']:.3f} F1={scores['f1-score']:.3f}"
                )

    # 11. Simpan hasil
    results = {
        "model"         : "IndoBERTweet + BiLSTM",
        "config"        : vars(cfg),
        "training_log"  : log_history,
        "test_metrics"  : {
            k: v for k, v in test_metrics.items()
            if k != "full_classification_report"
        },
        "test_report"   : test_metrics["full_classification_report"],
    }
    out = Path(cfg.result_dir) / "training_metrics_bilstm.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Metrik disimpan ke: {out}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("logs", exist_ok=True)
    train(CFG)

    print("\n" + "="*60)
    print("✅ STEP 4 (BiLSTM) SELESAI")
    print(f"   Model terbaik : models/best_model_bilstm.pt")
    print(f"   Metrik hasil  : results/training_metrics_bilstm.json")
    print("="*60)
