#!/usr/bin/env python3
"""
r2bert_asap.py

R2BERT reproduction for ASAP (Kaggle ASAP AES dataset).

Paper essentials implemented:
- Representation r = h_[CLS] from BERT last_hidden_state.  (R2BERT Sec 3.3)  :contentReference[oaicite:7]{index=7}
- Score mapping: FCNN(r)=Wr+b, then s = sigmoid(FCNN(r)) in [0,1].          :contentReference[oaicite:8]{index=8}
- Bias init: mean score of training essays used to init bias b.             :contentReference[oaicite:9]{index=9}
- Losses:
  - Regression loss Lm = MSE(pred, gold).                                   :contentReference[oaicite:10]{index=10}
  - Batch-wise ListNet ranking loss Lr (rank within batch).                 :contentReference[oaicite:11]{index=11}
  - Dynamic combination: L = τe*Lm + (1-τe)*Lr, τe schedule per epoch.       :contentReference[oaicite:12]{index=12}

This script trains one model PER essay_set (prompt) and reports per-prompt QWK + average QWK.
By default it uses random 60/20/20 split per prompt (train/val/test), and optional CV folds.

Notes:
- QWK is computed on DENORMALIZED, ROUNDED integer predictions (prompt-specific range).
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel


# -------------------------
# Repro / device
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------
# MinMax helpers (per prompt)
# -------------------------
def minmax_norm(y: float, y_min: float, y_max: float) -> float:
    denom = (y_max - y_min)
    if denom <= 1e-12:
        return 0.0
    return float((y - y_min) / denom)


def minmax_unnorm(y_norm: float, y_min: float, y_max: float) -> float:
    return float(y_norm * (y_max - y_min) + y_min)


# -------------------------
# Splits: 60/20/20 per prompt, optional folds
# -------------------------
def split_indices_60_20_20(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(np.floor(0.60 * n))
    n_val = int(np.floor(0.20 * n))
    n_test = n - n_train - n_val

    # safety
    n_train = max(2, min(n - 2, n_train))
    n_val = max(1, min(n - n_train - 1, n_val))
    n_test = n - n_train - n_val
    if n_test <= 0:
        # force at least 1 test
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train -= 1

    tr = perm[:n_train]
    va = perm[n_train:n_train + n_val]
    te = perm[n_train + n_val:]
    return tr, va, te


# -------------------------
# Dataset
# -------------------------
class ASAPPromptDataset(Dataset):
    def __init__(self, texts: List[str], y_norm: np.ndarray, essay_ids: np.ndarray):
        self.texts = [str(t) for t in texts]
        self.y = np.asarray(y_norm, dtype=np.float32)
        self.essay_ids = np.asarray(essay_ids, dtype=np.int64)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        return {
            "text": self.texts[idx],
            "y": float(self.y[idx]),
            "essay_id": int(self.essay_ids[idx]),
        }


def make_loader(ds: Dataset, tokenizer, batch_size: int, max_length: int, shuffle: bool, num_workers: int):
    def collate(batch):
        texts = [b["text"] for b in batch]
        y = torch.tensor([b["y"] for b in batch], dtype=torch.float32)
        essay_ids = torch.tensor([b["essay_id"] for b in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc.get("token_type_ids", None),
            "y": y,
            "essay_ids": essay_ids,
        }

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate,
    )


# -------------------------
# QWK
# -------------------------
def quadratic_weighted_kappa(y_true, y_pred, min_rating=None, max_rating=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if min_rating is None:
        min_rating = int(min(y_true.min(), y_pred.min()))
    if max_rating is None:
        max_rating = int(max(y_true.max(), y_pred.max()))

    K = max_rating - min_rating + 1
    if K <= 1:
        return np.nan

    O = np.zeros((K, K), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        O[a - min_rating, b - min_rating] += 1.0

    N = O.sum()
    if N == 0:
        return np.nan

    hist_true = O.sum(axis=1)
    hist_pred = O.sum(axis=0)
    E = np.outer(hist_true, hist_pred) / N

    W = np.zeros((K, K), dtype=np.float64)
    denom = (K - 1) ** 2
    for i in range(K):
        for j in range(K):
            W[i, j] = ((i - j) ** 2) / denom

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - (num / den if den > 0 else np.nan)


# -------------------------
# R2BERT model
# -------------------------
class R2BERT(nn.Module):
    def __init__(self, model_name: str, bias_init: float | None = None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        h = int(self.bert.config.hidden_size)
        self.score_head = nn.Linear(h, 1)

        # Initialize bias (paper: "mean score ... used to initialize bias b") :contentReference[oaicite:13]{index=13}
        if bias_init is not None:
            with torch.no_grad():
                self.score_head.bias.fill_(float(bias_init))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        cls = out.last_hidden_state[:, 0, :]  # r = h[CLS] :contentReference[oaicite:14]{index=14}
        logits = self.score_head(cls).squeeze(-1)
        scores = torch.sigmoid(logits)        # s in [0,1] :contentReference[oaicite:15]{index=15}
        return scores, logits


# -------------------------
# Losses: regression + batchwise listnet + dynamic weights
# -------------------------
def compute_gamma_for_tau1(E: int, tau1_target: float = 1e-6) -> float:
    """
    τe = 1 / (1 + exp(γ(E/2 - e)))  and choose γ s.t. τ1 = 1e-6. :contentReference[oaicite:16]{index=16}
    """
    mid = E / 2.0
    denom = (mid - 1.0)
    if abs(denom) < 1e-9:
        return 1.0
    rhs = (1.0 / tau1_target) - 1.0
    rhs = max(rhs, 1.0)
    return float(math.log(rhs) / denom)


def tau_e(e: int, E: int, gamma: float) -> float:
    return float(1.0 / (1.0 + math.exp(gamma * (E / 2.0 - float(e)))))


def batchwise_listnet_loss(y_true_norm: torch.Tensor, y_pred_norm: torch.Tensor) -> torch.Tensor:
    """
    Batch-wise ListNet (top-one probability):
    P(j) = exp(s_j)/sum_k exp(s_k)  (Φ(s)=exp(s))  :contentReference[oaicite:17]{index=17}
    Lr = CE(P_true, P_pred).                                :contentReference[oaicite:18]{index=18}

    Here we compute distribution over the *batch dimension* (rank within batch). :contentReference[oaicite:19]{index=19}
    """
    eps = 1e-12
    p_true = torch.softmax(y_true_norm, dim=0)
    p_pred = torch.softmax(y_pred_norm, dim=0)
    loss = -(p_true * torch.log(p_pred + eps)).sum()
    # normalize to keep scale stable as batch size changes
    return loss / y_true_norm.numel()


# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def eval_on_loader(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool):
    model.eval()
    all_pred = []
    all_true = []
    all_ids = []

    autocast = torch.amp.autocast
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        tti = batch["token_type_ids"]
        if tti is not None:
            tti = tti.to(device)

        y = batch["y"].to(device)
        essay_ids = batch["essay_ids"].cpu().numpy()

        with autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
            pred, _ = model(input_ids, attn, tti)

        all_pred.append(pred.detach().cpu().numpy())
        all_true.append(y.detach().cpu().numpy())
        all_ids.append(essay_ids)

    return (
        np.concatenate(all_pred),
        np.concatenate(all_true),
        np.concatenate(all_ids),
    )


def train_one_prompt(
    df_p: pd.DataFrame,
    prompt_id: int,
    model_name: str,
    num_epochs: int,
    lr: float,
    batch_size: int,
    max_length: int,
    ridge_unused: float,
    amp: bool,
    seed: int,
    cv_folds: int,
    out_dir: str,
):
    # prompt range from data
    y_min = float(df_p["domain1_score"].min())
    y_max = float(df_p["domain1_score"].max())
    min_r = int(y_min)
    max_r = int(y_max)

    texts = df_p["essay"].astype(str).tolist()
    y = df_p["domain1_score"].astype(float).to_numpy()
    essay_ids = df_p["essay_id"].astype(int).to_numpy()

    # normalize into [0,1]
    y_norm = np.array([minmax_norm(v, y_min, y_max) for v in y], dtype=np.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    fold_rows = []
    fold_qwks = []

    num_workers = 2 if torch.cuda.is_available() else 0
    gamma = compute_gamma_for_tau1(num_epochs, tau1_target=1e-6)  # τ1=1e-6 :contentReference[oaicite:20]{index=20}

    for fold in range(cv_folds):
        tr_idx, va_idx, te_idx = split_indices_60_20_20(len(texts), seed=seed + 1000 * prompt_id + fold)

        y_tr = y_norm[tr_idx]
        bias_init = float(np.mean(y_tr))  # paper says mean training score used for bias init :contentReference[oaicite:21]{index=21}

        model = R2BERT(model_name, bias_init=bias_init).to(get_device())
        device = next(model.parameters()).device

        optim = torch.optim.AdamW(model.parameters(), lr=lr)
        scaler = torch.amp.GradScaler(enabled=(amp and device.type == "cuda"))

        ds_tr = ASAPPromptDataset([texts[i] for i in tr_idx], y_norm[tr_idx], essay_ids[tr_idx])
        ds_va = ASAPPromptDataset([texts[i] for i in va_idx], y_norm[va_idx], essay_ids[va_idx])
        ds_te = ASAPPromptDataset([texts[i] for i in te_idx], y_norm[te_idx], essay_ids[te_idx])

        dl_tr = make_loader(ds_tr, tokenizer, batch_size, max_length, shuffle=True,  num_workers=num_workers)
        dl_va = make_loader(ds_va, tokenizer, batch_size, max_length, shuffle=False, num_workers=num_workers)
        dl_te = make_loader(ds_te, tokenizer, batch_size, max_length, shuffle=False, num_workers=num_workers)

        best_state = None
        best_val_qwk = -1.0

        print(f"\n[Prompt {prompt_id} | Fold {fold+1}/{cv_folds}] Train/Val/Test = {len(ds_tr)}/{len(ds_va)}/{len(ds_te)}")
        print(f"[Prompt {prompt_id} | Fold {fold+1}] score range = {min_r}-{max_r} | gamma={gamma:.6f}")

        autocast = torch.amp.autocast

        for e in range(1, num_epochs + 1):
            model.train()
            tau = tau_e(e, num_epochs, gamma)  # dynamic weight :contentReference[oaicite:22]{index=22}

            total_loss = 0.0
            steps = 0

            for batch in dl_tr:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                tti = batch["token_type_ids"]
                if tti is not None:
                    tti = tti.to(device)
                yb = batch["y"].to(device)

                optim.zero_grad(set_to_none=True)

                with autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
                    pred, _ = model(input_ids, attn, tti)

                    # Lm (MSE regression) :contentReference[oaicite:23]{index=23}
                    Lm = F.mse_loss(pred, yb)

                    # Lr (batch-wise ListNet) :contentReference[oaicite:24]{index=24}
                    Lr = batchwise_listnet_loss(yb, pred)

                    # Combined loss L = τe*Lm + (1-τe)*Lr :contentReference[oaicite:25]{index=25}
                    loss = (tau * Lm) + ((1.0 - tau) * Lr)

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                total_loss += float(loss.item())
                steps += 1

            # ---- validation each epoch (paper selects best on val) :contentReference[oaicite:26]{index=26}
            pred_va_norm, true_va_norm, ids_va = eval_on_loader(model, dl_va, device, amp)

            pred_va = np.array([minmax_unnorm(v, y_min, y_max) for v in pred_va_norm], dtype=np.float64)
            true_va = np.array([minmax_unnorm(v, y_min, y_max) for v in true_va_norm], dtype=np.float64)

            pred_va_int = np.clip(np.rint(pred_va).astype(int), min_r, max_r)
            true_va_int = np.clip(np.rint(true_va).astype(int), min_r, max_r)

            val_qwk = quadratic_weighted_kappa(true_va_int, pred_va_int, min_rating=min_r, max_rating=max_r)

            avg_loss = total_loss / max(steps, 1)
            print(f"[Prompt {prompt_id} | Fold {fold+1}] Epoch {e:03d}/{num_epochs} "
                  f"loss={avg_loss:.4f} tau={tau:.6f} val_qwk={val_qwk:.4f}")

            if val_qwk > best_val_qwk:
                best_val_qwk = float(val_qwk)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # ---- load best + test
        if best_state is not None:
            model.load_state_dict(best_state)

        pred_te_norm, true_te_norm, ids_te = eval_on_loader(model, dl_te, device, amp)

        pred_te = np.array([minmax_unnorm(v, y_min, y_max) for v in pred_te_norm], dtype=np.float64)
        true_te = np.array([minmax_unnorm(v, y_min, y_max) for v in true_te_norm], dtype=np.float64)

        pred_te_int = np.clip(np.rint(pred_te).astype(int), min_r, max_r)
        true_te_int = np.clip(np.rint(true_te).astype(int), min_r, max_r)

        test_qwk = quadratic_weighted_kappa(true_te_int, pred_te_int, min_rating=min_r, max_rating=max_r)

        fold_qwks.append(float(test_qwk))

        # save predictions for this fold
        pred_path = os.path.join(out_dir, f"preds_prompt{prompt_id}_fold{fold+1}.csv")
        pd.DataFrame({
            "essay_id": ids_te.astype(int),
            "essay_set": int(prompt_id),
            "true_score": true_te,
            "pred_score": pred_te,
            "pred_rounded": pred_te_int,
        }).to_csv(pred_path, index=False)

        fold_rows.append({
            "essay_set": int(prompt_id),
            "fold": int(fold + 1),
            "train_n": int(len(tr_idx)),
            "val_n": int(len(va_idx)),
            "test_n": int(len(te_idx)),
            "epochs": int(num_epochs),
            "lr": float(lr),
            "batch_size": int(batch_size),
            "max_length": int(max_length),
            "best_val_qwk": float(best_val_qwk),
            "test_qwk": float(test_qwk),
            "score_min": int(min_r),
            "score_max": int(max_r),
        })

        print(f"[Prompt {prompt_id} | Fold {fold+1}] BEST val_qwk={best_val_qwk:.4f} | TEST qwk={test_qwk:.4f}")
        print(f"[Prompt {prompt_id} | Fold {fold+1}] Wrote: {pred_path}")

    prompt_avg = float(np.mean(fold_qwks)) if len(fold_qwks) else float("nan")
    return fold_rows, prompt_avg


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asap_path", required=True, help="e.g., Multi-scale/asap-aes/training_set_rel3.tsv")
    ap.add_argument("--model_name", default="bert-base-uncased")

    ap.add_argument("--num_epochs", type=int, default=30, help="paper used 30 epochs (can increase)")
    ap.add_argument("--lr", type=float, default=4e-5, help="paper found 4e-5 best in their sweep")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)

    ap.add_argument("--cv_folds", type=int, default=1, help="set to 5 for 5-fold CV (slow)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--out_dir", required=True, help="output directory for summaries and predictions")

    args = ap.parse_args()
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.asap_path, sep="\t", encoding="latin-1")
    for c in ["essay_set", "essay", "domain1_score"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if "essay_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["essay_id"] = np.arange(len(df), dtype=np.int64)

    df = df[["essay_id", "essay_set", "essay", "domain1_score"]].dropna().copy()
    df["essay_set"] = df["essay_set"].astype(int)
    df["domain1_score"] = df["domain1_score"].astype(float)

    print("[Info] model_name:", args.model_name)
    print("[Info] epochs:", args.num_epochs, "| lr:", args.lr, "| batch_size:", args.batch_size, "| max_length:", args.max_length)
    print("[Info] cv_folds:", args.cv_folds, "| amp:", bool(args.amp))
    print("[Info] out_dir:", args.out_dir)

    all_fold_rows = []
    prompt_avg_rows = []

    prompt_ids = sorted(df["essay_set"].unique().tolist())

    for pid in prompt_ids:
        df_p = df[df["essay_set"] == pid].reset_index(drop=True)
        fold_rows, prompt_avg = train_one_prompt(
            df_p=df_p,
            prompt_id=int(pid),
            model_name=args.model_name,
            num_epochs=args.num_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_length=args.max_length,
            ridge_unused=0.0,
            amp=args.amp,
            seed=args.seed,
            cv_folds=max(1, int(args.cv_folds)),
            out_dir=args.out_dir,
        )
        all_fold_rows.extend(fold_rows)
        prompt_avg_rows.append({"essay_set": int(pid), "avg_test_qwk_across_folds": float(prompt_avg)})

    summary_path = os.path.join(args.out_dir, "r2bert_summary_folds.csv")
    pd.DataFrame(all_fold_rows).to_csv(summary_path, index=False)

    per_prompt_path = os.path.join(args.out_dir, "r2bert_summary_per_prompt.csv")
    pd.DataFrame(prompt_avg_rows).to_csv(per_prompt_path, index=False)

    avg_qwk = float(pd.DataFrame(prompt_avg_rows)["avg_test_qwk_across_folds"].mean())
    print("\n====================")
    print("[Done] Wrote:", summary_path)
    print("[Done] Wrote:", per_prompt_path)
    print(f"[Done] Average QWK across prompts (mean of per-prompt avg): {avg_qwk:.4f}")


if __name__ == "__main__":
    main()

# Avg QWK = 76.92%