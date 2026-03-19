#!/usr/bin/env python3
"""
r2bert_topkpool_cv.py

R2BERT (Regression + Ranking with dynamic weights) BUT with representation swapped:

Original R2BERT representation (paper):
  r = h[CLS]  (CLS vector)

This script (your requested change):
  r = Top-K avg pooled per dimension over last_hidden_state token vectors
      (padding ignored; CLS token excluded by default)

Pipeline per prompt (essay_set):
- 5-fold CV (60/20/20): for each fold:
    - Train: 60%, Val: 20%, Test: 20%
- Fine-tune BERT with combined loss:
    L = tau(e)*Lm + (1-tau(e))*Lr
    - Lm = MSE on normalized scores in [0,1]
    - Lr = batchwise ListNet (top-1 distribution) CE(P_true || P_pred)
    - tau(e) increases over epochs (dynamic weights)
- Evaluate on TEST by denormalizing back to the prompt’s score range and computing QWK.

Run examples are at the bottom of this file.
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

# sklearn is used only for split helpers; metrics are implemented locally
from sklearn.model_selection import KFold, train_test_split


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
# Score normalization (prompt-specific)
# -------------------------
def minmax_norm(y: float, y_min: float, y_max: float) -> float:
    denom = (y_max - y_min)
    if denom <= 1e-12:
        return 0.0
    return float((y - y_min) / denom)


def minmax_unnorm(y_norm: float, y_min: float, y_max: float) -> float:
    return float(y_norm * (y_max - y_min) + y_min)


# -------------------------
# QWK (quadratic weighted kappa)
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


def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray):
    err = y_pred - y_true
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))


# -------------------------
# Dataset
# -------------------------
class ASAPDataset(Dataset):
    def __init__(self, df: pd.DataFrame, y_min: float, y_max: float):
        self.df = df.reset_index(drop=True)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        y = float(r["domain1_score"])
        y_norm = minmax_norm(y, self.y_min, self.y_max)
        return {
            "essay_id": int(r["essay_id"]),
            "text": str(r["essay"]),
            "y": float(y),
            "y_norm": float(y_norm),
        }


def collate_batch(batch):
    return {
        "essay_id": torch.tensor([b["essay_id"] for b in batch], dtype=torch.long),
        "text": [b["text"] for b in batch],
        "y": torch.tensor([b["y"] for b in batch], dtype=torch.float32),
        "y_norm": torch.tensor([b["y_norm"] for b in batch], dtype=torch.float32),
    }


# -------------------------
# AMP compatibility (new torch.amp vs old torch.cuda.amp)
# -------------------------
def get_amp_components(device: torch.device):
    """
    Returns (autocast_ctx, scaler) compatible across torch versions.
    autocast_ctx is a context manager factory: autocast_ctx(enabled=True).
    """
    if device.type != "cuda":
        class Dummy:
            def __init__(self, enabled=False): pass
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False

        class DummyScaler:
            def __init__(self, enabled=False): self.enabled = False
            def scale(self, x): return x
            def step(self, opt): opt.step()
            def update(self): pass

        return (lambda enabled=False: Dummy(enabled=enabled)), DummyScaler(enabled=False)

    # CUDA
    try:
        from torch.amp import autocast as new_autocast
        from torch.amp import GradScaler as NewGradScaler
        return (lambda enabled=False: new_autocast(device_type="cuda", enabled=enabled)), NewGradScaler(enabled=True)
    except Exception:
        from torch.cuda.amp import autocast as old_autocast
        from torch.cuda.amp import GradScaler as OldGradScaler
        return (lambda enabled=False: old_autocast(enabled=enabled)), OldGradScaler(enabled=True)


# -------------------------
# Top-K avg pooling per dimension (padding ignored, CLS excluded)
# -------------------------
def topk_avg_pool_per_dim(
    last_hidden: torch.Tensor,          # (B,L,H)
    attention_mask: torch.Tensor,       # (B,L) 1=token, 0=pad
    k: int = 10,
    exclude_cls: bool = True,
) -> torch.Tensor:
    """
    For each sample and each hidden dimension j:
      pooled_j = mean(top-K token values along sequence dimension)

    - pads ignored
    - CLS excluded by default
    """
    B, L, H = last_hidden.shape
    k = int(min(k, L))

    # boolean mask for valid positions
    mask = attention_mask.to(dtype=torch.bool)  # (B,L)
    if exclude_cls and L > 0:
        mask = mask.clone()
        mask[:, 0] = False

    # use float32 for stable topk (even under amp)
    x = last_hidden.float()

    # set invalid tokens to a large negative so they don't appear in topk
    very_neg = torch.tensor(-1e9, device=x.device, dtype=x.dtype)
    x = x.masked_fill(~mask.unsqueeze(-1), very_neg)  # (B,L,H)

    vals, _ = torch.topk(x, k=k, dim=1)  # (B,k,H)

    # count how many of the topk values are actually valid (for very short sequences)
    valid = vals > -5e8
    denom = valid.sum(dim=1).clamp(min=1)  # (B,H)
    pooled = (vals * valid).sum(dim=1) / denom  # (B,H)

    return pooled.to(dtype=last_hidden.dtype)


# -------------------------
# Batchwise ListNet (top-1) loss
# -------------------------
def listnet_top1_ce(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Standard ListNet top-1 distribution cross-entropy in a batch:

      P_true = softmax(y_true)
      P_pred = softmax(y_pred)
      L = - sum_j P_true(j) * log(P_pred(j))

    y_true, y_pred are (B,) normalized scores in [0,1].
    """
    p_true = torch.softmax(y_true, dim=0)
    log_p_pred = torch.log_softmax(y_pred, dim=0)
    return -(p_true * log_p_pred).sum()


# -------------------------
# Dynamic weights tau(e)
# -------------------------
def compute_gamma_for_tau1(num_epochs: int, tau1: float = 1e-6) -> float:
    """
    Paper chooses gamma so that tau(1) ~= 1e-6 for their E.
    Solve:
      tau(1) = 1 / (1 + exp(gamma*(E/2 - 1))) = tau1
    """
    E = float(num_epochs)
    denom = (E / 2.0 - 1.0)
    if abs(denom) < 1e-12:
        return 1.0
    rhs = (1.0 / float(tau1)) - 1.0
    return float(math.log(rhs) / denom)


def tau_e(epoch_1based: int, num_epochs: int, gamma: float) -> float:
    E = float(num_epochs)
    e = float(epoch_1based)
    return float(1.0 / (1.0 + math.exp(gamma * (E / 2.0 - e))))


# -------------------------
# Model: BERT + pooling + linear head (+ sigmoid)
# -------------------------
class R2BERTTopKPool(nn.Module):
    def __init__(self, model_name: str, top_k_pool: int = 10, exclude_cls: bool = True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.top_k_pool = int(top_k_pool)
        self.exclude_cls = bool(exclude_cls)

        hidden = int(self.bert.config.hidden_size)
        self.scorer = nn.Linear(hidden, 1)

    def init_bias_to_mean(self, mean_y_norm: float):
        """
        If output is sigmoid(z), to make sigmoid(b)=mean -> b=logit(mean).
        """
        m = float(np.clip(mean_y_norm, 1e-6, 1.0 - 1e-6))
        b = math.log(m / (1.0 - m))
        with torch.no_grad():
            self.scorer.bias.fill_(b)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # (B,L,H)
        r = topk_avg_pool_per_dim(h, attention_mask, k=self.top_k_pool, exclude_cls=self.exclude_cls)  # (B,H)
        z = self.scorer(r).squeeze(-1)  # (B,)
        y_hat = torch.sigmoid(z)        # normalized in [0,1]
        return y_hat


# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int,
    use_amp: bool,
    tau: float,
):
    model.train()
    autocast_ctx, scaler = get_amp_components(device)

    total_loss = 0.0
    steps = 0

    for batch in loader:
        enc = tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        y_true = batch["y_norm"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx(enabled=(use_amp and device.type == "cuda")):
            y_pred = model(input_ids=input_ids, attention_mask=attn_mask)

            lm = F.mse_loss(y_pred, y_true)
            lr = listnet_top1_ce(y_true, y_pred)
            loss = float(tau) * lm + (1.0 - float(tau)) * lr

        if device.type == "cuda" and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        steps += 1

    return total_loss / max(steps, 1)


@torch.no_grad()
def predict_norm(
    model: nn.Module,
    loader: DataLoader,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    autocast_ctx, _ = get_amp_components(device)

    all_ids = []
    all_pred = []

    for batch in loader:
        enc = tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        with autocast_ctx(enabled=(use_amp and device.type == "cuda")):
            y_pred = model(input_ids=input_ids, attention_mask=attn_mask)

        all_ids.append(batch["essay_id"].cpu().numpy())
        all_pred.append(y_pred.detach().cpu().numpy())

    return np.concatenate(all_ids), np.concatenate(all_pred)


def eval_on_split(
    model: nn.Module,
    df_split: pd.DataFrame,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
    use_amp: bool,
    y_min: float,
    y_max: float,
) -> Dict[str, float]:
    ds = ASAPDataset(df_split, y_min=y_min, y_max=y_max)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    ids, pred_norm = predict_norm(model, dl, tokenizer, device, max_length, use_amp)
    pred_norm = np.clip(pred_norm, 0.0, 1.0)

    # align with df order by essay_id
    pred_map = {int(i): float(p) for i, p in zip(ids, pred_norm)}
    y_true = df_split["domain1_score"].to_numpy(np.float64)

    pred_denorm = np.array(
        [minmax_unnorm(pred_map[int(eid)], y_min, y_max) for eid in df_split["essay_id"].to_list()],
        dtype=np.float64,
    )

    min_r = int(y_min)
    max_r = int(y_max)
    pred_int = np.rint(pred_denorm).astype(int)
    pred_int = np.clip(pred_int, min_r, max_r)

    qwk = quadratic_weighted_kappa(np.rint(y_true).astype(int), pred_int, min_rating=min_r, max_rating=max_r)
    mae, rmse = mae_rmse(y_true, pred_denorm)

    return {
        "qwk": float(qwk),
        "mae": float(mae),
        "rmse": float(rmse),
    }


# -------------------------
# CV split (5-fold -> 60/20/20)
# -------------------------
def make_cv_splits(df_prompt: pd.DataFrame, n_splits: int, seed: int):
    """
    For each fold:
      - test = 1 fold (20%)
      - remaining 80% split into train (60%) and val (20%) by val_size=0.25 of remaining
    """
    idx = np.arange(len(df_prompt))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (trainval_idx, test_idx) in enumerate(kf.split(idx), start=1):
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=0.25,  # 0.25 * 0.80 = 0.20
            random_state=seed + fold,
            shuffle=True,
        )
        yield fold, train_idx, val_idx, test_idx


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--asap_path", required=True, help="ASAP training_set_rel3.tsv")
    ap.add_argument("--model_name", default="bert-base-uncased")

    ap.add_argument("--prompt_id", type=int, default=0,
                    help="0 = run all prompts (1..8). Otherwise run a single prompt_id.")
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--num_epochs", type=int, default=30,
                    help="Paper uses 30; you can set 80 if you want.")
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--top_k_pool", type=int, default=10,
                    help="Top-K tokens per hidden dimension for pooling.")
    ap.add_argument("--include_cls_in_pool", action="store_true",
                    help="If set, CLS token is NOT excluded from topk pooling.")

    ap.add_argument("--tau1", type=float, default=1e-6,
                    help="Target tau at epoch 1; gamma is solved from this.")
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--out_dir", required=True, help="Output directory for CSVs")

    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("[Info] Device:", device)
    print(f"[Info] model={args.model_name} | epochs={args.num_epochs} | lr={args.lr} | bs={args.batch_size} | amp={bool(args.amp)}")
    print(f"[Info] pooling=TopKAvgPerDim | top_k_pool={args.top_k_pool} | exclude_cls={not args.include_cls_in_pool}")
    print(f"[Info] CV folds={args.cv_folds} | prompt_id={args.prompt_id} (0=all prompts)")

    os.makedirs(args.out_dir, exist_ok=True)

    # load ASAP
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

    prompts = [args.prompt_id] if args.prompt_id != 0 else sorted(df["essay_set"].unique().tolist())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    gamma = compute_gamma_for_tau1(args.num_epochs, tau1=args.tau1)
    print(f"[Info] dynamic-weights: tau1={args.tau1} -> solved gamma={gamma:.6f}")

    all_summary_rows = []
    all_pred_rows = []

    for pid in prompts:
        dfp = df[df["essay_set"] == pid].copy().reset_index(drop=True)

        y_min = float(dfp["domain1_score"].min())
        y_max = float(dfp["domain1_score"].max())
        min_r = int(y_min)
        max_r = int(y_max)

        print(f"\n==================== Prompt {pid} | range [{min_r},{max_r}] | N={len(dfp)} ====================")

        for fold, train_idx, val_idx, test_idx in make_cv_splits(dfp, n_splits=args.cv_folds, seed=args.seed):
            df_train = dfp.iloc[train_idx].copy().reset_index(drop=True)
            df_val = dfp.iloc[val_idx].copy().reset_index(drop=True)
            df_test = dfp.iloc[test_idx].copy().reset_index(drop=True)

            mean_y_norm = np.mean([minmax_norm(v, y_min, y_max) for v in df_train["domain1_score"].to_list()])

            model = R2BERTTopKPool(
                model_name=args.model_name,
                top_k_pool=args.top_k_pool,
                exclude_cls=(not args.include_cls_in_pool),
            ).to(device)
            model.init_bias_to_mean(mean_y_norm)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            train_ds = ASAPDataset(df_train, y_min=y_min, y_max=y_max)
            val_ds = ASAPDataset(df_val, y_min=y_min, y_max=y_max)
            test_ds = ASAPDataset(df_test, y_min=y_min, y_max=y_max)

            train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
            val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
            test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

            best_val_qwk = -1e9
            best_state = None

            print(f"\n[Prompt {pid} Fold {fold}] train={len(df_train)} val={len(df_val)} test={len(df_test)}")

            for e in range(1, args.num_epochs + 1):
                tau = tau_e(e, args.num_epochs, gamma)
                loss = train_one_epoch(
                    model=model,
                    loader=train_dl,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    device=device,
                    max_length=args.max_length,
                    use_amp=bool(args.amp),
                    tau=tau,
                )

                val_metrics = eval_on_split(
                    model=model,
                    df_split=df_val,
                    tokenizer=tokenizer,
                    device=device,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    use_amp=bool(args.amp),
                    y_min=y_min,
                    y_max=y_max,
                )

                print(f"  [ep {e:02d}/{args.num_epochs}] tau={tau:.6f} loss={loss:.4f} | val_qwk={val_metrics['qwk']:.4f}")

                if val_metrics["qwk"] > best_val_qwk:
                    best_val_qwk = float(val_metrics["qwk"])
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            # load best model by val QWK
            if best_state is not None:
                model.load_state_dict(best_state)

            test_metrics = eval_on_split(
                model=model,
                df_split=df_test,
                tokenizer=tokenizer,
                device=device,
                max_length=args.max_length,
                batch_size=args.batch_size,
                use_amp=bool(args.amp),
                y_min=y_min,
                y_max=y_max,
            )
            print(f"[Prompt {pid} Fold {fold}] TEST | QWK={test_metrics['qwk']:.4f} MAE={test_metrics['mae']:.4f} RMSE={test_metrics['rmse']:.4f}")

            all_summary_rows.append({
                "prompt_id": int(pid),
                "fold": int(fold),
                "train_n": int(len(df_train)),
                "val_n": int(len(df_val)),
                "test_n": int(len(df_test)),
                "range_min": int(min_r),
                "range_max": int(max_r),
                "top_k_pool": int(args.top_k_pool),
                "exclude_cls": bool(not args.include_cls_in_pool),
                "num_epochs": int(args.num_epochs),
                "lr": float(args.lr),
                "tau1": float(args.tau1),
                "gamma": float(gamma),
                "best_val_qwk": float(best_val_qwk),
                "test_qwk": float(test_metrics["qwk"]),
                "test_mae": float(test_metrics["mae"]),
                "test_rmse": float(test_metrics["rmse"]),
            })

            # save per-essay predictions on test
            ids, pred_norm = predict_norm(model, test_dl, tokenizer, device, args.max_length, bool(args.amp))
            pred_norm = np.clip(pred_norm, 0.0, 1.0)
            pred_denorm = np.array([minmax_unnorm(v, y_min, y_max) for v in pred_norm], dtype=np.float64)
            pred_int = np.clip(np.rint(pred_denorm).astype(int), min_r, max_r)

            gold = df_test["domain1_score"].to_numpy(np.float64)
            for eid, g, pn, pdn, pi in zip(ids.tolist(), gold.tolist(), pred_norm.tolist(), pred_denorm.tolist(), pred_int.tolist()):
                all_pred_rows.append({
                    "prompt_id": int(pid),
                    "fold": int(fold),
                    "essay_id": int(eid),
                    "gold_score": float(g),
                    "pred_norm": float(pn),
                    "pred_score": float(pdn),
                    "pred_rounded": int(pi),
                })

    # write outputs
    summary_path = os.path.join(args.out_dir, "r2bert_topkpool_cv_summary.csv")
    preds_path = os.path.join(args.out_dir, "r2bert_topkpool_cv_test_preds.csv")

    pd.DataFrame(all_summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(all_pred_rows).to_csv(preds_path, index=False)

    # print overall averages
    summ = pd.DataFrame(all_summary_rows)
    if len(summ) > 0:
        avg_by_prompt = summ.groupby("prompt_id")["test_qwk"].mean().reset_index()
        overall_avg = float(avg_by_prompt["test_qwk"].mean()) if len(avg_by_prompt) else float("nan")
        print("\n==================== CV Results ====================")
        print(avg_by_prompt.to_string(index=False))
        print(f"\n[Overall] Average QWK across prompts (mean of prompt means): {overall_avg:.4f}")
        print("====================================================")

    print(f"\n[Saved] {summary_path}")
    print(f"[Saved] {preds_path}")


if __name__ == "__main__":
    main()

"""
Example runs:

1) Run ALL prompts (1..8), 5-fold CV, default 30 epochs (paper-like):
python r2bert_topkpool_cv.py \
  --asap_path Multi-scale/asap-aes/training_set_rel3.tsv \
  --model_name bert-base-uncased \
  --prompt_id 0 \
  --cv_folds 5 \
  --num_epochs 30 \
  --top_k_pool 10 \
  --batch_size 16 \
  --lr 4e-5 \
  --out_dir Multi-scale/out_r2bert_topkpool \
  --amp

2) Run ONLY prompt 7 with 80 epochs:
python r2bert_topkpool_cv.py \
  --asap_path Multi-scale/asap-aes/training_set_rel3.tsv \
  --model_name bert-base-uncased \
  --prompt_id 7 \
  --cv_folds 5 \
  --num_epochs 80 \
  --top_k_pool 10 \
  --batch_size 16 \
  --lr 2e-5 \
  --out_dir Multi-scale/out_r2bert_topkpool_p7_80ep \
  --amp
"""


