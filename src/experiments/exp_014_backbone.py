"""exp_014 - Phase 1: TCN+Transformer backbone detection benchmark.

Trains the new TemporalBackbone (src/models/backbone.py) on the repaired v3
split for anomaly detection and reports PR-AUC/ROC-AUC vs the TCN/LogReg
baselines. Smoke-run with --epochs 1 for wiring checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.backbone import TemporalBackbone
from src.utils.registry import append

INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
RES = ROOT / "outputs" / "results"


def load_splits():
    X = np.load(INTERIM / "chunks_X.npy")
    scaler = json.loads((PROCESSED / "scaler.json").read_text())
    loc = np.asarray(scaler["loc"])[:16]
    scale = np.asarray(scaler["scale"])[:16]
    Xs = X.copy().astype("float32")
    Xs[:, :, :16] = (Xs[:, :, :16] - loc) / scale
    # Feature engineering: first difference + rolling mean (window 5) per KPI
    K = Xs.shape[2]
    diff = np.zeros_like(Xs)
    diff[:, 1:, :16] = np.diff(Xs[:, :, :16], axis=1)
    roll = np.zeros_like(Xs)
    for t in range(5, Xs.shape[1]):
        roll[:, t, :16] = Xs[:, t-4:t+1, :16].mean(axis=1)
    Xs = np.concatenate([Xs, diff, roll], axis=2)  # (N, T, 18+16+16=50)
    idx = np.load(PROCESSED / "split_indices.npz")
    tar = dict(np.load(PROCESSED / "targets.npz"))
    return Xs, idx, tar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device=torch.device("cpu")
    print("device:", device)

    Xs, idx, tar = load_splits()
    tr, va, te = idx["train"], idx["val"], idx["test"]
    ytr = tar["y_anomaly"][tr].astype("float32")
    yva = tar["y_anomaly"][va].astype("float32")
    yte = tar["y_anomaly"][te].astype("float32")

    Xtr = torch.tensor(Xs[tr], dtype=torch.float32).to(device)
    Xva = torch.tensor(Xs[va], dtype=torch.float32).to(device)
    Xte = torch.tensor(Xs[te], dtype=torch.float32).to(device)
    ytr_t = torch.tensor(ytr).to(device)
    yva_t = torch.tensor(yva).to(device)

    model = TemporalBackbone(n_channels=Xs.shape[2], hidden=32, n_heads=4, n_tcn_layers=2, n_transformer_layers=2, dropout=0.3).to(device)
    head = nn.Linear(32, 1).to(device)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_val=float("inf"); patience=3; bad=0; best_state=None

    def forward(X):
        return head(model(X)).squeeze(-1)

    for ep in range(args.epochs):
        model.train(); head.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        for i in range(0, len(Xtr), args.batch):
            sl = perm[i:i + args.batch]
            xb = Xtr[sl]; yb = ytr_t[sl]
            opt.zero_grad()
            logits = forward(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(sl)
        # val
        model.eval(); head.eval()
        with torch.no_grad():
            vlogits = forward(Xva)
            vloss = float(loss_fn(vlogits, yva_t).item())
        print(f"epoch {ep+1}/{args.epochs} train_loss={total/len(Xtr):.4f} val_loss={vloss:.4f}")
        if vloss < best_val:
            best_val=vloss; bad=0; best_state={k:v.clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
        sched.step()
        if bad >= patience:
            print("early stop"); break

    model.eval(); head.eval()
    with torch.no_grad():
        tlogits = forward(Xte).detach().cpu().numpy()
    prob = 1 / (1 + np.exp(-tlogits))
    pr = float(average_precision_score(yte, prob))
    roc = float(roc_auc_score(yte, prob))
    print(f"TEST PR-AUC={pr:.4f} ROC-AUC={roc:.4f}")

    out = {
        "experiment": "exp_014",
        "phase": 1,
        "seed": args.seed,
        "epochs": args.epochs,
        "metrics": {"pr_auc": pr, "roc_auc": roc, "n_test": int(len(yte))},
    }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"exp_014_backbone_seed{args.seed}.json").write_text(json.dumps(out, indent=2))
    try:
        append({"experiment": "exp_014", "seed": args.seed, "status": "COMPLETE",
                "metrics": out["metrics"], "task": "phase1_backbone"})
    except Exception as e:
        print("registry append skipped:", e)


if __name__ == "__main__":
    main()
