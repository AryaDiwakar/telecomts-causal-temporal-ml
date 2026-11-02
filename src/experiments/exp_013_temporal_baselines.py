"""exp_013 - temporal deep baselines (TCN and GRU) on the shared v3 split.

These are *architecture* baselines for the detection task: they see exactly the
same windowed chunks, split, standardization and validation-tuned threshold as
TAGN (exp_002) and the classical baselines (exp_001). The only difference is the
inductive bias (a dilated causal conv stack / a recurrent net) with **no** KPI
graph and **no** hand-built features.

Usage:
    python -m src.experiments.exp_013_temporal_baselines --model tcn --seed 42
    python -m src.experiments.exp_013_temporal_baselines --model gru --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import src.experiments.common as C  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    bootstrap_anomaly_metrics,
    pr_auc,
    pr_f1_at_threshold,
    summarize,
)
from src.utils.registry import append  # noqa: E402

RES = ROOT / "outputs" / "results"
BATCH = 64


class Chomp1d(nn.Module):
    """Trim right padding so a convolution is strictly causal."""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp] if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.GELU(), nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.down(x)


class TCN(nn.Module):
    """Dilated causal temporal conv net with a per-window classification head.

    Pooling concatenates the final-step representation with the window mean; the
    last-step-only variant underfits badly on 128-step windows (val PR ~0.42 vs
    ~0.70 for the GRU), because a single terminal conv tap is a weak summary.
    """

    def __init__(self, in_ch: int, hidden: int = 64, levels: int = 4, dropout: float = 0.1):
        super().__init__()
        blocks, c = [], in_ch
        for i in range(levels):
            blocks.append(TemporalBlock(c, hidden, kernel=3, dilation=2 ** i, dropout=dropout))
            c = hidden
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x):  # x: [B, T, C]
        h = self.blocks(x.transpose(1, 2))  # [B, C, T]
        pooled = torch.cat([h[:, :, -1], h.mean(dim=2)], dim=1)
        return self.head(pooled).squeeze(-1)


class GRUNet(nn.Module):
    """Unidirectional GRU baseline with last + mean pooling."""

    def __init__(self, in_ch: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(in_ch, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x):  # x: [B, T, C]
        out, _ = self.gru(x)  # [B, T, H]
        pooled = torch.cat([out[:, -1], out.mean(dim=1)], dim=1)
        return self.head(self.drop(pooled)).squeeze(-1)


def build(model_name: str, in_ch: int) -> nn.Module:
    return TCN(in_ch) if model_name == "tcn" else GRUNet(in_ch)


def epoch_batches(frames, ratio, rng, batch=BATCH):
    f = frames["train"]
    ano = f["ano"]
    pos = (ano == 1).nonzero(as_tuple=True)[0]
    neg = (ano == 0).nonzero(as_tuple=True)[0]
    if ratio and ratio > 0:
        k = int(ratio * len(pos))
        if k < len(neg):
            sel = torch.from_numpy(rng.choice(len(neg), k, replace=False))
            neg = neg[sel]
    order = torch.cat([pos, neg])[torch.from_numpy(rng.permutation(len(pos) + len(neg)))]
    X = f["X"].index_select(0, order)
    y = f["ano"].index_select(0, order).float()
    for i in range(0, len(X), batch):
        yield X[i:i + batch], y[i:i + batch]


@torch.no_grad()
def predict(model, X, batch=256):
    model.eval()
    out = [model(X[i:i + batch]) for i in range(0, len(X), batch)]
    return torch.cat(out).detach().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tcn", "gru"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--ratio", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    frames, _, _ = C.load_splits()
    idx = C.split_indices()
    groups = C.chunk_groups(idx["test"])
    for split in frames:
        frames[split] = {k: v.to(device) for k, v in frames[split].items()}
    y_val = frames["val"]["ano"].cpu().numpy()
    y_test = frames["test"]["ano"].cpu().numpy()

    model = build(args.model, frames["train"]["X"].shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # Undersampled batches are already rebalanced -> no pos_weight (matches exp_002).
    if args.ratio and args.ratio > 0:
        lossf = nn.BCEWithLogitsLoss()
    else:
        pw = torch.tensor(C.negative_pos_weight(frames), device=device, dtype=torch.float32)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_pr, best_state, history = -1.0, None, []
    best_epoch = 0
    for ep in range(args.epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for Xb, yb in epoch_batches(frames, args.ratio, rng):
            logit = model(Xb)
            loss = lossf(logit, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * len(yb)
            n += len(yb)
        sched.step()
        s_val = predict(model, frames["val"]["X"])
        val_pr = float(pr_auc(y_val, s_val) or 0.0)
        history.append({"epoch": ep, "loss": total / max(1, n), "val_pr_auc": val_pr})
        print(f"  ep {ep:02d} loss {total / max(1, n):7.3f} val PR {val_pr:.4f} "
              f"({time.time() - t0:.0f}s)")
        if val_pr > best_pr:
            best_pr, best_epoch = val_pr, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep - best_epoch > args.patience:
            print("early stop")
            break

    model.load_state_dict(best_state)
    s_val = predict(model, frames["val"]["X"])
    thr = float(pr_f1_at_threshold(y_val, s_val)["threshold"])
    s_test = predict(model, frames["test"]["X"])
    res = summarize(y_test, s_test, threshold=thr)
    res["threshold_val_tuned"] = thr
    boot = bootstrap_anomaly_metrics(y_test, s_test, groups, threshold=thr,
                                     n_boot=args.boot, seed=args.seed)

    tag = f"{args.model}_seed{args.seed}"
    np.savez_compressed(
        RES / f"exp_013_{tag}_preds.npz",
        y_val=y_val, s_val=s_val, y_test=y_test, s_test=s_test, groups=groups,
    )
    out = {"experiment": "exp_013", "model": args.model, "seed": args.seed,
           "device": str(device), "epochs_run": len(history), "best_val_pr_auc": best_pr,
           "history": history, "test": res, "bootstrap": boot}
    (RES / f"exp_013_{tag}.json").write_text(json.dumps(out, indent=2))
    append({
        "experiment": "exp_013", "seed": args.seed, "task": "anomaly_detection",
        "model": f"temporal_{args.model}",
        "metrics": {k: res.get(k) for k in ("pr_auc", "roc_auc", "f1_max", "threshold_val_tuned")},
    })
    print(json.dumps({k: res.get(k) for k in ("pr_auc", "roc_auc", "f1_max", "threshold_val_tuned")},
                     indent=2))
    print(f"PR 95% CI [{boot['pr_auc']['lo']:.4f}, {boot['pr_auc']['hi']:.4f}] "
          f"over {boot['pr_auc']['n_clusters']} recordings")


if __name__ == "__main__":
    main()
