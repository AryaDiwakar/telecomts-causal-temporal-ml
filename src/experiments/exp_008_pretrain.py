"""exp_008 - self-supervised pretraining of the TAGN trunk (Bet 1).

Trains the shared trunk on the *unlabelled training windows only* with
masked-KPI reconstruction, next-step forecasting and NT-Xent contrastive
learning. No anomaly/type/RCA labels are read. Validation/test windows are
never used, so the downstream evaluation stays strictly inductive.

Outputs:
  checkpoints/exp_008/tagn_encoder_pretrained.pt
  outputs/results/exp_008_pretrain.json
  outputs/figures/exp/exp_008_pretrain_curves.png

Usage:
    python -m src.experiments.exp_008_pretrain --epochs 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.pretrain import (  # noqa: E402
    build_pretrainer, pretrained_encoder_path, pretrain_config, resolve_device, save_encoder,
)
from src.models.tagn import load_graphs, model_config  # noqa: E402

INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
RES = ROOT / "outputs" / "results"
FIG = ROOT / "outputs" / "figures" / "exp"


def load_unlabelled_train():
    """Training windows only. Deliberately ignores every label array."""
    import json as _json
    import pandas as pd  # noqa: F401  (kept for parity with other loaders)

    X = np.load(INTERIM / "chunks_X.npy")
    scaler = _json.loads((PROCESSED / "scaler.json").read_text())
    loc = np.asarray(scaler["loc"])[:16]
    scale = np.asarray(scaler["scale"])[:16]
    Xs = X.copy()
    Xs[:, :, :16] = (Xs[:, :, :16] - loc) / scale
    idx = np.load(PROCESSED / "split_indices.npz")
    tr = idx["train"]
    proto = np.load(PROCESSED / "graph_prototypes.npz")["chunk_prototypes"][tr]
    return (
        torch.tensor(Xs[tr], dtype=torch.float32),
        torch.tensor(proto, dtype=torch.long),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    pcfg = pretrain_config()
    if args.epochs:
        pcfg["epochs"] = args.epochs
    if args.batch:
        pcfg["batch_size"] = args.batch
    if args.lr:
        pcfg["lr"] = args.lr
    device = resolve_device(pcfg.get("device", "auto"))
    pcfg["device"] = str(device)

    mc = model_config()
    mc["device"] = device
    prototypes, _ = load_graphs()
    pre = build_pretrainer(mc, prototypes, pcfg, device)
    tagn = pre.tagn

    # Keep the full corpus on host memory. Moving only each mini-batch to the
    # accelerator avoids MPS memory pressure on the 700 MB training tensor.
    X, proto = load_unlabelled_train()
    n = len(X)
    print(f"pretraining on {n} unlabelled train windows on {device}", flush=True)

    params = [p for p in pre.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=pcfg["lr"], weight_decay=pcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=pcfg["epochs"])

    best = float("inf")
    best_state = None
    history = []
    wait = 0
    B = pcfg["batch_size"]

    for ep in range(pcfg["epochs"]):
        pre.train()
        order = torch.from_numpy(rng.permutation(n))
        tot = {"loss": 0.0, "recon": 0.0, "forecast": 0.0, "contrastive": 0.0}
        seen = 0
        for i in range(0, n, B):
            sl = order[i:i + B]
            out = pre(X[sl].to(device), proto[sl].to(device), rng)
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, pcfg["grad_clip"])
            opt.step()
            tot["loss"] += float(out["loss"]) * len(sl)
            for k in ("recon", "forecast", "contrastive"):
                tot[k] += float(out[k]) * len(sl)
            seen += len(sl)
        sched.step()
        rec = {k: v / max(1, seen) for k, v in tot.items()}
        rec["epoch"] = ep
        history.append(rec)
        print(
            f"ep {ep:02d} loss {rec['loss']:.4f} recon {rec['recon']:.4f} "
            f"fc {rec['forecast']:.4f} ctr {rec['contrastive']:.4f} "
            f"lr {sched.get_last_lr()[0]:.2e}",
            flush=True,
        )
        if rec["loss"] < best - 1e-5:
            best = rec["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in tagn.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= pcfg["patience"]:
                print("early stop", flush=True)
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in tagn.state_dict().items()}
    tagn.load_state_dict(best_state, strict=True)

    path = save_encoder(
        pre,
        extra={
            "seed": args.seed,
            "n_windows": int(n),
            "best_loss": float(best),
            "history": history,
        },
    )
    n_applied = len(tagn.encoder_state())
    print(f"saved {path} ({n_applied} trunk tensors)", flush=True)

    RES.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_windows": int(n),
        "best_loss": float(best),
        "epochs_run": len(history),
        "device": str(device),
        "n_encoder_tensors": n_applied,
        "history": history,
        "checkpoint": str(path),
    }
    (RES / "exp_008_pretrain.json").write_text(json.dumps(summary, indent=2))

    from src.utils.registry import append

    append({
        "experiment": "exp_008",
        "seed": args.seed,
        "task": "self_supervised_pretraining", "model": "tagn_pretrain",
        "metrics": {
            "n_windows": int(n),
            "best_loss": float(best),
            "epochs_run": len(history),
            "final_recon": history[-1]["recon"],
            "final_forecast": history[-1]["forecast"],
            "final_contrastive": history[-1]["contrastive"],
        },
    })

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, key, title in zip(
        ax,
        ("recon", "forecast", "contrastive"),
        ("masked-KPI recon MSE", "next-step forecast MSE", "NT-Xent contrastive"),
    ):
        a.plot([h["epoch"] for h in history], [h[key] for h in history], marker="o")
        a.set_xlabel("epoch"); a.set_title(title); a.grid(alpha=0.3)
    fig.suptitle(f"TAGN self-supervised pretraining (seed {args.seed})")
    fig.tight_layout()
    fig.savefig(FIG / "exp_008_pretrain_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
