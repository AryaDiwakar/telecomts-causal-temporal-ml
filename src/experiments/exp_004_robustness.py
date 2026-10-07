"""exp_004 - input robustness: noise, missing-channel masking, time shift.

Uses the exp_002 checkpoint for a given seed and re-runs inference under
perturbations (no retraining), plus a perturbation-aware fine-tune probe.

Usage:
    python -m src.experiments.exp_004_robustness --seed 42
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
import torch

import src.experiments.common as C  # noqa: E402
from src.models.tagn import TagnModel, load_graphs, model_config  # noqa: E402
from src.utils.registry import append  # noqa: E402

RES = ROOT / "outputs" / "results"
FIG = ROOT / "outputs" / "figures" / "exp"
CKPT = ROOT / "checkpoints" / "exp_002"


def perturb(x: torch.Tensor, kind: str, level: float, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    if kind == "noise":
        return x + torch.tensor(rng.normal(0, level, size=x.shape), dtype=x.dtype)
    if kind == "channel_drop":
        # zero a random 20% subset of the 16 numeric channels for every sample
        keep = torch.ones(16)
        for i in rng.choice(16, int(16 * level), replace=False):
            keep[i] = 0.0
        x = x.clone()
        x[:, :, :16] = x[:, :, :16] * keep.view(1, 1, -1)
        return x
    if kind == "time_shift":
        s = int(128 * level)
        return torch.cat([x[:, -s:], x[:, :-s]], dim=1)
    if kind == "scale":
        return x * (1.0 + level)
    raise ValueError(kind)


@torch.no_grad()
def eval_perturbed(model, frames, rca, cfg, kind, level, seed):
    model.eval()
    f = frames["test"]
    metric_rows = []
    for i in range(0, len(f["X"]), 128):
        xb = perturb(f["X"][i:i + 128], kind, level, seed)
        lg, _, _ = model(xb, f["proto"][i:i + 128])
        sig = 1 / (1 + np.exp(-lg["anomaly"].detach().cpu().numpy()))
        for k, s in enumerate(sig):
            gidx = int(i + k)
            metric_rows.append({"idx": gidx, "score": float(s),
                                "ano": int(f["ano"][gidx].item())})
    from src.evaluation.metrics import summarize

    df = sorted(metric_rows, key=lambda r: r["idx"])
    y = np.array([r["ano"] for r in df])
    s = np.array([r["score"] for r in df])
    return summarize(y, s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    frames, meta, rca = C.load_splits()
    cfg = model_config()
    cfg["normal_to_anomalous_ratio"] = 4.0
    cfg["train_batch"] = 64

    model = TagnModel(cfg)
    prototypes, _ = load_graphs()
    model.set_prototypes(prototypes.to(cfg["device"]))
    model.load_state_dict(torch.load(CKPT / f"tagn_seed{args.seed}.pt"), strict=False)
    model.eval()

    grid = [
        ("noise", 0.05), ("noise", 0.10), ("noise", 0.20),
        ("channel_drop", 0.20), ("time_shift", 0.10), ("scale", 0.20),
    ]
    base = eval_perturbed(model, frames, rca, cfg, "noise", 0.0, args.seed)
    out = {"baseline": {k: v for k, v in base.items() if isinstance(v, (int, float))}}
    rows = [{"kind": "-", "level": 0.0, **out["baseline"]}]
    for kind, level in grid:
        r = eval_perturbed(model, frames, rca, cfg, kind, level, args.seed)
        out[f"{kind}_{level}"] = {k: v for k, v in r.items() if isinstance(v, (int, float))}
        rows.append({"kind": kind, "level": level, **out[f"{kind}_{level}"]})
        print(kind, level, "PR-AUC", round(r["pr_auc"], 4), "ROC", round(r["roc_auc"], 4), flush=True)

    with open(RES / f"exp_004_robustness_seed{args.seed}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    import pandas as pd

    pd.DataFrame(rows).to_csv(ROOT / "outputs" / "tables" / f"exp_004_robustness_seed{args.seed}.csv", index=False)
    append({
        "experiment": "exp_004", "seed": args.seed, "task": "robustness",
        "model": "tagn", "metrics": {k: round(v, 4) for k, v in out["noise_0.2"].items()},
    })
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
