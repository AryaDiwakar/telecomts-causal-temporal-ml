"""Paired cluster bootstrap: is TAGN better than the strongest baseline?

Uncertainty for a *single* model (finding M5) is a cluster bootstrap over
recordings. For the paper's central claim -- "TAGN > baselines" -- the right
statistic is the paired difference, because both models are evaluated on the
*same* test chunks. We resample recordings with replacement, recompute PR/ROC
for both models on the shared resample, and report the difference distribution.

Reports, per TAGN seed:
  dPR  = PR_TAGN - PR_baseline   (95% CI, two-sided)
  dROC = ROC_TAGN - ROC_baseline
  p    = fraction of resamples with dPR <= 0  (one-sided evidence for TAGN)

Inputs:
  checkpoints/exp_002/tagn_seed*.pt              (reconstructed, trunk weight 0)
  outputs/results/preds/exp_001_test_preds.npz   (classical baselines)

Outputs:
  outputs/results/paired_test.json

Usage:
    python -m scripts.paired_test --baseline logreg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.evaluation.metrics import pr_auc, roc_auc  # noqa: E402
from src.models.tagn import TagnModel, load_graphs, model_config  # noqa: E402

RES = ROOT / "outputs" / "results"
CKPT = ROOT / "checkpoints" / "exp_002"


def tag_scores(seed: int, frames_test: dict) -> np.ndarray:
    """Reconstruct TAGN test anomaly scores (matches scripts/bootstrap_ci.py)."""
    cfg = model_config()
    cfg["anomaly_trunk_weight"] = 0.0  # exp_002 default
    model = TagnModel(cfg)
    model.to(cfg["device"])
    prototypes, _ = load_graphs()
    model.set_prototypes(prototypes.to(cfg["device"]))
    state = torch.load(CKPT / f"tagn_seed{seed}.pt", map_location=cfg["device"])
    model.load_state_dict(state, strict=False)
    model.eval()
    ft = {k: (v.to(cfg["device"]) if torch.is_tensor(v) else v)
          for k, v in frames_test.items()}
    chunks = []
    with torch.no_grad():
        for i in range(0, len(ft["X"]), 32):
            lg, _, _ = model(ft["X"][i:i + 32], ft["proto"][i:i + 32])
            chunks.append(lg["anomaly"])
    return torch.cat(chunks).detach().cpu().numpy()


def metric_delta(y, sa, sb, groups, metric, n_boot, seed):
    idx_by_group = [np.where(groups == g)[0] for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    point = float(metric(y, sa) - metric(y, sb))
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx_by_group), len(idx_by_group))
        sel = np.concatenate([idx_by_group[p] for p in pick])
        yy = y[sel]
        if yy.min() == yy.max():
            continue
        deltas.append(float(metric(yy, sa[sel]) - metric(yy, sb[sel])))
    arr = np.asarray(deltas)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {
        "point": point, "lo": float(lo), "hi": float(hi),
        "std": float(arr.std(ddof=1)), "frac_le_0": float((arr <= 0).mean()),
        "n_valid": int(len(arr)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1337])
    ap.add_argument("--baseline", type=str, default="logreg")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    from src.experiments.common import load_splits

    d = np.load(RES / "preds" / "exp_001_test_preds.npz", allow_pickle=True)
    y, groups = d["y"].astype(int), d["groups"]
    base = d[args.baseline].astype(float)
    frames, _, _ = load_splits()
    frames_test = frames["test"]

    out = {"baseline": args.baseline, "n": int(len(y)), "n_pos": int(y.sum()),
           "n_clusters": int(len(np.unique(groups))), "seeds": {}}
    for seed in args.seeds:
        s = tag_scores(seed, frames_test)
        assert len(s) == len(y), (len(s), len(y))
        dpr = metric_delta(y, s, base, groups, pr_auc, args.n_boot, seed)
        droc = metric_delta(y, s, base, groups, roc_auc, args.n_boot, seed)
        out["seeds"][str(seed)] = {"dPR": dpr, "dROC": droc}
        print(f"seed {seed}: dPR {dpr['point']:+.4f} "
              f"[{dpr['lo']:+.4f}, {dpr['hi']:+.4f}] p(dPR<=0)={dpr['frac_le_0']:.4f} "
              f"| dROC {droc['point']:+.4f} [{droc['lo']:+.4f}, {droc['hi']:+.4f}]",
              flush=True)

    (RES / "paired_test.json").write_text(json.dumps(out, indent=2))
    print("wrote outputs/results/paired_test.json")


if __name__ == "__main__":
    main()
