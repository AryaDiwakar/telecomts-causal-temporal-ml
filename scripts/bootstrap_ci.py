"""Cluster-bootstrap confidence intervals for the headline anomaly metrics.

Finding M5 (docs/benchmark_audit_v3.md): chunks are grouped into recordings and
the models can partly memorise a recording's session signature, so neither a
chunk-level bootstrap nor a 5-seed min/max is an honest uncertainty statement.
This script reloads a trained checkpoint, re-derives its test predictions
(``common.anomaly_scores``), and resamples *recordings* with replacement.

It also verifies the reconstruction: the reproduced PR/ROC-AUC is compared to
the value stored in the experiment's JSON, so a silent loading mismatch cannot
masquerade as a valid CI.

Usage:
    python -m scripts.bootstrap_ci --seed 42 --n-boot 2000
    python -m scripts.bootstrap_ci --seeds 42 123 456 789 1337 --tag exp_002
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

import src.experiments.common as C  # noqa: E402
from src.evaluation.metrics import bootstrap_anomaly_metrics  # noqa: E402
from src.models.tagn import TagnModel, load_graphs, model_config  # noqa: E402

RES = ROOT / "outputs" / "results"
CKPT = ROOT / "checkpoints" / "exp_002"


def _load_model(seed: int):
    ckpt = CKPT / f"tagn_seed{seed}.pt"
    if not ckpt.exists():
        raise SystemExit(f"missing checkpoint {ckpt}")
    cfg = model_config()
    # Match exp_002's training config: the anomaly head is the summary anchor
    # plus a *zero-weighted* trunk residual (--trunk-anomaly-weight default 0).
    # model_config() itself omits the key, so TagnModel.forward would otherwise
    # default it to 1.0 and the reconstruction would not match the checkpoint.
    cfg["anomaly_trunk_weight"] = 0.0
    model = TagnModel(cfg)
    model.to(cfg["device"])
    prototypes, _ = load_graphs()
    model.set_prototypes(prototypes.to(cfg["device"]))
    state = torch.load(ckpt, map_location=cfg["device"])
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


def _result_json(seed: int) -> dict:
    p = RES / f"exp_002_tagn_seed{seed}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--preds", type=str, default=None,
                    help="bootstrap every score array in this .npz (needs y+groups) "
                         "instead of reconstructing a TAGN checkpoint")
    args = ap.parse_args()

    if args.preds:
        d = np.load(args.preds, allow_pickle=True)
        y, groups = d["y"], d["groups"]
        out = {"source": args.preds, "n": int(len(y)),
               "n_pos": int((y > 0).sum()), "n_clusters": int(len(np.unique(groups))),
               "models": {}}
        for key in d.files:
            if key in ("chunk_ids", "y", "groups"):
                continue
            arr = d[key]
            if arr.ndim != 1 or len(arr) != len(y):
                continue
            boot = bootstrap_anomaly_metrics(y, arr, groups, n_boot=args.n_boot, seed=0)
            out["models"][key] = boot
            pc = boot["pr_auc"]
            print(f"{key:12s} PR {pc['point']:.4f} [{pc['lo']:.4f}, {pc['hi']:.4f}]  "
                  f"ROC {boot['roc_auc']['point']:.4f} "
                  f"[{boot['roc_auc']['lo']:.4f}, {boot['roc_auc']['hi']:.4f}]", flush=True)
        path = RES / "bootstrap_ci_classical.json"
        path.write_text(json.dumps(out, indent=2))
        print("wrote", path)
        return

    frames, meta, _ = C.load_splits()
    idx = C.split_indices()
    groups = C.chunk_groups(idx["test"])

    summary = {}
    for seed in args.seeds:
        model, cfg = _load_model(seed)
        dev = cfg["device"]
        for split in frames:
            frames[split] = {
                k: v.to(dev) if torch.is_tensor(v) else v for k, v in frames[split].items()
            }
        y, scores = C.anomaly_scores(model, frames, "test")
        saved = _result_json(seed)
        thr = saved.get("test", {}).get("anomaly", {}).get("threshold_val_tuned")
        boot = bootstrap_anomaly_metrics(
            y, scores, groups, threshold=thr, n_boot=args.n_boot, seed=seed
        )
        # reconstruction check
        chk = {}
        for key in ("pr_auc", "roc_auc"):
            if saved:
                ref = saved["test"]["anomaly"].get(key)
                chk[f"{key}_saved"] = ref
                chk[f"{key}_reproduced"] = boot[key]["point"]
                chk[f"{key}_abs_diff"] = (
                    abs(ref - boot[key]["point"]) if ref is not None else None
                )
        summary[seed] = {"n": int(len(y)), "n_pos": int((y > 0).sum()),
                         "n_clusters": int(len(np.unique(groups))),
                         "bootstrap": boot, "reconstruction_check": chk}
        pc, lc = boot["pr_auc"], boot["roc_auc"]
        print(f"seed {seed}: PR {pc['point']:.4f} [{pc['lo']:.4f}, {pc['hi']:.4f}]  "
              f"ROC {lc['point']:.4f} [{lc['lo']:.4f}, {lc['hi']:.4f}]  "
              f"({pc['n_clusters']} recordings, {pc['n_boot']} boots)", flush=True)

    out = RES / "bootstrap_ci.json"
    out.write_text(json.dumps(summary, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
