"""Dump per-chunk test/val anomaly scores for the TAGN checkpoints.

Phase 1 (error anatomy) needs the *raw* per-chunk scores of every headline
model on the same v3 test split, so we can slice errors by subgroup / type /
recording. The TAGN scores are not stored by exp_002, so they are reconstructed
here from the checkpoints (exactly as ``scripts/bootstrap_ci.py`` does) and
written to ``outputs/results/preds/tagn_seed<seed>_preds.npz``.

The classical baselines already live in ``preds/exp_001_test_preds.npz`` and the
temporal baselines are dumped by ``src/experiments/exp_013_temporal_baselines``
(``preds/exp_013_<model>_seed<seed>_preds.npz``).

Usage:
    python -m scripts.score_models --seeds 42 123 456 789 1337
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
from src.evaluation.metrics import pr_auc, roc_auc  # noqa: E402
from src.models.tagn import TagnModel, load_graphs, model_config  # noqa: E402

RES = ROOT / "outputs" / "results"
PREDS = RES / "preds"
CKPT = ROOT / "checkpoints" / "exp_002"


def _load_model(seed: int):
    ckpt = CKPT / f"tagn_seed{seed}.pt"
    if not ckpt.exists():
        raise SystemExit(f"missing checkpoint {ckpt}")
    cfg = model_config()
    # exp_002's headline detection uses a zero-weighted trunk residual (the
    # stage-A summary anchor dominates). model_config() omits the key, so match
    # the checkpoint explicitly.
    cfg["anomaly_trunk_weight"] = 0.0
    model = TagnModel(cfg)
    model.to(cfg["device"])
    prototypes, _ = load_graphs()
    model.set_prototypes(prototypes.to(cfg["device"]))
    model.load_state_dict(torch.load(ckpt, map_location=cfg["device"]), strict=False)
    model.eval()
    return model, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1337])
    ap.add_argument("--batch", type=int, default=32, help="keep small: MPS OOMs at 128")
    args = ap.parse_args()

    PREDS.mkdir(parents=True, exist_ok=True)
    frames, _, _ = C.load_splits()
    idx = C.split_indices()
    test_ids = np.asarray(idx["test"])
    groups = C.chunk_groups(test_ids)

    for seed in args.seeds:
        model, cfg = _load_model(seed)
        frames = {
            s: {k: (v.to(cfg["device"]) if torch.is_tensor(v) else v) for k, v in d.items()}
            for s, d in frames.items()
        }
        y_test, s_test = C.anomaly_scores(model, frames, "test", batch=args.batch)
        y_val, s_val = C.anomaly_scores(model, frames, "val", batch=args.batch)
        path = PREDS / f"tagn_seed{seed}_preds.npz"
        np.savez_compressed(
            PREDS / f"tagn_seed{seed}_preds.npz",
            y_test=y_test, s_test=s_test, y_val=y_val, s_val=s_val,
            groups=groups, chunk_ids=test_ids,
        )
        ref = json.loads((RES / f"exp_002_tagn_seed{seed}.json").read_text())["test"]["anomaly"]
        pr, roc = pr_auc(y_test, s_test), roc_auc(y_test, s_test)
        ok = abs(pr - ref["pr_auc"]) < 1e-6 and abs(roc - ref["roc_auc"]) < 1e-6
        print(f"seed {seed:4d}  PR {pr:.4f} (json {ref['pr_auc']:.4f})  "
              f"ROC {roc:.4f} (json {roc:.4f})  {'OK' if ok else 'MISMATCH'}  -> {ckpt_name(seed)}")

    print(f"wrote {PREDS}/tagn_seed*_preds.npz")


def ckpt_name(seed: int) -> str:
    return f"preds/tagn_seed{seed}_preds.npz"


if __name__ == "__main__":
    main()
