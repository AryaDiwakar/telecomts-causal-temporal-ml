"""exp_005 - Setting II: leave-recordings-out cross-validation (TAGN full).

For each of the 5 folds (GroupKFold by recording), train on the complement
recordings and evaluate on held-out recordings — the honest generalization
test that blocks the recording-memorization shortcut.

Leakage fix (docs/benchmark_audit_v3.md, finding R4): the scaler and the graph
KMeans prototypes are fit on the Setting-I train set, which overlaps 69-70% of
each fold's held-out recordings. This now refits both per fold using only that
fold's train recordings (``--fit-per-fold``, default on). Pass
``--legacy-global-artifacts`` to reproduce the pre-fix behaviour for continuity.

Usage:
    python -m src.experiments.exp_005_loro --seed 42 --epochs 8 --patience 3
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
from src.models.graph_fold import fold_prototypes  # noqa: E402
from src.preprocessing.prepare import fit_scaler, load_config  # noqa: E402
from src.utils.registry import append  # noqa: E402

RES = ROOT / "outputs" / "results"
LORO = ROOT / "data" / "processed" / "loro"


def _apply_scaler(X: np.ndarray, scaler: dict) -> np.ndarray:
    """Scale the 16 numeric KPI channels, leaving the 2 protocol channels raw.

    Matches the original exp_005 behaviour; the scaler dict is now fit per fold.
    """
    loc = np.asarray(scaler["loc"])[:16]
    scale = np.asarray(scaler["scale"])[:16]
    out = X.copy()
    out[:, :, :16] = (out[:, :, :16] - loc) / scale
    return out


def frames_from_ids(ids: np.ndarray, X, tar, proto) -> dict:
    return {
        "X": torch.tensor(X[ids], dtype=torch.float32),
        "ano": torch.tensor(tar["y_anomaly"][ids], dtype=torch.long),
        "typ": torch.tensor(tar["y_type"][ids], dtype=torch.long),
        "aff": torch.tensor(tar["y_affected"][ids], dtype=torch.float32),
        "ewy": torch.tensor(tar["y_ew"][ids], dtype=torch.float32),
        "ewm": torch.tensor(tar["ew_mask"][ids], dtype=torch.float32),
        "ons": torch.tensor(tar["onset"][ids], dtype=torch.long),
        "proto": torch.tensor(proto[ids], dtype=torch.long),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--legacy-global-artifacts",
        action="store_true",
        help="Reuse the Setting-I train scaler/prototypes (pre-fix, leakage-prone).",
    )
    args = ap.parse_args()

    X = np.load(ROOT / "data" / "interim" / "chunks_X.npy")
    tar = dict(np.load(ROOT / "data" / "processed" / "targets.npz"))
    rca = json.loads((ROOT / "data" / "processed" / "rca_templates.json").read_text())
    method = load_config()["windowing"]["normalization_method"]
    gb = np.load(ROOT / "data" / "processed" / "graph_bins.npz")
    proto_bins = np.load(ROOT / "data" / "processed" / "graph_prototypes.npz")
    chunk_bins, adj = proto_bins["chunk_bins"], gb["adjacencies"]

    if args.legacy_global_artifacts:
        print("WARNING: --legacy-global-artifacts uses Setting-I train scaler "
              "+ prototypes (leaks ~69-70% of each fold's test recordings).",
              flush=True)
        scaler = json.loads((ROOT / "data" / "processed" / "scaler.json").read_text())
        Xs = _apply_scaler(X, scaler)
        global_proto = proto_bins["chunk_prototypes"]

    progress_fp = RES / f"exp_005_loro_seed{args.seed}.progress.json"
    fold_results = []
    start_fold = 0
    if args.resume and progress_fp.exists():
        progress = json.loads(progress_fp.read_text())
        fold_results = progress["fold_results"]
        start_fold = progress["next_fold"]
        print(f"resuming at fold {start_fold}", flush=True)

    for k in range(start_fold, 5):
        ids = {role: np.load(LORO / f"fold{k}_{role}.npy") for role in ("train", "val", "test")}
        if args.legacy_global_artifacts:
            fold_X, fold_assignment = Xs, global_proto
            fold_protos, _ = load_graphs()
        else:
            # fit scaler + prototypes on this fold's train recordings only (R4 fix)
            fold_X = _apply_scaler(X, fit_scaler(X, ids["train"], method))
            fold_protos, fold_assignment = fold_prototypes(chunk_bins, adj, ids["train"], X=X)
            fold_protos = torch.tensor(np.asarray(fold_protos), dtype=torch.float32)
        frames = {role: frames_from_ids(ids[role], fold_X, tar, fold_assignment) for role in ids}
        cfg = model_config()
        cfg["epochs"] = args.epochs
        cfg["patience"] = args.patience
        cfg["normal_to_anomalous_ratio"] = 4.0
        cfg["train_batch"] = 64
        model = TagnModel(cfg)
        model.to(cfg["device"])
        model.set_prototypes(fold_protos.to(cfg["device"]))
        for split in frames:
            frames[split] = {k: v.to(cfg["device"]) if torch.is_tensor(v) else v for k, v in frames[split].items()}
        print(f"[loro fold {k}]", flush=True)
        res = C.fit(model, frames, rca, cfg, args.seed * (k + 1))
        fold_results.append({
            "fold": k,
            "test_anomaly": res["test"].get("anomaly", {}),
            "test_type": res["test"].get("type", {}),
            "test_rca": res["test"].get("rca", {}),
            "best_val_pr_auc": res["best_val_pr_auc"],
            "n_test_anom": int(frames["test"]["ano"].sum()),
        })
        progress_fp.write_text(json.dumps({"next_fold": k + 1, "fold_results": fold_results}, indent=2))

    def avg(key, sub):
        vals = [f["test_" + key].get(sub) for f in fold_results if f["test_" + key].get(sub) is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "folds": fold_results,
        "aggregate": {
            "anomaly_pr_auc": avg("anomaly", "pr_auc"),
            "anomaly_roc_auc": avg("anomaly", "roc_auc"),
            "type_accuracy": avg("type", "accuracy"),
            "type_macro_f1_seen": avg("type", "macro_f1_seen"),
            "rca_type_template_jaccard": avg("rca", "type_template_jaccard"),
            "rca_topk_shift_jaccard": avg("rca", "topk_shift_jaccard"),
        },
    }
    with open(RES / f"exp_005_loro_seed{args.seed}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    if progress_fp.exists():
        progress_fp.unlink()
    append({"experiment": "exp_005", "seed": args.seed, "task": "loro_generalization",
            "model": "tagn", "metrics": {k: v for k, v in summary["aggregate"].items() if v is not None}})
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
