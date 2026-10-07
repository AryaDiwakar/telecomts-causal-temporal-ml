"""exp_003 - ablations over the proposed model's components.

Variants (short runs, seed 42):
  A full            : reference (joint heads, dynamic graph)
  B no_graph        : drop GAT entirely
  C static_graph    : single mean adjacency for every chunk
  D undirected      : symmetrize dynamic graphs
  E no_nodeconv     : replace depthwise node conv with pointwise projection
  F hazard_only     : only train the early-warning head

Usage:
    python -m src.experiments.exp_003_ablations
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import src.experiments.common as C  # noqa: E402
from src.models.tagn import TagnModel, load_graphs, model_config  # noqa: E402
from src.utils.registry import append  # noqa: E402

RES = ROOT / "outputs" / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--variant", type=str, default=None)  # run one variant
    args = ap.parse_args()

    variants = [args.variant] if args.variant else [
        "full", "no_graph", "static_graph", "undirected", "no_nodeconv", "hazard_only",
    ]
    frames, meta, rca = C.load_splits()
    out_all = {}
    for variant in variants:
        cfg = model_config()
        cfg["epochs"] = args.epochs
        cfg["patience"] = args.patience
        cfg["normal_to_anomalous_ratio"] = 4.0
        cfg["train_batch"] = 64
        active = None if variant != "hazard_only" else ["early_warning"]
        model = TagnModel(cfg, variant=variant)
        model.to(cfg["device"])
        prototypes, _ = load_graphs()
        model.set_prototypes(prototypes.to(cfg["device"]))
        for split in frames:
            frames[split] = {k: v.to(cfg["device"]) if torch.is_tensor(v) else v for k, v in frames[split].items()}
        print(f"[ablation] {variant}", flush=True)
        res = C.fit(model, frames, rca, cfg, args.seed, active_heads=active)
        res.pop("best_state", None)
        out_all[variant] = {"test": res["test"], "best_val_pr_auc": res["best_val_pr_auc"],
                            "history": res["history"]}
        with open(RES / f"exp_003_ablation_{variant}.json", "w") as fh:
            json.dump(out_all[variant], fh, indent=2)
        append({
            "experiment": "exp_003", "seed": args.seed, "task": "ablation",
            "model": "tagn", "variant": variant,
            "metrics": {
                "test_pr_auc": res["test"].get("anomaly", {}).get("pr_auc"),
                "test_roc_auc": res["test"].get("anomaly", {}).get("roc_auc"),
                "test_type_acc": res["test"].get("type", {}).get("accuracy"),
                "test_rca_jaccard": res["test"].get("rca", {}).get("type_template_jaccard"),
                "test_ew_before_frac": res["test"].get("early_warning", {}).get("warning_before_onset_frac"),
                "best_val_pr_auc": res["best_val_pr_auc"],
                "epochs": len(res["history"]),
            },
        })
    print(json.dumps({v: {"pr_auc": out_all[v]["test"].get("anomaly", {}).get("pr_auc"),
                          "acc": out_all[v]["test"].get("type", {}).get("accuracy")}
                      for v in variants}, indent=2))


if __name__ == "__main__":
    main()