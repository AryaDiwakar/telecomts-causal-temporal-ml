"""exp_003 - component ablations over the proposed TAGN model.

Protocol note (audit finding)
-----------------------------
The original version trained every variant from scratch with ``C.fit`` and no
stage-A anomaly anchor. That made its ``full`` variant much weaker than the
headline model in ``exp_002`` (test PR ~0.55 vs ~0.93), so the comparison was
confounded: it measured *training-protocol* differences, not the contribution of
the graph. This version mirrors the headline recipe:

1. calibrate the summary normaliser ; fit the linear summary anchor
   (``fit_summary_head`` + ``fit_summary_residual``) exactly as exp_002 does;
2. freeze ``summary_head`` so stage B only learns a residual on top of it;
3. let the trunk contribute to the anomaly logit (``trunk_anomaly_weight``,
   default 1.0) so the ablation can actually move detection;
4. train stage B with the shared ``C.fit`` loop.

Every variant therefore shares the anchor, the split, the seed, the optimiser and
the validation-tuned threshold. The reference readout (anchor only, no trunk) is
recorded alongside so the table shows anchor -> full -> each ablation.

Variants:
  full          : reference (joint heads, dynamic graph)
  no_graph      : drop the GAT entirely
  static_graph  : one mean adjacency for every chunk
  undirected    : symmetrise the dynamic graphs
  no_nodeconv   : pointwise node projection instead of depthwise conv
  hazard_only   : train only the early-warning head

Usage:
    python -m src.experiments.exp_003_ablations --variant full --seed 42
    python -m src.experiments.exp_003_ablations --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import src.experiments.common as C  # noqa: E402
from src.evaluation.metrics import summarize  # noqa: E402
from src.models.tagn import (  # noqa: E402
    TagnModel,
    chunk_summary,
    load_graphs,
    model_config,
)
from src.utils.registry import append  # noqa: E402

RES = ROOT / "outputs" / "results"
ALL_VARIANTS = [
    "full", "no_graph", "static_graph", "undirected", "no_nodeconv", "hazard_only",
]


def anchor_metrics(model, frames, cfg) -> dict:
    """Val-tuned-threshold test metrics for the frozen summary anchor."""
    with torch.no_grad():
        val_stats = (chunk_summary(frames["val"]["X"]) - model.summary_loc) / model.summary_scale
        val_logits = (model.summary_head(val_stats).squeeze(-1)
                      + model.summary_residual(val_stats).squeeze(-1))
        test_stats = (chunk_summary(frames["test"]["X"]) - model.summary_loc) / model.summary_scale
        test_logits = (model.summary_head(test_stats).squeeze(-1)
                       + model.summary_residual(test_stats).squeeze(-1))
    val_an = summarize(frames["val"]["ano"].cpu().numpy(), val_logits.cpu().numpy())
    test_an = summarize(frames["test"]["ano"].cpu().numpy(), test_logits.cpu().numpy(),
                        threshold=val_an["threshold"])
    return {"val": val_an, "test": test_an}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--variant", type=str, default=None)
    ap.add_argument("--stage-a", dest="stage_a", action="store_true", default=True)
    ap.add_argument("--no-stage-a", dest="stage_a", action="store_false")
    ap.add_argument("--summary-epochs", type=int, default=200)
    ap.add_argument("--summary-residual-epochs", type=int, default=40)
    ap.add_argument("--trunk-anomaly-weight", type=float, default=1.0)
    args = ap.parse_args()

    variants = [args.variant] if args.variant else ALL_VARIANTS
    frames, _, rca = C.load_splits()
    rng = np.random.default_rng(args.seed)
    out_all: dict = {}

    for variant in variants:
        cfg = model_config()
        cfg["epochs"] = args.epochs
        cfg["patience"] = args.patience
        cfg["normal_to_anomalous_ratio"] = 4.0
        cfg["train_batch"] = 64
        cfg["anomaly_trunk_weight"] = args.trunk_anomaly_weight
        active = ["early_warning"] if variant == "hazard_only" else None

        model = TagnModel(cfg, variant=variant)
        model.to(cfg["device"])
        prototypes, _ = load_graphs()
        model.set_prototypes(prototypes.to(cfg["device"]))
        for split in frames:
            frames[split] = {
                k: v.to(cfg["device"]) if torch.is_tensor(v) else v
                for k, v in frames[split].items()
            }

        print(f"[ablation] {variant}", flush=True)
        record: dict = {"variant": variant, "seed": args.seed}
        if args.stage_a:
            model.calibrate_summary(frames["train"]["X"])
            model._summary_calibrated = True
            C.fit_summary_head(model, frames, cfg, rng, epochs=args.summary_epochs)
            C.fit_summary_residual(model, frames, cfg, rng,
                                   epochs=args.summary_residual_epochs)
            for p in model.summary_head.parameters():
                p.requires_grad = False
            record["anchor"] = anchor_metrics(model, frames, cfg)
            print(f"  anchor test PR {record['anchor']['test']['pr_auc']:.4f}", flush=True)

        res = C.fit(model, frames, rca, cfg, args.seed, active_heads=active)
        res.pop("best_state", None)
        record["test"] = res["test"]
        record["best_val_pr_auc"] = res["best_val_pr_auc"]
        record["history"] = res["history"]
        out_all[variant] = record
        with open(RES / f"exp_003_ablation_{variant}.json", "w") as fh:
            json.dump(record, fh, indent=2)
        append({
            "experiment": "exp_003",
            "seed": args.seed,
            "task": "ablation",
            "model": f"tagn_{variant}",
            "metrics": {
                "pr_auc": res["test"].get("anomaly", {}).get("pr_auc"),
                "roc_auc": res["test"].get("anomaly", {}).get("roc_auc"),
                "type_accuracy": res["test"].get("type", {}).get("accuracy"),
                "rca_type_template_jaccard": res["test"].get("rca", {}).get("type_template_jaccard"),
                "anchor_pr_auc": (record.get("anchor", {}).get("test", {})
                                  .get("pr_auc")),
            },
            "config": {"stage_a": args.stage_a,
                       "trunk_anomaly_weight": args.trunk_anomaly_weight,
                       "epochs": args.epochs},
        })

    print("\n" + "=" * 72)
    hdr = f"{'variant':<14} {'anchor PR':>10} {'test PR':>9} {'test ROC':>9} {'type acc':>9}"
    print(hdr)
    print("-" * 72)
    for v in variants:
        t = out_all[v]["test"]
        a = out_all[v].get("anchor", {}).get("test", {})
        print(f"{v:<14} {a.get('pr_auc', float('nan')):>10.4f} "
              f"{t.get('anomaly', {}).get('pr_auc', float('nan')):>9.4f} "
              f"{t.get('anomaly', {}).get('roc_auc', float('nan')):>9.4f} "
              f"{t.get('type', {}).get('accuracy', float('nan')):>9.4f}")


if __name__ == "__main__":
    main()
