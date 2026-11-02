"""Root-cause ranking metrics from a trained TAGN checkpoint (no retraining).

exp_002 has its own ``evaluate`` and therefore does not emit the fixed-k RCA
ranking metrics added to ``src/evaluation/metrics.rca_ranking_metrics``. We can
recover them exactly by reloading the saved ``best_state`` and re-running the
per-KPI head on the anomalous test chunks -- the checkpoint fully determines the
forward pass (see scripts/bootstrap_ci.py for the reconstruction check).

Outputs:
  outputs/results/rca_ranking_exp_002.json

Usage:
    python -m scripts.rca_from_checkpoint --seeds 42 123 456 789 1337
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
from src.evaluation.metrics import rca_ranking_metrics  # noqa: E402
from src.models.tagn import TagnModel, load_graphs, model_config  # noqa: E402

RES = ROOT / "outputs" / "results"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints" / "exp_002"


def load_model(seed: int):
    cfg = model_config()
    cfg["anomaly_trunk_weight"] = 0.0  # exp_002 default; see bootstrap_ci.py
    model = TagnModel(cfg)
    model.to(cfg["device"])
    prototypes, _ = load_graphs()
    model.set_prototypes(prototypes.to(cfg["device"]))
    state = torch.load(ROOT / "checkpoints" / "exp_002" / f"tagn_seed{seed}.pt",
                       map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


@torch.no_grad()
def test_logits(model, frames, batch=128):
    f = frames["test"]
    type_l, kpi_l = [], []
    for i in range(0, len(f["X"]), batch):
        lg, _, _ = model(f["X"][i:i + batch], f["proto"][i:i + batch])
        type_l.append(lg["type"])
        kpi_l.append(lg["kpi"])
    return torch.cat(type_l).cpu().numpy(), torch.cat(kpi_l).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1337])
    args = ap.parse_args()

    frames, meta, rca = C.load_splits()
    kpi_names = json.load((ROOT / "data" / "interim" / "kpi_names.json").open())["numeric_kpis"]
    kpi_idx = {k: i for i, k in enumerate(kpi_names)}
    templates = rca["rca_templates"]
    inv = {v: k for k, v in rca["type_map"].items()}
    y_test = frames["test"]["ano"].numpy()
    aff = frames["test"]["aff"].numpy()
    anom = y_test == 1
    type_gt = frames["test"]["typ"].numpy()
    gt_sets = [set(np.where(aff[j])[0]) for j in np.where(anom)[0]]

    out = {}
    for seed in args.seeds:
        model, cfg = load_model(seed)
        dev = cfg["device"]
        for split in frames:
            frames[split] = {k: (v.to(dev) if torch.is_tensor(v) else v)
                             for k, v in frames[split].items()}
        type_logits, kpi_logits = test_logits(model, frames)
        kpi_anom = kpi_logits[anom]
        type_pred = type_logits[anom].argmax(1)
        rank = rca_ranking_metrics(kpi_anom, gt_sets)
        # type-template jaccard for continuity with exp_002's own output
        jac = []
        for j, tpred in enumerate(type_pred):
            gtset = gt_sets[j]
            tname = inv[int(tpred)]
            pred_t = {kpi_idx[k] for k in templates[tname] if k in kpi_idx}
            jac.append(len(gtset & pred_t) / max(1, len(gtset | pred_t)))
        rank["type_template_jaccard"] = float(np.mean(jac))
        rank["type_accuracy"] = float(np.mean(type_pred == type_gt[anom]))
        out[str(seed)] = rank
        print(f"seed {seed}: recall@1={rank['recall@1']:.4f} recall@3={rank['recall@3']:.4f} "
              f"recall@5={rank['recall@5']:.4f} MRR={rank['mrr']:.4f} MAP={rank['map']:.4f} "
              f"type_jac={rank['type_template_jaccard']:.4f}")

    (ROOT / "outputs" / "results" / "rca_ranking_exp_002.json").write_text(
        json.dumps(out, indent=2)
    )
    print("wrote outputs/results/rca_ranking_exp_002.json")


if __name__ == "__main__":
    main()
