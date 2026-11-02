#!/usr/bin/env python3
"""Per-subset (template vs instance-varying) detection + RCA breakdown.

The shipped TelecomTS test windows carry a constant per-type RCA template; the
Track 2 injected windows carry instance-varying ground truth. Reporting one
aggregate number hides that the model behaves very differently on each. This
script re-scores the corrected test split for the exp_002 baseline and the
exp_012 GRL model and reports ROC / PR / F1 and top-k RCA Jaccard separately on
the template and instance-varying subsets.

Writes outputs/results/subset_breakdown.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.evaluation.metrics import summarize
from src.experiments.common import evaluate, load_splits
from src.models.tagn import TagnModel, load_graphs, model_config

RES = ROOT / "outputs" / "results"
CKPT = ROOT / "checkpoints" / "exp_002"


def score_model(model, frames, split="test", batch=128):
    """Return anomaly logits, kpi logits, type preds for a split."""
    model.eval()
    f = frames[split]
    anom, kpi, typ = [], [], []
    with torch.no_grad():
        for i in range(0, len(f["X"]), batch):
            lg, _, _ = model(f["X"][i:i + batch], f["proto"][i:i + batch])
            anom.append(lg["anomaly"].detach().cpu())
            kpi.append(lg["kpi"].detach().cpu())
            typ.append(lg["type"].detach().cpu())
    return torch.cat(anom).numpy(), torch.cat(kpi).numpy(), torch.cat(typ).numpy()


def topk_jaccard(kpi_logits, aff, anom_mask):
    idx = np.where(anom_mask)[0]
    jac = []
    for j in idx:
        gt = set(np.where(aff[j] >= 1)[0])
        topk = set(np.argsort(kpi_logits[j])[::-1][:max(1, len(gt))])
        jac.append(len(gt & topk) / max(1, len(gt | topk)))
    return float(np.mean(jac)) if jac else float("nan")


def load_ckpt_model(cfg, ckpt_path, frames):
    model = TagnModel(cfg, n_types=11)
    p, _ = load_graphs()
    model.to(cfg["device"])
    model.set_prototypes(p.to(cfg["device"]))
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    for s in frames:
        frames[s] = {
            k: (v.to(cfg["device"]) if hasattr(v, "device") else v) for k, v in frames[s].items()
        }
    model.calibrate_summary(frames["train"]["X"])
    model._summary_calibrated = True
    return model


def main() -> None:
    cfg = model_config()
    frames, meta, rca = load_splits()
    te_idx = np.load(ROOT / "data" / "processed" / "split_indices.npz")["test"]
    is_gen = meta.recording.str.contains("synthetic_gen").to_numpy()[te_idx]

    ano = frames["test"]["ano"].cpu().numpy()
    aff = frames["test"]["aff"].cpu().numpy()
    anom_mask = ano == 1
    neg_mask = ~anom_mask  # injected recordings are all-positive, so negatives
    # come from the normal/shipped pool; detection is scored per anomaly family
    # against this common negative pool.

    out = {"n_test": int(len(ano)), "n_pos": int(anom_mask.sum()),
           "n_neg": int(neg_mask.sum()),
           "n_instance_varying": int((anom_mask & is_gen).sum()),
           "n_template": int((anom_mask & ~is_gen).sum())}

    variants = {
        "tagn_baseline": CKPT / "tagn_seed42.pt",
        "tagn_grl": CKPT / "exp_012_grl_seed42.pt",
    }
    for name, ckpt in variants.items():
        if not ckpt.exists():
            print(f"skip {name}: {ckpt} missing")
            continue
        model = load_ckpt_model(cfg, ckpt, frames)
        al, kl, tl = score_model(model, frames, split="test")
        block = {}
        for label, pos_mask in (("template", ~is_gen), ("instance_varying", is_gen)):
            sel = neg_mask | (anom_mask & pos_mask)
            block[label] = {
                "detection": summarize(ano[sel], al[sel]),
                "rca_topk_jaccard": topk_jaccard(kl, aff, anom_mask & pos_mask),
                "n": int(sel.sum()),
                "n_pos": int((anom_mask & pos_mask).sum()),
            }
        block["all"] = {
            "detection": summarize(ano, al),
            "rca_topk_jaccard": topk_jaccard(kl, aff, anom_mask),
        }
        out[name] = block
        print(f"{name}: template RCA {block['template']['rca_topk_jaccard']:.3f} "
              f"inst-var RCA {block['instance_varying']['rca_topk_jaccard']:.3f}")

    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / "subset_breakdown.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {RES / 'subset_breakdown.json'}")


if __name__ == "__main__":
    main()
