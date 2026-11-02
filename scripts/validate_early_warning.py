"""Event-level early-warning evaluation for TAGN checkpoints.

The original paper reported the argmax hazard lag over the whole window, which
can be dominated by post-onset steps.  This script reports:

  * per-step AUC/AP on the masked pre-onset region,
  * event-level detection rate at a validation-calibrated false-alarm budget,
  * lead time among detected events (median, p10, p90),
  * test false alarms per normal window,
  * the old argmax-lag metric for comparison.

Usage:
    python scripts/validate_early_warning.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.tagn import TagnModel, load_graphs, model_config

INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints" / "exp_002"
RES = ROOT / "outputs" / "results"
SEEDS = (42, 123, 456, 789, 1337)


def load_split(name):
    X = np.load(INTERIM / "chunks_X.npy")
    scaler = json.loads((PROCESSED / "scaler.json").read_text())
    loc = np.asarray(scaler["loc"])[:16]
    scale = np.asarray(scaler["scale"])[:16]
    Xs = X.copy().astype(np.float32)
    Xs[:, :, :16] = (Xs[:, :, :16] - loc) / scale
    idx = np.load(PROCESSED / "split_indices.npz")[name]
    tar = dict(np.load(PROCESSED / "targets.npz"))
    proto = np.load(PROCESSED / "graph_prototypes.npz")["chunk_prototypes"][idx]
    meta = pd.read_parquet(INTERIM / "chunks_meta.parquet").iloc[idx]
    return {
        "X": torch.tensor(Xs[idx], dtype=torch.float32),
        "proto": torch.tensor(proto, dtype=torch.long),
        "ano": tar["y_anomaly"][idx],
        "onset": tar["onset"][idx],
        "yew": tar["y_ew"][idx],
        "mask": tar["ew_mask"][idx],
        "groups": meta["recording"].to_numpy(),
    }


@torch.no_grad()
def hazard(model, f, batch=128):
    out = []
    for i in range(0, len(f["X"]), batch):
        lg, _, _ = model(f["X"][i:i + batch], f["proto"][i:i + batch])
        out.append(torch.sigmoid(lg["hazard"]))
    return torch.cat(out).numpy()


def load_model(seed):
    cfg = model_config()
    m = TagnModel(cfg)
    m.set_prototypes(load_graphs()[0])
    state = torch.load(CKPT / f"tagn_seed{seed}.pt", map_location="cpu")
    m.load_state_dict(state, strict=True)
    m.eval()
    return m


def max_hazard(sig, rows, mask=None):
    out = []
    for i in rows:
        s = sig[i]
        if mask is not None:
            s = s[mask[i] > 0]
        out.append(float(s.max()) if len(s) else 0.0)
    return np.asarray(out)


def event_metrics(sig, f, threshold):
    ano, onset = f["ano"], f["onset"]
    ev = np.where((ano == 1) & (onset > 0))[0]
    norm = np.where(ano == 0)[0]
    leads, detected = [], 0
    for i in ev:
        pre = sig[i, : onset[i]]
        hit = np.where(pre >= threshold)[0]
        if len(hit):
            detected += 1
            leads.append(float(onset[i] - hit[0]))
    fa = float(np.mean([sig[i].max() >= threshold for i in norm])) if len(norm) else 0.0
    return {
        "n_events": int(len(ev)),
        "detection_rate": detected / max(1, len(ev)),
        "median_lead_steps": float(np.median(leads)) if leads else None,
        "p10_lead_steps": float(np.percentile(leads, 10)) if leads else None,
        "p90_lead_steps": float(np.percentile(leads, 90)) if leads else None,
        "false_alarms_per_normal_window": fa,
        "threshold": float(threshold),
    }


def main():
    val = load_split("val")
    test = load_split("test")
    # Per-step AUC/AP on masked pre-onset steps.
    all_rows = []
    for seed in SEEDS:
        m = load_model(seed)
        sv, st = hazard(m, val), hazard(m, test)

        pick = np.where((test["ano"] == 1) & (test["onset"] > 0))[0]
        ystep, pstep = [], []
        for i in pick:
            mk = test["mask"][i] > 0
            ystep.append(test["yew"][i][mk])
            pstep.append(st[i][mk])
        norm = np.where(test["ano"] == 0)[0]
        ystep.append(np.zeros(128 * len(norm)))
        pstep.append(st[norm].reshape(-1))
        ystep = np.concatenate(ystep)
        pstep = np.concatenate(pstep)
        step_auc = float(roc_auc_score(ystep, pstep))
        step_ap = float(average_precision_score(ystep, pstep))

        # Threshold for a 1% and 5% false-alarm budget on validation normals.
        vnorm = np.where(val["ano"] == 0)[0]
        vmax = np.asarray([sv[i].max() for i in vnorm])
        budgets = {}
        for q in (0.99, 0.95):
            thr = float(np.quantile(vmax, q))
            budgets[str(q)] = event_metrics(st, test, thr)

        # Old argmax-over-window lag, for comparison.
        ev = np.where((test["ano"] == 1) & (test["onset"] > 0))[0]
        pred = st[ev].argmax(1)
        lag = pred - test["onset"][ev]
        old = {
            "n_chunks": int(len(ev)),
            "median_lag_steps": float(np.median(lag)),
            "mean_abs_lead_err_steps": float(np.mean(np.abs(lag))),
            "warning_before_onset_frac": float(np.mean(lag < 0)),
        }

        all_rows.append({
            "seed": seed,
            "per_step_auc": step_auc,
            "per_step_ap": step_ap,
            "event_level_1pct": budgets["0.99"],
            "event_level_5pct": budgets["0.95"],
            "argmax_lag": old,
        })
        print(f"seed {seed}: step AUC {step_auc:.3f} AP {step_ap:.3f} | "
              f"1% FA det {budgets['0.99']['detection_rate']:.2f} "
              f"lead {budgets['0.99']['median_lead_steps']} | "
              f"old warn-before {old['warning_before_onset_frac']:.2f}")

    def agg(key):
        vals = [r[key] for r in all_rows]
        return {"mean": float(np.mean(vals)), "sd": float(np.std(vals))}

    summary = {
        "n_seeds": len(SEEDS),
        "n_test_events": all_rows[0]["event_level_1pct"]["n_events"],
        "per_step_auc": agg("per_step_auc"),
        "per_step_ap": agg("per_step_ap"),
        "event_level_1pct": {
            k: {
                "mean": float(np.mean([r["event_level_1pct"][k] for r in all_rows])),
                "sd": float(np.std([r["event_level_1pct"][k] for r in all_rows])),
            }
            for k in ("detection_rate", "false_alarms_per_normal_window")
        },
        "event_level_5pct": {
            k: {
                "mean": float(np.mean([r["event_level_5pct"][k] for r in all_rows])),
                "sd": float(np.std([r["event_level_5pct"][k] for r in all_rows])),
            }
            for k in ("detection_rate", "false_alarms_per_normal_window")
        },
        "argmax_lag": {
            "warning_before_onset_frac": {
                "mean": float(np.mean([r["argmax_lag"]["warning_before_onset_frac"] for r in all_rows])),
                "sd": float(np.std([r["argmax_lag"]["warning_before_onset_frac"] for r in all_rows])),
            },
            "mean_abs_lead_err_steps": {
                "mean": float(np.mean([r["argmax_lag"]["mean_abs_lead_err_steps"] for r in all_rows])),
                "sd": float(np.std([r["argmax_lag"]["mean_abs_lead_err_steps"] for r in all_rows])),
            },
        },
        "rows": all_rows,
    }
    RES.mkdir(parents=True, exist_ok=True)
    path = RES / "early_warning_validation.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
