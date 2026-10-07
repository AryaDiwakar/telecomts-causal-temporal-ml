"""exp_007 - early-warning analysis at step granularity.

Uses the trained TAGN (seed 42). For anomalous test windows with a pre-onset
region (onset>0): hazard curve over steps, onset-within-horizon ROC-AUC,
lead-time distribution, and an alarm-threshold sweep (alarm -> time-to-detect
vs false alarms per 1k windows).

Usage:
    python -m src.experiments.exp_007_early_warning
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import src.experiments.common as C  # noqa: E402
from src.models.tagn import TagnModel, load_graphs, model_config  # noqa: E402
from src.utils.registry import append  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score

RES = ROOT / "outputs" / "results"
TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures" / "exp"
CKPT = ROOT / "checkpoints" / "exp_002"


@torch.no_grad()
def main() -> None:
    frames, meta, rca = C.load_splits()
    cfg = model_config()
    model = TagnModel(cfg)
    prototypes, _ = load_graphs()
    model.set_prototypes(prototypes.to(cfg["device"]))
    model.load_state_dict(torch.load(CKPT / "tagn_seed42.pt"), strict=False)
    model.eval()
    f = frames["test"]

    haz_all, hazard, mask, onsets = [], None, None, []
    for i in range(0, len(f["X"]), 128):
        lg, _, _ = model(f["X"][i:i + 128], f["proto"][i:i + 128])
        haz_all.append(1 / (1 + np.exp(-lg["hazard"].detach().cpu().numpy())))
    H = np.concatenate(haz_all)  # [n,128]
    labels = f["ewy"].detach().cpu().numpy()    # per-step onset-within-horizon label
    masks = f["ewm"].detach().cpu().numpy()

    # onset-window evaluation
    ano = f["ano"].detach().cpu().numpy()
    ons = f["ons"].detach().cpu().numpy()
    oi = np.where((ano == 1) & (ons > 0))[0]
    step_auc, step_ap = [], []
    for i in oi:
        m = masks[i] > 0
        if m.sum() > 0 and np.unique(labels[i][m]).size == 2:
            step_auc.append(roc_auc_score(labels[i][m], H[i][m]))
            step_ap.append(average_precision_score(labels[i][m], H[i][m]))
    step_stats = {"n_chunks": int(len(oi)),
                  "per_step_auc_mean": float(np.mean(step_auc)) if step_auc else None,
                  "per_step_ap_mean": float(np.mean(step_ap)) if step_ap else None}

    lead = np.argmax(H[oi], axis=1) - ons[oi]
    kw = {
        "n_chunks": int(len(oi)),
        "warning_before_onset_frac": float(np.mean(lead < 0)),
        "median_lead_steps": float(np.median(lead)),
        "mean_abs_lead_err_steps": float(np.mean(np.abs(lead))),
        "p10_lead_steps": float(np.percentile(lead, 10)),
        "p90_lead_steps": float(np.percentile(lead, 90)),
    }
    kw.update(step_stats)

    # alarm threshold sweep over PRE-onset steps only:
    # time-to-detect (chunk steps) vs alarms on normal windows
    alarm_rows = []
    for thr in [0.3, 0.5, 0.7, 0.9]:
        ttd = []
        for i in oi:
            k = int(f["ons"][i])
            pre = H[i][:k]
            hit = np.where(pre > thr)[0]
            ttd.append(int(hit[0] - k) if len(hit) else None)  # negative = before onset
        fp = (H[~ano.astype(bool)].max(axis=1) > thr).mean()
        alarm_rows.append({"threshold": thr, "fp_per_window": float(fp),
                           "median_ttd_steps": float(np.nanmedian(
                               [t for t in ttd if t is not None])) if any(t is not None for t in ttd) else None,
                           "frac_any_preonset_alarm": float(np.mean([t is not None for t in ttd]))})
    pd.DataFrame(alarm_rows).to_csv(TAB / "exp_007_alarm_sweep.csv", index=False)

    # figure: hazard curves for a few onset-in-window examples
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i in oi[:5]:
        ax.plot(H[i], lw=1.5, label=f"onset={int(f['ons'][i])}")
        ax.axvline(int(f["ons"][i]), ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("step (0.1 s)"); ax.set_ylabel("hazard (P onset within 5 s)")
    ax.legend(fontsize=7); ax.set_title("Early-warning hazard curves (test, onset-in-window)")
    fig.tight_layout(); fig.savefig(FIG / "exp_007_hazard_curves.png", dpi=150); plt.close(fig)

    out = {"early_warning_test": kw, "alarm_sweep": alarm_rows}
    with open(RES / "exp_007_early_warning.json", "w") as fh:
        json.dump(out, fh, indent=2)
    append({"experiment": "exp_007", "seed": 42, "task": "early_warning",
            "model": "tagn", "metrics": {k: v for k, v in kw.items() if v is not None}})
    print(json.dumps(kw, indent=2))


if __name__ == "__main__":
    main()
