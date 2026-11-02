#!/usr/bin/env python3
"""Figure: input-free type-oracle vs model RCA, and detection, by subset.

Left panel: top-k RCA Jaccard on template vs instance-varying test windows for
the input-free type-oracle, the exp_002 baseline, and the exp_012 GRL model.
Right panel: detection ROC-AUC / PR-AUC / F1-max of the baseline per subset.

Writes outputs/figures/exp/shortcut_rca_subsets.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RES = ROOT / "outputs" / "results"
FIG = ROOT / "outputs" / "figures" / "exp"


def main() -> None:
    audit = json.loads((RES / "shortcut_audit.json").read_text())
    sub = json.loads((RES / "subset_breakdown.json").read_text())

    oracle = {s["subset"]: s["test_type_oracle_jaccard"] for s in audit["test_subsets"]}
    subsets = ["template", "instance_varying"]
    labels = ["Template (shipped)", "Instance-varying (Track 2)"]
    oracle_vals = [oracle["template_test(shipped)"], oracle["instance_varying_test(track2)"]]
    base_vals = [sub["tagn_baseline"][s]["rca_topk_jaccard"] for s in subsets]
    grl_vals = [sub["tagn_grl"][s]["rca_topk_jaccard"] for s in subsets]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    x = np.arange(len(subsets))
    w = 0.26
    ax1.bar(x - w, oracle_vals, w, label="input-free type-oracle", color="#888888")
    ax1.bar(x, base_vals, w, label="TAGN baseline", color="#1f77b4")
    ax1.bar(x + w, grl_vals, w, label="TAGN + GRL", color="#d62728")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("top-k RCA Jaccard")
    ax1.set_title("RCA: input-free oracle vs model")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8)
    for xi, vv in zip(x - w, oracle_vals):
        ax1.text(xi, vv + 0.02, f"{vv:.2f}", ha="center", fontsize=8)
    for xi, vv in zip(x, base_vals):
        ax1.text(xi, vv + 0.02, f"{vv:.2f}", ha="center", fontsize=8)
    for xi, vv in zip(x + w, grl_vals):
        ax1.text(xi, vv + 0.02, f"{vv:.2f}", ha="center", fontsize=8)

    metrics = ["roc_auc", "pr_auc", "f1_max"]
    mnames = ["ROC-AUC", "PR-AUC", "F1-max"]
    x2 = np.arange(len(metrics))
    w2 = 0.35
    tvals = [sub["tagn_baseline"]["template"]["detection"][m] for m in metrics]
    gvals = [sub["tagn_baseline"]["instance_varying"]["detection"][m] for m in metrics]
    ax2.bar(x2 - w2 / 2, tvals, w2, label="template", color="#2ca02c")
    ax2.bar(x2 + w2 / 2, gvals, w2, label="instance-varying", color="#ff7f0e")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(mnames)
    ax2.set_ylabel("score")
    ax2.set_title("Detection per anomaly family (TAGN baseline)")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    for xi, vv in zip(x2 - w2 / 2, tvals):
        ax2.text(xi, vv + 0.02, f"{vv:.2f}", ha="center", fontsize=8)
    for xi, vv in zip(x2 + w2 / 2, gvals):
        ax2.text(xi, vv + 0.02, f"{vv:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "shortcut_rca_subsets.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
