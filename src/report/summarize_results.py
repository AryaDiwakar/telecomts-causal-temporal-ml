"""Aggregate per-seed exp_002 results into summary stats + report tables.

Usage:
    python -m src.report.summarize_results
"""

from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "outputs" / "results"
TAB = ROOT / "outputs" / "tables"

AD_METRICS = ["roc_auc", "pr_auc", "f1_max", "precision", "recall",
              "tpr_at_fpr_0.01", "tpr_at_fpr_0.05"]
TYPE_METRICS = ["accuracy", "macro_f1_seen", "micro_f1"]
RCA_METRICS = ["type_template_jaccard", "topk_shift_jaccard"]
EW_METRICS = ["n_chunks", "median_lag_steps", "warning_before_onset_frac"]


def pick(metrics: dict, keys: list[str]) -> dict:
    return {k: metrics.get(k) for k in keys if k in metrics}


def main() -> None:
    files = sorted(glob.glob(str(RES / "exp_002_tagn_seed*.json")))
    if not files:
        print("no exp_002 seed files yet -> RESULT_PENDING")
        return
    rows = []
    for fp in files:
        seed = Path(fp).stem.split("_")[-1].removeprefix("seed")
        d = json.load(open(fp))
        row = {"seed": np.int64(seed)}
        row.update(pick(d["test"].get("anomaly", {}), AD_METRICS))
        row.update(pick(d["test"].get("type", {}), TYPE_METRICS))
        row.update(pick(d["test"].get("rca", {}), RCA_METRICS))
        row.update(pick(d["test"].get("early_warning", {}), EW_METRICS))
        row["best_val_pr_auc"] = d.get("best_val_pr_auc")
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("seed")

    summary = pd.DataFrame(
        {
            "metric": df.columns[1:],
            "mean": [df[c].mean() for c in df.columns[1:]],
            "std": [df[c].std() for c in df.columns[1:]],
            "min": [df[c].min() for c in df.columns[1:]],
            "max": [df[c].max() for c in df.columns[1:]],
        }
    )
    df.to_csv(TAB / "exp_002_multi_seed_raw.csv", index=False)
    summary.to_csv(TAB / "exp_002_multi_seed_stats.csv", index=False)
    print(df.to_string(index=False))
    print("\nsummary:\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
