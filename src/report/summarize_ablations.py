"""Summarize exp_003 ablation JSON files relative to the full variant."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "outputs" / "results"
TAB = ROOT / "outputs" / "tables"


def main() -> None:
    rows = []
    for fp in sorted(glob.glob(str(RES / "exp_003_ablation_*.json"))):
        variant = Path(fp).stem.removeprefix("exp_003_ablation_")
        d = json.loads(Path(fp).read_text())
        a = d["test"].get("anomaly", {})
        rows.append({
            "variant": variant,
            "epochs": len(d.get("history", [])),
            "best_val_pr_auc": d.get("best_val_pr_auc"),
            "test_pr_auc": a.get("pr_auc"),
            "test_roc_auc": a.get("roc_auc"),
            "type_accuracy": d["test"].get("type", {}).get("accuracy"),
            "rca_jaccard": d["test"].get("rca", {}).get("type_template_jaccard"),
            "early_warning_before": d["test"].get("early_warning", {}).get("warning_before_onset_frac"),
        })
    if not rows:
        print("RESULT_PENDING: no ablation files")
        return
    df = pd.DataFrame(rows)
    full = df.loc[df.variant == "full", "test_pr_auc"]
    if len(full):
        df["delta_pr_vs_full"] = df.test_pr_auc - float(full.iloc[0])
    df.to_csv(TAB / "exp_003_ablation_summary.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
