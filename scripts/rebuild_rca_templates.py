"""Rebuild ``data/processed/rca_templates.json`` from train-only majority vote.

Fixes finding G1 in ``docs/benchmark_audit_v3.md``: the previous artifact stored
the *last* anomalous chunk's KPI set over the whole dataset (val/test included),
which is both label-contaminated and non-representative for instance-varying
types. This script only rewrites ``rca_templates.json`` (archiving the old file);
it does **not** touch ``split_indices.npz``, ``targets.npz`` or ``scaler.json``.

Usage:
    python -m scripts.rebuild_rca_templates
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
ARCHIVE = ROOT / "outputs" / "archive"


def main() -> None:
    meta = pd.read_parquet(INTERIM / "chunks_meta.parquet")
    idx = json.loads((PROCESSED / "rca_templates.json").read_text())
    import numpy as np

    npz = np.load(PROCESSED / "split_indices.npz")
    train_idx = npz["train"]

    source = set(int(i) for i in train_idx)
    counts: dict[str, Counter] = {}
    for i in sorted(source):
        row = meta.iloc[i]
        if bool(row.anomaly):
            key = tuple(sorted(json.loads(row.affected_kpis)))
            counts.setdefault(row.anomaly_type, Counter())[key] += 1

    templates: dict[str, list[str]] = {}
    provenance: dict[str, dict] = {}
    for tname, counter in counts.items():
        best, n = max(counter.items(), key=lambda kv: (kv[1], kv[0]))
        templates[tname] = list(best)
        provenance[tname] = {
            "chosen_n": int(n),
            "n_train_anomalous": int(sum(counter.values())),
            "n_distinct_sets": int(len(counter)),
            "is_train_majority": True,
        }

    out = {
        "type_map": idx["type_map"],          # preserved (must match targets.npz)
        "ew_horizon": idx.get("ew_horizon"),
        "rca_templates": templates,
        "rca_template_source": "train",
        "rca_template_criterion": "majority",
        "provenance": provenance,
    }

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    old = PROCESSED / "rca_templates.json"
    if old.exists():
        shutil.copy2(old, ARCHIVE / "rca_templates_lastwritewins.json")
    old.write_text(json.dumps(out, indent=2))

    print("archived old -> outputs/archive/rca_templates_lastwritewins.json")
    for t in sorted(templates):
        p = provenance[t]
        print(f"{t}: size={len(templates[t])} chosen_n={p['chosen_n']}/{p['n_train_anomalous']} "
              f"distinct={p['n_distinct_sets']}")


if __name__ == "__main__":
    main()
