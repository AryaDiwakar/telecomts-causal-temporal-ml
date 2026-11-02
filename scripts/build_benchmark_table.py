"""Build the consolidated benchmark table + machine-readable manifest.

Reads ``outputs/results_registry.json`` (the source of truth for registered
runs) and, per (task, model, variant), aggregates metrics across seeds as
mean ± std over seeds, flagging rows whose ``data_fingerprint`` differs from the
current data (mixed/vintage rows). Missing results stay explicitly MISSING.

Outputs:
  outputs/benchmark_manifest.json
  outputs/tables/benchmark_v3.csv
  outputs/tables/benchmark_v3.md

Usage:
    python -m scripts.build_benchmark_table
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "outputs" / "results_registry.json"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

MANIFEST_PROTOCOLS = [
    "P-det (pooled window; val-tuned threshold)",
    "P-rca-template (train-majority templates)",
    "P-rca-instance (fixed-k Recall@k / MRR / MAP)",
    "P-ew (hazard head)",
    "P-loro (recording-grouped 5-fold)",
]


def _manifest() -> dict:
    from src.utils.registry import code_revision, config_hash, data_fingerprint

    stats = json.loads((PROCESSED / "split_stats.json").read_text()) if (PROCESSED / "split_stats.json").exists() else {}
    templates = json.loads((PROCESSED / "rca_templates.json").read_text()) if (PROCESSED / "rca_templates.json").exists() else {}
    counts = stats.get("counts", {})
    return {
        "benchmark_version": "v3",
        "dataset": {
            "name": "TelecomTS",
            "track2": "instance-varying synthetic injection",
            "data_fingerprint": data_fingerprint(),
        },
        "config_hash": config_hash(),
        "code_revision": code_revision(),
        "splits": {
            "protocol": stats.get("protocol", "temporal_within_recording"),
            "ratios": stats.get("ratios"),
            "train": counts.get("train", {}).get("n"),
            "val": counts.get("val", {}).get("n"),
            "test": counts.get("test", {}).get("n"),
            "test_positives": counts.get("test", {}).get("n_anomalous"),
        },
        "preprocessing": {
            "scaler": "train-only robust z-score",
            "rca_templates_source": templates.get("rca_template_source", "unknown"),
            "rca_templates_criterion": templates.get("rca_template_criterion", "unknown"),
        },
        "protocols": MANIFEST_PROTOCOLS,
        "evaluation_rules": [
            "val-tuned operating thresholds",
            "provenance required for COMPLETE",
            "no test-set model/seed/weight selection",
            "shortcut baselines reported for RCA",
        ],
    }


def _flatten(metrics: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in metrics.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
        elif isinstance(v, (int, float)):
            out[key] = float(v)
    return out


def build() -> None:
    records = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []
    current_fp = _manifest()["dataset"]["data_fingerprint"]

    # Group by a key that includes the test-set size so different splitters do
    # not silently mix (finding C2: exp_009 was n=4768/223 for some seeds).
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        m = _flatten(r.get("metrics", {}))
        n = m.get("n")
        pos = m.get("n_positives")
        key = (r.get("task"), r.get("model"), r.get("variant"), n, pos)
        groups.setdefault(key, []).append(r)

    rows = []
    for (task, model, variant, n, pos), recs in sorted(
        groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), str(kv[0][3]))
    ):
        flat = [_flatten(r.get("metrics", {})) for r in recs]
        metric_keys = sorted({k for f in flat for k in f})
        row = {
            "task": task, "model": model, "variant": variant, "n_seeds": len(recs),
            "test_n": int(n) if n is not None else "MISSING",
            "n_positives": int(pos) if pos is not None else "MISSING",
        }
        fps = {r.get("data_fingerprint", "MISSING") for r in recs}
        row["provenance"] = "current" if fps == {current_fp} else ("legacy" if fps == {"MISSING"} else "mixed")
        for m in metric_keys:
            vals = [f[m] for f in flat if m in f]
            if not vals:
                continue
            row[m] = f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}" if len(vals) > 1 else f"{vals[0]:.4f}"
        rows.append(row)

    TABLES.mkdir(parents=True, exist_ok=True)

    base_cols = ["task", "model", "variant", "n_seeds", "test_n", "n_positives", "provenance"]

    # CSV (flat, union of all metric columns)
    import csv

    metric_cols = sorted({k for r in rows for k in r if k not in base_cols})
    with open(TABLES / "benchmark_v3.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=base_cols + metric_cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Markdown
    lines = [
        "# Consolidated benchmark table — v3", "",
        "Aggregated from `outputs/results_registry.json` (mean ± std over seeds).",
        "Rows are keyed by task/model/variant **and test-set size**, so results on",
        "different splits never mix. `provenance`: `current` (matches live data",
        "fingerprint), `legacy` (pre-provenance registry entry), `mixed`.", "",
    ]
    for task in sorted({r["task"] for r in rows}, key=str):
        task_rows = [r for r in rows if r["task"] == task]
        metrics = [m for m in metric_cols if any(m in r for r in task_rows)]
        metrics = [m for m in metrics if m not in ("n", "n_positives")]
        header = ["model", "variant", "n_seeds", "test_n", "n_pos", "provenance"] + metrics
        lines.append(f"## {task}")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for r in task_rows:
            cells = [str(r["model"]), str(r.get("variant") or ""), str(r["n_seeds"]),
                     str(r["test_n"]), str(r["n_positives"]), r["provenance"]]
            cells += [str(r.get(m, "MISSING")) for m in metrics]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    (TABLES / "benchmark_v3.md").write_text("\n".join(lines))
    manifest = _manifest()
    (ROOT / "outputs" / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"wrote {TABLES/'benchmark_v3.md'} ({len(rows)} rows), manifest, csv")
    print(f"benchmark_version=v3 data_fingerprint={manifest['dataset']['data_fingerprint']} "
          f"test={manifest['splits']['test']} pos={manifest['splits']['test_positives']}")


if __name__ == "__main__":
    build()
