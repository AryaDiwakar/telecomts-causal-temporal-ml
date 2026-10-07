"""Inspect the raw TelecomTS dataset and emit schema + summary artifacts.

Usage:
    python -m src.data.inspect_dataset
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "TelecomTS"
INTERIM = ROOT / "data" / "interim"
DOCS = ROOT / "docs"

EXPECTED_TOP_KEYS = [
    "start_time",
    "end_time",
    "sampling_rate",
    "KPIs",
    "description",
    "anomalies",
    "statistics",
    "labels",
    "QnA",
]


def discover_files() -> dict[str, list[Path]]:
    files = sorted(RAW.rglob("chunked.jsonl"))
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        rel = f.relative_to(RAW)
        # group by condition: e.g. anomalous/synthetic/Zone_A/File
        parts = list(rel.parts)
        group = "/".join(parts[:-2])
        groups[group].append(f)
    return dict(groups)


def stream_records(paths: list[Path]):
    for p in paths:
        with open(p) as fh:
            for line_no, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield p, line_no, rec


def main() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    groups = discover_files()
    n_files = sum(len(v) for v in groups.values())

    top_key_counts: Counter = Counter()
    kpi_names: list[str] | None = None
    kpi_types: dict[str, Counter] = defaultdict(Counter)
    anomaly_types: Counter = Counter()
    affected_kpi_counter: Counter = Counter()
    n_affected_hist: Counter = Counter()
    labels_values: dict[str, Counter] = defaultdict(Counter)
    sampling_rates: Counter = Counter()
    stat_keys: Counter = Counter()
    qna_categories: Counter = Counter()

    total = 0
    n_anomalous = 0
    n_missing_kpi_values = 0
    n_kpi_entries = 0
    duration_start: list[int] = []
    duration_end: list[int] = []
    duration_len: list[int] = []
    onset_gt0 = 0
    start_times: list[pd.Timestamp] = []
    per_file: list[dict] = []
    qa_total = 0
    chunk_hashes: Counter = Counter()
    duplicate_lines = 0

    for group, paths in sorted(groups.items()):
        f_total = 0
        f_anom = 0
        f_first_ts = None
        f_last_ts = None
        prev_end = None
        gaps = 0
        overlaps = 0
        for p, _, rec in stream_records(paths):
            total += 1
            f_total += 1
            for k in rec:
                top_key_counts[k] += 1

            sampling_rates[rec.get("sampling_rate")] += 1

            kpis = rec.get("KPIs", {})
            if kpi_names is None:
                kpi_names = list(kpis.keys())
            for name, vals in kpis.items():
                kpi_types[name][type(vals[0]).__name__ if vals else "empty"] += 1
                for v in vals:
                    n_kpi_entries += 1
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        n_missing_kpi_values += 1

            ts = pd.Timestamp(rec["start_time"])
            te = pd.Timestamp(rec["end_time"])
            start_times.append(ts)
            if f_first_ts is None:
                f_first_ts = ts
            f_last_ts = te
            if prev_end is not None:
                delta = (ts - prev_end).total_seconds()
                if abs(delta) < 1e-9:
                    pass  # contiguous
                elif delta > 0:
                    gaps += 1
                else:
                    overlaps += 1
            prev_end = te

            anom = rec.get("anomalies", {})
            if anom.get("exists"):
                n_anomalous += 1
                f_anom += 1
                anomaly_types[anom.get("type")] += 1
                aff = anom.get("affected_kpis", []) or []
                n_affected_hist[len(aff)] += 1
                for k in aff:
                    affected_kpi_counter[k] += 1
                d = anom.get("anomaly_duration", {}) or {}
                s, e = d.get("start"), d.get("end")
                if s is not None and e is not None:
                    duration_start.append(int(s))
                    duration_end.append(int(e))
                    duration_len.append(int(e) - int(s) + 1)
                    if int(s) > 0:
                        onset_gt0 += 1

            for k, v in (rec.get("labels") or {}).items():
                labels_values[k][str(v)] += 1

            for k in (rec.get("statistics") or {}):
                stat_keys[k] += 1

            qn = rec.get("QnA") or {}
            for cat, items in qn.items():
                qna_categories[cat] += len(items)
                qa_total += len(items)

            key = (group, rec.get("start_time"), rec.get("end_time"))
            h = hash(json.dumps(rec.get("KPIs"), sort_keys=True))
            if (key, h) in chunk_hashes:
                duplicate_lines += 1
            chunk_hashes[(key, h)] += 1

        per_file.append(
            {
                "group": group,
                "n_chunks": f_total,
                "n_anomalous": f_anom,
                "first_start": str(f_first_ts),
                "last_end": str(f_last_ts),
                "internal_gaps": gaps,
                "internal_overlaps": overlaps,
            }
        )

    # cross-file timestamp overlap check
    per_file_sorted = sorted(per_file, key=lambda r: r["first_start"])
    cross_overlaps = 0
    for i in range(len(per_file_sorted)):
        for j in range(i + 1, len(per_file_sorted)):
            a, b = per_file_sorted[i], per_file_sorted[j]
            if b["first_start"] < a["last_end"]:
                cross_overlaps += 1

    summary = {
        "n_files": n_files,
        "n_condition_groups": len(groups),
        "total_chunks": total,
        "n_anomalous_chunks": n_anomalous,
        "n_normal_chunks": total - n_anomalous,
        "top_level_keys": dict(top_key_counts),
        "sampling_rates_hz": dict(sampling_rates),
        "kpi_names": kpi_names,
        "kpi_value_types": {k: dict(v) for k, v in kpi_types.items()},
        "anomaly_type_counts": dict(anomaly_types.most_common()),
        "affected_kpi_counts": dict(affected_kpi_counter.most_common()),
        "n_affected_kpis_hist": dict(sorted(n_affected_hist.items())),
        "anomaly_duration": {
            "n_with_duration": len(duration_start),
            "onset_gt0": onset_gt0,
            "start_mean": float(np.mean(duration_start)) if duration_start else None,
            "len_mean": float(np.mean(duration_len)) if duration_len else None,
            "len_min": int(np.min(duration_len)) if duration_len else None,
            "len_max": int(np.max(duration_len)) if duration_len else None,
            "len_median": float(np.median(duration_len)) if duration_len else None,
        },
        "label_values": {k: dict(v) for k, v in labels_values.items()},
        "statistics_keys": dict(stat_keys),
        "qna_counts": dict(qna_categories),
        "qna_total": qa_total,
        "missing_kpi_values": n_missing_kpi_values,
        "kpi_entries_total": n_kpi_entries,
        "duplicate_chunks": duplicate_lines,
        "timestamp_range": [
            str(min(start_times)) if start_times else None,
            str(max(start_times)) if start_times else None,
        ],
        "per_file": per_file,
        "cross_file_timestamp_overlap_pairs": cross_overlaps,
    }

    with open(INTERIM / "dataset_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"chunks={total} anomalous={n_anomalous} files={n_files}")
    print(f"anomaly types={len(anomaly_types)}")
    print(f"missing kpi values={n_missing_kpi_values}/{n_kpi_entries}")
    print(f"duplicate chunks={duplicate_lines}")
    print(f"cross-file overlap pairs={cross_overlaps}")
    print(f"onset>0 chunks={onset_gt0}")
    print(f"wrote {INTERIM / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
