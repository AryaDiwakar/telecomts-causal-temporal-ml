#!/usr/bin/env python3
"""Generate Track 2 synthetic anomaly recordings and run the validation gate.

Usage:
    python -m scripts.generate_synthetic --count 12 --seed 2026 [--dry-run]
    python -m scripts.generate_synthetic --list-sources

The injector writes recording dirs under configs/synthetic.yaml `track2.out_root`
(auto-discovered by src/data/loader.py), plus a manifest + validation report in
data/processed/synthetic_gen/.

Validation gate (docs/data_acquisition_plan.md §2.3): confirms, per generated
recording, that affected KPIs deviate from the pre-injection baseline in the
intended direction, that boundaries/NaN are respected, and that only
single-type anomalous windows are emitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.data.synthetic_inject import (
    InjectedRecording,
    inject_recording,
    load_config,
    write_recording,
)

ROOT = Path(__file__).resolve().parents[1]


def list_sources(cfg: dict) -> list[Path]:
    src = ROOT / cfg["track2"]["source_root"]
    return sorted(src.rglob("raw/metrics.csv"))


def choose_sources(cfg: dict, count: int, seed: int) -> list[tuple[Path, Path]]:
    tr = cfg["track2"]
    src = ROOT / tr["source_root"]
    max_rows = int(tr.get("max_source_rows", 10**9))
    all_metric = sorted(src.rglob("raw/metrics.csv"))
    eligible = [
        p for p in all_metric
        if sum(1 for _ in p.open()) - 1 <= max_rows
    ]
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(eligible), size=min(count, len(eligible)),
                        replace=False)
    return [(eligible[int(i)], eligible[int(i)].relative_to(src))
            for i in np.sort(picked)]


def direction_accuracy(rec: InjectedRecording, cfg: dict) -> dict:
    """Window-level check: transformed vs pre-injection mean sign for each
    affected KPI must match the sampled parameter direction (non-sinusoidal).
    Sinusoidal effects (expected_dir == 0) are checked for detectability only.
    """
    tr = cfg["track2"]
    wlen, stride = tr["window_length"], tr["stride"]
    boundaries = tr["boundaries"]
    num, x = rec.raw_num, rec.raw_x
    kpi_idx = {k: i for i, k in enumerate(
        ["RSRP", "DL_BLER", "DL_MCS", "UL_BLER", "UL_MCS", "UL_NPRB", "UL_SNR",
         "TX_Bytes", "RX_Bytes", "Estimated_UL_Buffer", "PRBs_DL_Current",
         "PRBs_UL_Current", "PRB_Utilization_DL", "PRB_Utilization_UL",
         "UL_NumberOfPackets", "DL_NumberOfPackets"])}

    matched = mismatched = n_dir = n_var = n_detect = 0
    detect_fail = unobservable = 0
    for tgt in rec.per_event_targets:
        w0 = tgt["window"] * stride
        k = kpi_idx[tgt["kpi"]]
        # measure only the event-overlap portion of the window, which is the
        # true observable (a window may extend beyond its anomaly's interval)
        o0 = max(w0, int(tgt.get("ev_start", tgt["window"] * stride)))
        o1 = min(w0 + wlen - 1, int(tgt.get("ev_end", tgt["window"] * stride + wlen - 1)))
        if o1 < o0:
            unobservable += 1
            continue
        pre = num[o0 : o1 + 1, k]
        post = x[o0 : o1 + 1, k]
        diff = float(np.mean(post) - np.mean(pre))
        lo_k, hi_k = boundaries[tgt["kpi"]]
        at_hi = bool(pre.max() >= hi_k - 1e-6)
        at_lo = bool(pre.min() <= lo_k + 1e-6)
        # all-zero base within the event's segment -> scale/exp transform is
        # invisible (e.g. DL_BLER all zeros); not countable
        if tgt.get("base_inert"):
            unobservable += 1
            continue
        # windows whose base is constant zero cannot show a sign under a
        # scale/growth transform (e.g. all-zero DL_BLER) -> not countable
        if pre.std() < 1e-12 and abs(diff) < 1e-12 and post.std() < 1e-12:
            unobservable += 1
            continue
        # KPI already pinned at its bound toward the transform direction
        # (saturation) -> the change is clipped, not countable either
        if (tgt["expected_dir"] > 0 and at_hi) or (tgt["expected_dir"] < 0 and at_lo):
            unobservable += 1
            continue
        if tgt["expected_dir"] != 0:
            n_dir += 1
            if np.sign(diff) == tgt["expected_dir"]:
                matched += 1
            else:
                mismatched += 1
        else:
            n_var += 1
            sigma = float(pre.std())
            if sigma > 1e-12 and abs(diff) > 0.1 * sigma:
                n_detect += 1
            elif abs(diff) > 1e-3:
                n_detect += 1
            else:
                detect_fail += 1

    acc = matched / n_dir if n_dir else float("nan")
    return {
        "direction_targets": n_dir,
        "direction_matched": matched,
        "direction_mismatched": mismatched,
        "direction_accuracy": round(acc, 4),
        "unobservable_base": unobservable,
        "sinusoidal_targets": n_var,
        "sinusoidal_detectable": n_detect,
        "sinusoidal_not_detectable": detect_fail,
    }


def bounds_check(rec: InjectedRecording, cfg: dict) -> dict:
    tr = cfg["track2"]
    boundaries = tr["boundaries"]
    ok = True
    for j, k in enumerate(
        ["RSRP", "DL_BLER", "DL_MCS", "UL_BLER", "UL_MCS", "UL_NPRB", "UL_SNR",
         "TX_Bytes", "RX_Bytes", "Estimated_UL_Buffer", "PRBs_DL_Current",
         "PRBs_UL_Current", "PRB_Utilization_DL", "PRB_Utilization_UL",
         "UL_NumberOfPackets", "DL_NumberOfPackets"]):
        lo, hi = boundaries[k]
        c = rec.raw_x[:, j]
        if np.isfinite(lo) and c.min() < lo - 1e-6:
            ok = False
        if np.isfinite(hi) and c.max() > hi + 1e-6:
            ok = False
    return {
        "bounds_respected": bool(ok),
        "nan_inf": int(np.isnan(rec.raw_x).sum() + np.isinf(rec.raw_x).sum()),
    }


def per_type_counts(rec: InjectedRecording) -> dict:
    counts: dict[str, int] = {}
    for ev in rec.events:
        counts[ev.anomaly_type] = counts.get(ev.anomaly_type, 0) + 1
    return counts


def run() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-sources", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    tr = cfg["track2"]

    if args.list_sources:
        for p in list_sources(cfg):
            print(p.relative_to(ROOT))
        return

    out_root = ROOT / tr["out_root"]
    pairs = choose_sources(cfg, args.count, args.seed)

    manifest: list[dict] = []
    reports: dict[str, dict] = {}
    seed = args.seed

    for i, (csv, rel) in enumerate(pairs):
        rng = np.random.default_rng(seed + i * 7919)
        rec = inject_recording(csv, cfg, rng, rel)
        rec_dir = out_root / rel.parent.parent  # <zone>/<app>
        if not args.dry_run:
            write_recording(rec_dir, rec, csv)
        report = {
            "source": str(rel),
            "out_dir": str(rec_dir.relative_to(ROOT)),
            "events": len(rec.events),
            "type_counts": per_type_counts(rec),
            "anomalous_windows": len(rec.chunks),
            "dropped_all_normal": rec.windows_dropped_all_normal,
            "dropped_multi_type": rec.windows_dropped_multi_type,
            **direction_accuracy(rec, cfg),
            **bounds_check(rec, cfg),
        }
        reports[str(i)] = report
        manifest.append(report)
        print(json.dumps({k: v for k, v in report.items()
                          if k not in ("type_counts",)}, ensure_ascii=False)[:400])

    accs = [r["direction_accuracy"] for r in manifest
        if isinstance(r["direction_accuracy"], (int, float))
        and np.isfinite(r["direction_accuracy"])]
    summary = {
        "seed": seed,
        "n_recordings": len(pairs),
        "n_anomalous_windows": sum(r["anomalous_windows"] for r in manifest),
        "n_events": sum(r["events"] for r in manifest),
        "mean_direction_accuracy": round(np.mean(accs), 4) if accs else None,
        "bounds_respected_all": all(r["bounds_respected"] for r in manifest),
        "nan_inf_total": sum(r["nan_inf"] for r in manifest),
        "multi_type_dropped_total": sum(r["dropped_multi_type"] for r in manifest),
    }

    out = ROOT / "data" / "processed" / "synthetic_gen"
    out.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print("\nSummary:", json.dumps(summary, indent=2))
        print(f"Manifest: {out / 'manifest.json'}")


if __name__ == "__main__":
    sys.exit(run())