"""Chronological, leakage-safe preprocessing for TelecomTS.

Split protocol (see docs/data_leakage.md):
  * Every recording (raw file) is split along its own timeline at 70% / 85%.
  * Chunks that straddle a boundary are dropped (embargo = one full chunk).
  * Anomalous recordings: overlapping burst-clusters are atomic; a cluster that
    straddles a boundary is dropped entirely so one fault event never spans splits.
  * Recordings whose entire anomalous content is a single cluster covering the
    whole timeline (jammer) are assigned atomically by chronological rank
    within their condition stratum.
  * Normal recordings: chunk-level chronological split (chunks never share raw
    timesteps across splits because straddling chunks are dropped).
  * Scaling statistics are fit on TRAIN chunks only.

Outputs (data/processed/):
  split_indices.npz     train/val/test chunk-id arrays
  scaler.json           per-channel location/scale fit on train only
  targets.npz           y_anomaly, y_type, y_affected, y_onset, y_ew_label, masks
  rca_templates.json    anomaly_type -> affected-KPI multi-hot template
  split_stats.json      human-readable split audit

Usage:
    python -m src.preprocessing.prepare
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
CONFIG = ROOT / "configs" / "data.yaml"

CHUNK_STEPS = 128
EW_HORIZON_DEFAULT = 50


def load_config() -> dict:
    with open(CONFIG) as fh:
        return yaml.safe_load(fh)


def cluster_chunks(sub: pd.DataFrame) -> pd.Series:
    """Group overlapping chunks (sorted by time) into burst clusters.

    Returns a Series indexed by chunk_id (so it can be assigned to any
    ordering of ``sub`` via pandas index alignment).
    """
    sub_sorted = sub.sort_values("start_time")
    clusters = np.zeros(len(sub_sorted), dtype=np.int64)
    cid = 0
    prev_end = None
    for i, row in enumerate(sub_sorted.itertuples()):
        if prev_end is not None and row.start_time > prev_end:
            cid += 1
        clusters[i] = cid
        prev_end = row.end_time if prev_end is None else max(prev_end, row.end_time)
    return pd.Series(clusters, index=sub_sorted.index)


def assign_splits(meta: pd.DataFrame, cfg: dict) -> tuple[dict[str, np.ndarray], dict]:
    sp = cfg["split"]
    ratios = (sp["train_ratio"], sp["val_ratio"], sp["test_ratio"])
    assert abs(sum(ratios) - 1.0) < 1e-9

    assignments: dict[int, str] = {}
    audit: dict[str, dict] = {}

    # --- jammer recordings: single continuous fault -> assign whole recording,
    #     chronologically within stratum (File -> train, Twitch -> val,
    #     YouTube -> test) so one fault event never spans splits ---
    anomalous_recordings = (
        meta[meta.anomaly].groupby("recording")["chunk_id"].count().index.tolist()
    )
    atomic_recs = [r for r in anomalous_recordings if "/jammer/" in r]
    strata: dict[str, list[str]] = {}
    for rec in atomic_recs:
        strata.setdefault("jammer", []).append(rec)
    for stratum, recs in strata.items():
        recs_sorted = sorted(
            recs,
            key=lambda r: meta[meta.recording == r].start_time.min(),
        )
        n = len(recs_sorted)
        if n >= 3:
            n_train = max(1, int(round(ratios[0] * n)))
            n_val = max(1, int(round(ratios[1] * n)))
            n_test = n - n_train - n_val
            if n_val + n_test >= n:
                n_val, n_test = 1, 1
                n_train = n - 2
            elif n_test < 1:
                n_train -= 1
                n_test = 1
            n_train = n - n_val - n_test
        else:
            n_train, n_val, n_test = n, 0, 0
        for i, r in enumerate(recs_sorted):
            split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            for cid in meta.loc[meta.recording == r, "chunk_id"]:
                assignments[int(cid)] = split
            audit.setdefault("atomic_recordings", {})[r] = split

    # --- remaining recordings: time-based split ---
    done = set(a for a in assignments)
    for rec, sub in meta.groupby("recording"):
        sub = sub.sort_values("start_time")
        if all(int(c) in assignments for c in sub.chunk_id):
            continue
        t0 = sub.start_time.min()
        t1 = sub.end_time.max()
        span = (t1 - t0).total_seconds()
        if span <= 0:
            raise ValueError(f"non-positive span for {rec}")
        t_train = t0 + pd.Timedelta(seconds=span * ratios[0])
        t_val = t0 + pd.Timedelta(seconds=span * (ratios[0] + ratios[1]))

        is_anomalous_recording = bool(sub.anomaly.any())
        if is_anomalous_recording:
            cl = cluster_chunks(sub)
            tmp = sub.assign(cluster=cl)
            for cid, g in tmp.groupby("cluster"):
                c_start, c_end = g.start_time.min(), g.end_time.max()
                if c_end <= t_train:
                    split = "train"
                elif c_start >= t_train and c_end <= t_val:
                    split = "val"
                elif c_start >= t_val:
                    split = "test"
                else:
                    split = None  # straddling cluster -> drop
                if split:
                    for i in g.chunk_id:
                        assignments[int(i)] = split
            dropped = int((~tmp.chunk_id.isin(assignments)).sum())
            if dropped:
                audit.setdefault("dropped_clusters", {})[rec] = dropped
        else:
            for row in sub.itertuples():
                if row.end_time <= t_train:
                    split = "train"
                elif row.start_time >= t_train and row.end_time <= t_val:
                    split = "val"
                elif row.start_time >= t_val:
                    split = "test"
                else:
                    split = None  # straddling chunk -> drop (embargo)
                if split:
                    assignments[int(row.chunk_id)] = split
            dropped = int((~sub.chunk_id.isin(assignments)).sum())
            if dropped:
                audit.setdefault("dropped_chunks", {})[rec] = dropped

    splits = {
        name: np.array(sorted([c for c, s in assignments.items() if s == name]), dtype=np.int64)
        for name in ("train", "val", "test")
    }
    return splits, audit


def fit_scaler(X: np.ndarray, train_idx: np.ndarray, method: str) -> dict:
    Xt = X[train_idx]
    if method == "zscore":
        loc = Xt.mean(axis=(0, 1))
        scale = Xt.std(axis=(0, 1))
    elif method == "robust_zscore":
        loc = np.median(Xt, axis=(0, 1))
        q75, q25 = np.percentile(Xt, [75, 25], axis=(0, 1))
        scale = (q75 - q25) / 1.349
    elif method == "minmax":
        loc = Xt.min(axis=(0, 1))
        scale = Xt.max(axis=(0, 1)) - loc
    else:
        raise ValueError(method)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return {"method": method, "loc": loc.tolist(), "scale": scale.tolist()}


def build_targets(meta: pd.DataFrame, cfg: dict, kpis: list[str]) -> dict:
    n = len(meta)
    horizon = cfg["windowing"].get("forecast_horizon", EW_HORIZON_DEFAULT)
    kpi_index = {k: i for i, k in enumerate(kpis)}

    y_anomaly = meta.anomaly.to_numpy(dtype=np.int64)
    onset = meta.onset_step.to_numpy(dtype=np.int64)
    offset = meta.offset_step.to_numpy(dtype=np.int64)

    type_names = sorted(meta.loc[meta.anomaly, "anomaly_type"].unique())
    type_map = {t: i for i, t in enumerate(type_names)}
    y_type = np.full(n, -1, dtype=np.int64)
    for i, row in enumerate(meta.itertuples()):
        if row.anomaly:
            y_type[i] = type_map[row.anomaly_type]

    y_affected = np.zeros((n, len(kpis)), dtype=np.float32)
    templates: dict[str, list[str]] = {}
    for i, row in enumerate(meta.itertuples()):
        if row.anomaly:
            aff = json.loads(row.affected_kpis)
            templates[row.anomaly_type] = aff
            for k in aff:
                if k in kpi_index:
                    y_affected[i, kpi_index[k]] = 1.0

    # early-warning hazard labels: 1 at pre-onset steps where onset within horizon
    y_ew = np.zeros((n, CHUNK_STEPS), dtype=np.float32)
    ew_mask = np.ones((n, CHUNK_STEPS), dtype=np.float32)
    for i in range(n):
        if not y_anomaly[i]:
            continue
        k = onset[i]
        if k <= 0:
            ew_mask[i, :] = 0.0  # no pre-onset region -> no EW supervision
        else:
            lo = max(0, k - horizon)
            y_ew[i, lo:k] = 1.0
            ew_mask[i, k:] = 0.0  # post-onset steps excluded from EW loss

    return {
        "y_anomaly": y_anomaly,
        "y_type": y_type,
        "y_affected": y_affected,
        "onset": onset,
        "offset": offset,
        "y_ew": y_ew,
        "ew_mask": ew_mask,
    }, {"type_map": type_map, "rca_templates": templates, "ew_horizon": horizon}


def main() -> None:
    cfg = load_config()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    X = np.load(INTERIM / "chunks_X.npy")
    meta = pd.read_parquet(INTERIM / "chunks_meta.parquet")
    with open(INTERIM / "kpi_names.json") as fh:
        info = json.load(fh)
    kpis = info["numeric_kpis"]

    splits, audit = assign_splits(meta, cfg)
    train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]

    scaler = fit_scaler(X, train_idx, cfg["windowing"]["normalization_method"])
    targets, label_info = build_targets(meta, cfg, kpis)

    np.savez_compressed(
        PROCESSED / "split_indices.npz",
        train=train_idx, val=val_idx, test=test_idx,
    )
    np.savez_compressed(PROCESSED / "targets.npz", **targets)
    with open(PROCESSED / "scaler.json", "w") as fh:
        json.dump(scaler, fh, indent=2)
    with open(PROCESSED / "rca_templates.json", "w") as fh:
        json.dump(label_info, fh, indent=2)

    stats: dict = {
        "protocol": cfg["split"]["protocol"],
        "ratios": [cfg["split"][k] for k in ("train_ratio", "val_ratio", "test_ratio")],
        "counts": {},
        "audit": audit,
    }
    for name, idx in splits.items():
        sub = meta.iloc[idx]
        stats["counts"][name] = {
            "n": int(len(idx)),
            "n_anomalous": int(sub.anomaly.sum()),
            "n_recordings": int(sub.recording.nunique()),
            "types": sub.loc[sub.anomaly, "anomaly_type"].value_counts().to_dict(),
            "onset_gt0": int((sub.anomaly & (sub.onset_step > 0)).sum()),
        }
    stats["dropped_total"] = int(len(meta) - sum(len(v) for v in splits.values()))
    with open(PROCESSED / "split_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    print(json.dumps(stats["counts"], indent=2))
    print("dropped:", stats["dropped_total"])
    print("audit:", json.dumps(audit, indent=2)[:2000])


if __name__ == "__main__":
    main()
