"""Per-chunk statistical features for classical baselines.

Features are computed on the train-scaled channels, so the feature space is
comparable across models and includes only train-fit statistics (no leakage).

Usage:
    from src.features.classical import extract_features
"""

from __future__ import annotations

import numpy as np

METRICS = ["mean", "std", "min", "max", "median", "p10", "p90", "slope", "energy", "zero_frac", "iqr"]


def _slope(x: np.ndarray) -> float:
    t = np.arange(len(x), dtype=np.float64)
    x = x.astype(np.float64)
    if len(x) < 2:
        return 0.0
    return float(np.polyfit(t, x, 1)[0])


def extract_features(X: np.ndarray, n_numeric: int = 16) -> np.ndarray:
    """X: [N, 128, C]; returns [N, n_numeric * len(METRICS) + (C - n_numeric)]."""
    N, T, C = X.shape
    feats = np.zeros((N, n_numeric * len(METRICS) + (C - n_numeric)), dtype=np.float64)
    for j in range(n_numeric):
        col = X[:, :, j].astype(np.float64)
        base = j * len(METRICS)
        feats[:, base + 0] = col.mean(axis=1)
        feats[:, base + 1] = col.std(axis=1)
        feats[:, base + 2] = col.min(axis=1)
        feats[:, base + 3] = col.max(axis=1)
        feats[:, base + 4] = np.median(col, axis=1)
        q = np.percentile(col, [10, 90], axis=1)
        feats[:, base + 5] = q[0]
        feats[:, base + 6] = q[1]
        t = np.arange(T, dtype=np.float64)
        sx = col.sum(axis=1)
        st = np.full(N, T * (T - 1) / 2.0)
        stt = np.full(N, np.sum(t * t))
        stx = col @ t
        # The regression design has T observations per chunk. Using N here
        # (the number of chunks) silently produced incorrect slopes whenever
        # the batch size differed from the window length.
        denom = T * stt - st * st
        feats[:, base + 7] = (T * stx - st * sx) / denom
        feats[:, base + 8] = np.sum(col ** 2, axis=1)
        feats[:, base + 9] = np.mean(col == 0, axis=1)
        feats[:, base + 10] = q[1] - q[0]
    if C > n_numeric:
        feats[:, n_numeric * len(METRICS):] = X[:, 0, n_numeric:]
    return feats


def make_feature_names(kpis: list[str], n_numeric: int = 16) -> list[str]:
    names = [f"{kpi}_{metric}" for kpi in kpis[:n_numeric] for metric in METRICS]
    extras = ["UL_Protocol_code", "DL_Protocol_code"]
    return names + extras[: max(0, 18 - n_numeric)]
