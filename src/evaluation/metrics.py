"""Evaluation metrics helpers (pertinent for imbalanced anomaly data)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def pr_f1_at_threshold(y: np.ndarray, scores: np.ndarray) -> dict:
    """F1 at the threshold maximizing F1 on label==1 support, plus the threshold."""
    p, r, t = precision_recall_curve(y, scores)
    fs = 2 * p * r / np.maximum(p + r, 1e-12)
    best = int(np.argmax(fs))
    threshold = t[best] if best < len(t) else 0.5
    pred = (scores >= threshold).astype(int)
    return {
        "f1_max": float(f1_score(y, pred)),
        "precision": float(p[best]),
        "recall": float(r[best]),
        "threshold": float(threshold),
    }


def tpr_at_fpr(y: np.ndarray, scores: np.ndarray, fpr_level: float = 0.01) -> float:
    """True positive rate at a fixed false positive rate (label==1 is positive)."""
    n_neg = int(np.sum(y == 0))
    n_pos = int(np.sum(y == 1))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    fps = np.cumsum(1 - y_sorted)
    tps = np.cumsum(y_sorted)
    allowed = fps / n_neg <= fpr_level
    if not allowed.any():
        return 0.0
    return float(tps[allowed].max() / n_pos)


def summarize(y: np.ndarray, scores: np.ndarray, threshold: float | None = None) -> dict:
    """Summarize ranking metrics and optionally score at a fixed threshold.

    ``threshold`` is used for the operating-point metrics only. Ranking metrics
    remain threshold-free, which keeps validation-selected test reporting
    comparable to the usual PR-AUC/ROC-AUC convention.
    """
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    out = {
        "n": int(len(y)),
        "n_positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, scores)) if np.unique(y).size == 2 else None,
        "pr_auc": float(average_precision_score(y, scores)),
    }
    if threshold is None:
        out.update(pr_f1_at_threshold(y, scores))
    else:
        pred = (scores >= threshold).astype(int)
        out.update(
            {
                "f1_max": float(f1_score(y, pred, zero_division=0)),
                "precision": float(np.sum((pred == 1) & (y == 1)) / max(1, np.sum(pred == 1))),
                "recall": float(np.sum((pred == 1) & (y == 1)) / max(1, np.sum(y == 1))),
                "threshold": float(threshold),
            }
        )
    out["tpr_at_fpr_0.01"] = tpr_at_fpr(y, scores, 0.01)
    out["tpr_at_fpr_0.05"] = tpr_at_fpr(y, scores, 0.05)
    return out


def jaccard_sets(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union else 0.0
