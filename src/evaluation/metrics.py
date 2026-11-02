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


def pr_auc(y, scores) -> float | None:
    """Average precision (PR-AUC); None when only one class is present."""
    y = np.asarray(y).ravel() > 0
    scores = np.asarray(scores).ravel().astype(float)
    if np.unique(y).size < 2:
        return None
    return float(average_precision_score(y, scores))


def roc_auc(y, scores) -> float | None:
    """ROC-AUC; None when only one class is present."""
    y = np.asarray(y).ravel() > 0
    scores = np.asarray(scores).ravel().astype(float)
    if np.unique(y).size < 2:
        return None
    return float(roc_auc_score(y, scores))


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


def _ranked_kpi_order(scores: np.ndarray) -> list[int]:
    """KPI indices sorted by descending score (stable, ties broken by index)."""
    return list(np.argsort(-np.asarray(scores, dtype=float), kind="mergesort"))


def rca_ranking_metrics(
    pred_scores: np.ndarray,
    gt_sets: list[set],
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict:
    """Top-k ranking metrics for KPI-level root-cause localization.

    ``pred_scores`` is ``[n, n_kpi]`` (higher = more likely a root cause);
    ``gt_sets`` is a list of ground-truth KPI-index sets. Returns Recall@k,
    MRR, MAP and the oracle-cardinality top-k Jaccard (``topk_jaccard_oracle_k``).

    Note: Recall@k / MRR / MAP use a *fixed* k, unlike the legacy
    ``topk_shift_jaccard`` which sets k = |ground-truth set| (label-leaking
    cardinality; see docs/benchmark_audit_v3.md, finding M4).
    """
    pred_scores = np.asarray(pred_scores, dtype=float)
    n = len(pred_scores)
    out: dict = {"n_rca": int(n)}
    if n == 0:
        for k in ks:
            out[f"recall@{k}"] = float("nan")
        out.update({"mrr": float("nan"), "map": float("nan"),
                    "topk_jaccard_oracle_k": float("nan")})
        return out

    recall_hits = {k: 0.0 for k in ks}
    rr_sum = 0.0
    ap_sum = 0.0
    jac_oracle = 0.0
    for scores, gt in zip(pred_scores, gt_sets):
        gt = set(int(i) for i in gt)
        if not gt:
            continue
        order = _ranked_kpi_order(scores)
        for k in ks:
            recall_hits[k] += len(set(order[:k]) & gt) / len(gt)
        # reciprocal rank of the first hit
        rr = 0.0
        for rank, idx in enumerate(order, start=1):
            if idx in gt:
                rr = 1.0 / rank
                break
        rr_sum += rr
        # average precision: precision at each hit position
        hits, prec_sum = 0, 0.0
        for rank, idx in enumerate(order, start=1):
            if idx in gt:
                hits += 1
                prec_sum += hits / rank
        ap_sum += prec_sum / len(gt)
        oracle_k = max(1, len(gt))
        top = set(order[:oracle_k])
        jac_oracle += len(top & gt) / max(1, len(top | gt))

    for k in ks:
        out[f"recall@{k}"] = float(recall_hits[k] / n)
    out["mrr"] = float(rr_sum / n)
    out["map"] = float(ap_sum / n)
    out["topk_jaccard_oracle_k"] = float(jac_oracle / n)
    return out


def cluster_bootstrap_ci(
    y,
    scores,
    groups,
    metric_fn,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI under a *cluster* (recording) resampling scheme.

    Finding M5 (docs/benchmark_audit_v3.md): chunks are not independent — they
    are grouped into recordings, and every model here can partly memorise a
    recording's session signature. A chunk-level bootstrap (or a seed-level
    min/max) therefore understates uncertainty. We instead resample whole
    clusters with replacement, pool their chunks, and recompute the metric.

    ``groups`` is the per-chunk cluster id (e.g. recording name). Returns the
    point estimate on the full sample plus the percentile interval, the number
    of clusters, and how many resamples yielded a defined metric.
    """
    y = np.asarray(y).ravel()
    scores = np.asarray(scores).ravel()
    groups = np.asarray(groups).ravel()
    if not (len(y) == len(scores) == len(groups)):
        raise ValueError("y, scores, groups must be the same length")

    point = metric_fn(y, scores)
    uniq = np.unique(groups)
    idx_by_group = [np.where(groups == g)[0] for g in uniq]
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx_by_group[p] for p in pick])
        v = metric_fn(y[sel], scores[sel])
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "mean": float("nan"), "std": float("nan"), "level": level,
                "n_boot": n_boot, "n_valid": 0, "n_clusters": int(len(uniq))}
    arr = np.asarray(vals)
    lo, hi = np.percentile(arr, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return {
        "point": point,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "lo": float(lo),
        "hi": float(hi),
        "level": float(level),
        "n_boot": int(n_boot),
        "n_valid": int(len(arr)),
        "n_clusters": int(len(uniq)),
    }


def bootstrap_anomaly_metrics(
    y, scores, groups, threshold: float | None = None, n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Cluster-bootstrap CIs for the headline anomaly metrics (PR/ROC/F1)."""
    out = {
        "pr_auc": cluster_bootstrap_ci(y, scores, groups, pr_auc, n_boot=n_boot, seed=seed),
        "roc_auc": cluster_bootstrap_ci(y, scores, groups, roc_auc, n_boot=n_boot, seed=seed),
    }
    if threshold is not None:
        def _f1(yy, ss):
            pred = (np.asarray(ss) >= threshold).astype(int)
            yy = np.asarray(yy) > 0
            tp = float(np.sum((pred == 1) & (yy)))
            fp = float(np.sum((pred == 1) & (~yy)))
            fn = float(np.sum((pred == 0) & (yy)))
            return (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0

        out["f1_at_val_threshold"] = cluster_bootstrap_ci(
            y, scores, groups, _f1, n_boot=n_boot, seed=seed
        )
    return out
