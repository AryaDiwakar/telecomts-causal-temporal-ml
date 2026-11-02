"""Deferral / selective RCA (Phase 6).

Attribution-stability + entropy calibrator -> coverage-risk curve.
Low-stability windows are deferred ("no reliable root cause").
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def attribution_stability(attributions, n_perturb=5, noise=0.05):
    """Mean rank-correlation stability of attribution under input perturbation.
    attributions: (B, K) array-like; returns (B,) stability scores in [0,1].
    """
    A = np.asarray(attributions, dtype=float)
    B, K = A.shape
    scores = np.zeros(B)
    for i in range(B):
        base = A[i]
        corrs = []
        for _ in range(n_perturb):
            pert = base + np.random.normal(0, noise * (np.abs(base).mean() + 1e-6), size=K)
            # rank correlation (spearman) approximated
            r1 = np.argsort(np.argsort(base))
            r2 = np.argsort(np.argsort(pert))
            if r1.std() == 0 or r2.std() == 0:
                corrs.append(0.0)
                continue
            corrs.append(float(np.corrcoef(r1, r2)[0, 1]))
        scores[i] = float(np.mean(corrs))
    return np.clip(scores, 0.0, 1.0)


def fit_deferral_calibrator(stability, entropy, max_prob, correct):
    """Fit logistic calibrator g([s, entropy, max_prob]) -> P(top-k correct)."""
    X = np.column_stack([stability, entropy, max_prob])
    y = np.asarray(correct).astype(int)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf


def coverage_risk_curve(calibrator, stability, entropy, max_prob, correct, n_thresholds=20):
    """Returns list of (coverage, precision) pairs."""
    X = np.column_stack([stability, entropy, max_prob])
    p = calibrator.predict_proba(X)[:, 1]
    order = np.argsort(-p)
    correct = np.asarray(correct).astype(float)
    results = []
    for frac in np.linspace(0.1, 1.0, n_thresholds):
        k = max(1, int(frac * len(order)))
        idx = order[:k]
        prec = float(correct[idx].mean()) if len(idx) else 0.0
        results.append((float(k / len(order)), prec))
    return results
