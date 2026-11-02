"""Causal intervention utilities for RCA (Phase 3).

Instead of using the KPI graph as message-passing (detection-neutral), use it
for counterfactual intervention: estimate P(anomaly | do(KPI_i perturbed)).

Interventional RCA fidelity = fraction of windows where intervening on the
predicted root-KPI set reduces the anomaly score more than intervening on a
random non-root set.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def intervene(x, kpi_indices, value=0.0):
    """Zero-out (or set) specified KPI channels over time. x: (B,T,C)."""
    out = x.clone()
    out[:, :, kpi_indices] = value
    return out


def interventional_fidelity(anomaly_fn, x, pred_root, non_root, threshold=0.0):
    """
    anomaly_fn: callable (B,T,C)->(B,) anomaly score (higher=more anomalous)
    Returns fraction of samples where intervening on pred_root reduces score
    more than intervening on non_root.
    """
    base = anomaly_fn(x)
    xr = intervene(x, pred_root)
    xn = intervene(x, non_root)
    sr = anomaly_fn(xr)
    sn = anomaly_fn(xn)
    win = (base - sr) > (base - sn)
    win = torch.as_tensor(win)
    return float(win.float().mean().item())


class CausalConsistencyLoss(nn.Module):
    """RCA must change under intervention on true roots, stay stable otherwise."""

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, rca_logits, x, true_roots, anomaly_fn):
        import torch.nn.functional as F
        if true_roots is None:
            return torch.tensor(0.0, device=rca_logits.device)
        target = torch.zeros_like(rca_logits)
        for i, roots in enumerate(true_roots):
            for r in roots:
                if r < rca_logits.shape[1]:
                    target[i, r] = 1.0
        loss = F.binary_cross_entropy_with_logits(rca_logits, target)
        return self.weight * loss
