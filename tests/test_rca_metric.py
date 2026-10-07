"""Regression test: type_template_jaccard must be prediction-based, not GT.

If the metric used ground-truth types, it would be ~1.0 by construction
(affected_kpis are per-type constant templates). The metric must drop toward
~type accuracy for a bad type classifier.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _compute_jac(yp, yt, gt_vecs, rca, kpi_names):
    inv = {v: k for k, v in rca["type_map"].items()}
    kpi_idx = {k: i for i, k in enumerate(kpi_names)}
    jac_t = []
    for j in range(len(gt_vecs)):
        gi = set(np.where(gt_vecs[j])[0])
        ti = inv[int(yp[j])]
        pt = {kpi_idx[k] for k in rca["rca_templates"][ti] if k in kpi_idx}
        jac_t.append(len(gi & pt) / max(1, len(gi | pt)))
    return float(np.mean(jac_t))


def test_rca_uses_predicted_type():
    rca = json.loads((ROOT / "data" / "processed" / "rca_templates.json").read_text())
    kpi_names = json.load((ROOT / "data" / "interim" / "kpi_names.json").open())["numeric_kpis"]
    rng = np.random.default_rng(0)
    n = 100
    types = np.array([rng.integers(0, 11) for _ in range(n)])
    gt = np.zeros((n, 16), dtype=float)
    inv = {v: k for k, v in rca["type_map"].items()}
    kpi_idx = {k: i for i, k in enumerate(kpi_names)}
    for j, t in enumerate(types):
        for k in rca["rca_templates"][inv[t]]:
            if k in kpi_idx:
                gt[j, kpi_idx[k]] = 1.0

    # Perfect type prediction -> templates overlap with GT masks.
    exact = _compute_jac(types, types, gt, rca, kpi_names)
    assert exact > 0.99, f"perfect-type jaccard should be ~1.0, got {exact}"

    # Random type prediction: clearly below perfect but with a high floor,
    # because RCA templates share a large common KPI core (Jamming/UL_BLER/
    # DL_BLER/TX/RX bytes appear in ~every template). ~uniform random over 11
    # types gives ~0.56 given the template overlap.
    rand = _compute_jac(rng.integers(0, 11, n), types, gt, rca, kpi_names)
    assert rand < exact, "random-type jaccard must be below perfect-type"
    assert 0.4 < rand < 0.8, f"random-type jaccard in template-overlap band, got {rand:0.3f}"