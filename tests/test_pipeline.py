"""Sanity tests for the TelecomTS pipeline module boundaries.

Run: python -m pytest tests/ -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"


def _require_artifacts():
    assert (INTERIM / "chunks_X.npy").exists()
    assert (INTERIM / "chunks_meta.parquet").exists()
    assert (PROCESSED / "split_indices.npz").exists()


def test_array_shapes():
    _require_artifacts()
    X = np.load(INTERIM / "chunks_X.npy")
    assert X.shape == (32000, 128, 18)
    assert X.dtype == np.float32
    assert np.isfinite(X).all()


def test_meta_consistency():
    _require_artifacts()
    meta = pd.read_parquet(INTERIM / "chunks_meta.parquet")
    assert len(meta) == 32000
    assert meta.chunk_id.is_unique
    assert not meta.duplicated(["recording", "start_time"]).any()
    assert meta.anomaly.dtype == bool
    assert int(meta.anomaly.sum()) == 1235


def test_split_integrity():
    _require_artifacts()
    z = np.load(PROCESSED / "split_indices.npz")
    tr, va, te = z["train"], z["val"], z["test"]
    assert len(set(tr) & set(va)) == 0
    assert len(set(tr) & set(te)) == 0
    assert len(set(va) & set(te)) == 0
    assert len(tr) + len(va) + len(te) <= 32000  # some straddling chunks dropped
    assert len(tr) > 20000


def test_targets_consistency():
    _require_artifacts()
    tar = dict(np.load(PROCESSED / "targets.npz"))
    meta = pd.read_parquet(INTERIM / "chunks_meta.parquet")
    assert (tar["y_anomaly"] == meta.anomaly.astype(int)).all()
    # type labels align with meta types
    rca_info = json.loads((PROCESSED / "rca_templates.json").read_text())
    for tname, tid in rca_info["type_map"].items():
        assert tid >= 0
    # affected_kpis templates cover all 11 types
    assert len(rca_info["rca_templates"]) == 11
    # onset/offset sanity
    ano = meta.anomaly.to_numpy()
    assert (tar["onset"][ano] >= 0).all()
    assert (tar["offset"][ano] >= 0).all()
    assert (tar["offset"][ano] <= 127).all()


def test_graph_mapping():
    _require_artifacts()
    g = np.load(PROCESSED / "graph_prototypes.npz")
    assert g["prototypes"].shape == (6, 16, 16)
    assert len(g["chunk_prototypes"]) == 32000
    assert (g["chunk_prototypes"] >= 0).all()
    assert (g["chunk_prototypes"] < 6).all()


def test_feature_extraction():
    sys_path_insert()
    from src.features.classical import extract_features, make_feature_names

    X = np.random.randn(5, 128, 18)
    F = extract_features(X)
    assert F.shape == (5, 16 * 11 + 2)
    assert np.isfinite(F).all()
    names = make_feature_names(["k%d" % i for i in range(16)], 16)
    assert len(names) == 16 * 11 + 2


def test_metrics():
    sys_path_insert()
    from src.evaluation.metrics import summarize

    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.05).astype(int)
    s = rng.random(1000)
    if y.sum() == 0:
        y[0] = 1
    res = summarize(y, s)
    assert 0.0 <= res["roc_auc"] <= 1.0
    assert 0.0 <= res["pr_auc"] <= 1.0
    assert "tpr_at_fpr_0.01" in res


def test_metrics_accept_validation_threshold():
    sys_path_insert()
    from src.evaluation.metrics import summarize

    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.45, 0.9])
    res = summarize(y, scores, threshold=0.5)
    assert res["threshold"] == 0.5
    assert res["recall"] == 0.5


def sys_path_insert():
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
