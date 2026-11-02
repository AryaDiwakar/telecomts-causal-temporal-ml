"""Regression tests for defects confirmed in docs/benchmark_audit_v3.md.

Each test pins a specific finding so it cannot silently regress. Artifact-based
tests skip cleanly when the pipeline outputs are absent.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from src.evaluation.metrics import (
    pr_f1_at_threshold,
    rca_ranking_metrics,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"


# ---------------------------------------------------------------- finding M5
def test_cluster_bootstrap_captures_group_correlation():
    """The cluster bootstrap must reflect recording-level correlation.

    Each cluster has a constant score, and there are many chunks per cluster, so
    an i.i.d. *chunk* bootstrap keeps every score's support and yields a nearly
    degenerate interval. Resampling whole clusters drops/keeps the one crossing
    negative (score 10, label 0), which changes ROC-AUC -- that extra spread is
    exactly what finding M5 says the old seed-level min/max was missing.
    """
    from src.evaluation.metrics import cluster_bootstrap_ci, roc_auc

    per = 50
    lab = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    sc = np.array([9.0, 8, 7, 6, 5, 4, 3, 2, 1, 10])  # cluster 9 is crossing
    y = np.repeat(lab, per)
    s = np.repeat(sc, per)
    groups = np.repeat(np.arange(10), per)

    cl = cluster_bootstrap_ci(y, s, groups, roc_auc, n_boot=500, seed=4)
    ch = cluster_bootstrap_ci(y, s, np.arange(len(y)), roc_auc, n_boot=500, seed=4)

    w_cl, w_ch = cl["hi"] - cl["lo"], ch["hi"] - ch["lo"]
    assert w_cl > ch["hi"] - ch["lo"], "cluster CI must be wider"
    assert w_cl > 3 * w_ch, "recording-level correlation should dominate (finding M5)"
    assert cl["n_clusters"] == 10


def test_cluster_bootstrap_point_matches_full_sample():
    from src.evaluation.metrics import pr_auc, cluster_bootstrap_ci

    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(15), 20)
    scores = rng.normal(0, 1, 300) + rng.normal(0, 2, 15)[groups]
    y = (scores > 0).astype(int)
    ci = cluster_bootstrap_ci(y, scores, groups, pr_auc, n_boot=300, seed=0)
    assert ci["point"] == pytest.approx(pr_auc(y, scores))
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["n_clusters"] == 15


def test_bootstrap_anomaly_metrics_keys():
    from src.evaluation.metrics import bootstrap_anomaly_metrics

    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 30)
    scores = rng.normal(0, 1, 300) + rng.normal(0, 2, 10)[groups]
    y = (scores > 0).astype(int)
    out = bootstrap_anomaly_metrics(y, scores, groups, threshold=0.0, n_boot=100, seed=0)
    assert {"pr_auc", "roc_auc", "f1_at_val_threshold"} <= set(out)
    for key in out:
        assert {"point", "lo", "hi", "n_clusters"} <= set(out[key])


# ---------------------------------------------------------------- finding M6
def test_stats_agg_uses_t_quantile_not_z():
    """n=5 seeds is far too few for a z-interval (finding M6)."""
    import importlib

    sweep = importlib.import_module("scripts.run_ci_sweep")
    rows = [{"m": v} for v in (0.80, 0.82, 0.84, 0.86, 0.88)]
    agg = sweep.stats_agg("m", rows)
    # t_{0.95,4} = 2.132 > z = 1.645, so the 90% interval must be wider than z
    width_t = agg["ci90_high"] - agg["ci90_low"]
    sd = agg["std"]
    width_z = 2 * 1.645 * sd / 5 ** 0.5
    assert width_t > width_z
    assert agg["t_quantile_90"] == pytest.approx(2.1318, abs=1e-3)
    assert agg["t_quantile_95"] == pytest.approx(2.7764, abs=1e-3)


# ---------------------------------------------------------------- finding M2
def test_unsupervised_score_samples_is_normality_oriented():
    """IsolationForest.score_samples is higher for normal points, so the anomaly
    score must be its negation. Guards the exp_001 sign-inversion bug (M2)."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 1.0, size=(300, 4))
    anomaly = rng.normal(6.0, 1.0, size=(40, 4))
    X = np.vstack([normal, anomaly])
    y = np.r_[np.zeros(len(normal)), np.ones(len(anomaly))].astype(int)

    model = IsolationForest(n_estimators=100, random_state=0).fit(normal)
    raw = model.score_samples(X)
    assert roc_auc_score(y, raw) < 0.5           # raw = normality score
    assert roc_auc_score(y, -raw) > 0.5          # negated = anomaly score


# ---------------------------------------------------------------- finding M3
def test_summarize_reports_threshold_used():
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.6, 0.9, 0.95])
    # explicit (validation) threshold is reported verbatim and drives F1
    out = summarize(y, s, threshold=0.5)
    assert out["threshold"] == 0.5
    assert out["precision"] == pytest.approx(1.0)
    assert out["recall"] == pytest.approx(1.0)
    # a stricter threshold keeps the reported value and lowers recall
    out2 = summarize(y, s, threshold=0.8)
    assert out2["threshold"] == 0.8
    assert out2["recall"] == pytest.approx(2 / 3)


def test_summarize_none_threshold_is_test_optimal_marked_optimistic():
    """summarize(threshold=None) optimises F1 on the given labels; the exp_* code
    must pass a val threshold for the reported operating point (M3)."""
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.6, 0.9, 0.95])
    best = pr_f1_at_threshold(y, s)
    assert best["f1_max"] == pytest.approx(1.0)


# ---------------------------------------------------------------- finding M4
def test_rca_ranking_metrics_perfect_and_reversed():
    gt = [{0, 1}, {2}, {3, 4, 5}]
    perfect = np.zeros((3, 6))
    # distinct descending scores so the ranking is unambiguous
    perfect[0, 0], perfect[0, 1] = 10.0, 9.0
    perfect[1, 2] = 10.0
    perfect[2, 3], perfect[2, 4], perfect[2, 5] = 10.0, 9.0, 8.0
    m = rca_ranking_metrics(perfect, gt, ks=(1, 3, 5))
    assert m["recall@3"] == pytest.approx(1.0)
    assert m["recall@5"] == pytest.approx(1.0)
    assert m["mrr"] == pytest.approx(1.0)
    assert m["map"] == pytest.approx(1.0)
    assert m["topk_jaccard_oracle_k"] == pytest.approx(1.0)

    mr = rca_ranking_metrics(-perfect, gt, ks=(1, 3, 5))
    assert mr["recall@1"] == pytest.approx(0.0)
    assert mr["mrr"] < m["mrr"]
    assert mr["map"] < 1.0


def test_rca_ranking_metrics_uses_fixed_k_not_oracle_k():
    # single window, gt size 3, only one true root ranked first
    scores = np.array([[1.0, 0.0, 0.0, 0.0]])
    m = rca_ranking_metrics(scores, [{0, 1, 2}], ks=(1, 3))
    assert m["recall@1"] == pytest.approx(1 / 3)
    assert m["recall@3"] == pytest.approx(1.0)  # fixed k=3 finds all three
    assert m["topk_jaccard_oracle_k"] == pytest.approx(1.0)  # oracle k=|gt|=3


# ---------------------------------------------------------------- finding G1
def _require_rca_artifacts():
    for f in ("chunks_meta.parquet",):
        if not (INTERIM / f).exists():
            pytest.skip("pipeline artifacts not built")
    if not (PROCESSED / "split_indices.npz").exists() or not (PROCESSED / "rca_templates.json").exists():
        pytest.skip("pipeline artifacts not built")


def test_rca_templates_are_train_only_and_not_lastwritewins():
    """rca_templates.json must be the TRAIN majority set per type, never the last
    anomalous chunk over all splits (finding G1)."""
    _require_rca_artifacts()
    import pandas as pd

    meta = pd.read_parquet(INTERIM / "chunks_meta.parquet")
    npz = np.load(PROCESSED / "split_indices.npz")
    train_idx = set(int(i) for i in npz["train"])
    art = json.loads((PROCESSED / "rca_templates.json").read_text())
    templates = art["rca_templates"]

    counts: dict[str, Counter] = {}
    last_all: dict[str, tuple] = {}
    for i, row in meta.iterrows():
        if not bool(row.anomaly):
            continue
        key = tuple(sorted(json.loads(row.affected_kpis)))
        last_all[row.anomaly_type] = key
        if int(i) in train_idx:
            counts.setdefault(row.anomaly_type, Counter())[key] += 1

    assert art.get("rca_template_source") == "train", "templates not marked train-only"
    for tname, counter in counts.items():
        best = max(counter.items(), key=lambda kv: (kv[1], kv[0]))[0]
        assert tuple(sorted(templates[tname])) == best, (
            f"{tname}: template is not the train-majority set"
        )
    # at least one type must differ from the contaminated last-write-wins set,
    # otherwise this regression test would not catch the original bug
    differing = sum(
        tuple(sorted(templates[t])) != last_all.get(t) for t in templates
    )
    assert differing > 0


# ---------------------------------------------------------------- finding S1
def test_no_cross_split_timestep_overlap():
    """Retained train/test windows must not share a raw timestep (finding S1)."""
    if not (INTERIM / "chunks_meta.parquet").exists() or not (PROCESSED / "split_indices.npz").exists():
        pytest.skip("pipeline artifacts not built")
    import pandas as pd

    meta = pd.read_parquet(INTERIM / "chunks_meta.parquet")
    npz = np.load(PROCESSED / "split_indices.npz")

    def spans(idxs):
        return [
            (meta.iloc[int(i)].recording, meta.iloc[int(i)].start_time, meta.iloc[int(i)].end_time)
            for i in idxs
        ]

    train = spans(npz["train"][:4000])
    test = spans(npz["test"])
    tr = {(r, s, e) for r, s, e in train}
    te = {(r, s, e) for r, s, e in test}
    assert not (tr & te), "identical train/test windows"


# ---------------------------------------------------------------- Phase 6
def test_registry_enrich_adds_provenance():
    from src.utils import registry

    rec = registry.enrich({"experiment": "test", "seed": 0, "task": "t", "model": "m", "metrics": {}})
    assert rec["status"] in registry.VALID_STATUS
    assert rec["code_revision"]
    assert rec["data_fingerprint"]
    assert rec["config_hash"]
    assert rec["timestamp"]


def test_registry_rejects_invalid_status():
    from src.utils import registry

    with pytest.raises(ValueError):
        registry.append({"experiment": "x", "seed": 0, "status": "BOGUS", "metrics": {}})


# ---------------------------------------------------------------- finding R4
def test_fold_scaler_ignores_val_test_data():
    """The LORO scaler must depend only on fold-train chunks (finding R4)."""
    from src.models.graph_fold import fold_scaler

    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(40, 8, 3)).astype(np.float32)
    train = np.arange(0, 25)
    test = np.arange(25, 40)
    base = fold_scaler(X, train, "zscore")
    # perturb only test chunks -> scaler must not move
    X2 = X.copy()
    X2[test] += 1000.0
    after = fold_scaler(X2, train, "zscore")
    assert np.allclose(base["loc"], after["loc"])
    assert np.allclose(base["scale"], after["scale"])
    # perturb train chunks -> scaler must move
    X3 = X.copy()
    X3[train] += 1000.0
    moved = fold_scaler(X3, train, "zscore")
    assert not np.allclose(base["loc"], moved["loc"])


def test_fold_prototypes_fit_only_on_train_pure_bins():
    """Centroids must depend only on bins whose chunks are all in fold-train."""
    from src.models.graph_fold import fold_prototypes

    rng = np.random.default_rng(0)
    n_bins, K, n = 8, 4, 16
    adj = rng.normal(0, 1, size=(n_bins, 4, 4)).astype(np.float32)
    # bins 0..3 are pure-train; bins 4..7 contain test chunks
    members = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7],
               4: [8, 9], 5: [10, 11], 6: [12, 13], 7: [14, 15]}
    chunk_bins = np.full(n, -1, dtype=np.int64)
    for b, cs in members.items():
        for c in cs:
            chunk_bins[c] = b
    train_ids = np.arange(0, 8)  # only chunks 0..7 -> only pure bins

    p1, a1 = fold_prototypes(chunk_bins, adj, train_ids, n_prototypes=K)
    adj2 = adj.copy()
    adj2[4:] += 500.0  # corrupt every impure (test-containing) bin
    p2, a2 = fold_prototypes(chunk_bins, adj2, train_ids, n_prototypes=K)
    # prototypes depend only on pure bins -> unaffected by impure-bin corruption
    assert np.allclose(p1, p2), "impure bins leaked into prototype fit"
    assert (a1 >= 0).all() and (a2 >= 0).all()


def test_paired_cluster_bootstrap_detects_dominance():
    """Paired difference CI must separate a strictly better model (finding M5)."""
    from scripts.paired_test import metric_delta
    from src.evaluation.metrics import pr_auc

    rng = np.random.default_rng(0)
    n = 600
    groups = np.repeat([f"rec{i}" for i in range(20)], n // 20)
    y = np.zeros(n, dtype=int)
    # recordings 0..9 are anomalous; give model A clearly more signal
    y[: n // 2] = 1
    good = y * 1.5 + rng.normal(0, 0.4, n)
    bad = y * 0.3 + rng.normal(0, 0.4, n)
    d = metric_delta(y, good, bad, groups, pr_auc, n_boot=300, seed=1)
    assert d["point"] > 0
    assert d["lo"] > 0, "CI should exclude zero for a strictly better model"
    assert d["frac_le_0"] < 0.05


def test_paired_cluster_bootstrap_is_symmetric():
    from scripts.paired_test import metric_delta
    from src.evaluation.metrics import pr_auc

    rng = np.random.default_rng(3)
    n = 400
    groups = np.repeat([f"r{i}" for i in range(10)], n // 10)
    y = (rng.random(n) < 0.4).astype(int)
    a = y * 0.8 + rng.normal(0, 0.5, n)
    b = y * 0.8 + rng.normal(0, 0.5, n)
    d1 = metric_delta(y, a, b, groups, pr_auc, n_boot=200, seed=2)
    d2 = metric_delta(y, b, a, groups, pr_auc, n_boot=200, seed=2)
    assert abs(d1["point"] + d2["point"]) < 1e-9
    assert abs(d1["lo"] + d2["hi"]) < 1e-9


# ---------------------------------------------------------- findings G2/G3/G4
def _mk_event(start, end, kpi, func, params):
    from src.data.synthetic_inject import Effect, Event

    return Event(anomaly_type="A", start=start, end=end,
                 effects=[Effect(kpi=kpi, func=func, params=params, expected_dir=0)])


def test_g2_observable_kpis_drops_identity_and_inert_effects():
    """aff/kpi labels must reflect an *actual* post-injection change (G2).

    A `constant_shift` with delta=0 is the identity map, and a scale effect on an
    all-zero base is invisible; neither may be reported as an affected KPI.
    """
    from src.data.synthetic_inject import KPI_ORDER, _observable_kpis

    n = 8
    num = np.zeros((len(KPI_ORDER), 8)).T.copy()
    num[:, KPI_ORDER.index("RSRP")] = 5.0
    transformed = num.copy()

    identity = _mk_event(0, 7, "RSRP", "constant_shift", {"delta": 0.0})
    inert = _mk_event(0, 7, "DL_BLER", "constant_scale", {"factor": 3.0})  # base all-zero

    assert _observable_kpis(num, transformed, [identity, inert], 0, 7) == []

    # a genuine shift is observable
    from src.data.synthetic_inject import KPI_ORDER as KO
    from src.data.synthetic_inject import _apply_func

    real = _mk_event(0, 7, "RSRP", "constant_shift", {"delta": -3.0})
    t2 = num.copy()
    col = KO.index("RSRP")
    t2[:, col] = _apply_func("constant_shift", t2[:, col], np.linspace(0, 1, 8), {"delta": -3.0})
    assert _observable_kpis(num, t2, [real], 0, 7) == ["RSRP"]


def test_g3_observable_kpis_unions_overlapping_events():
    """A multi-event window must pool affected KPIs from every overlapping event."""
    from src.data.synthetic_inject import KPI_ORDER, _apply_func, _observable_kpis

    n = 8
    num = np.full((n, 16), 5.0)
    u = np.linspace(0, 1, n)
    ev_a = _mk_event(0, 4, "RSRP", "constant_shift", {"delta": -2.0})
    ev_b = _mk_event(3, 7, "UL_SNR", "constant_shift", {"delta": -2.0})
    t = num.copy()
    for ev, kpi in ((ev_a, "RSRP"), (ev_b, "UL_SNR")):
        c = KPI_ORDER.index(kpi)
        t[:, c] = _apply_func("constant_shift", t[:, c], u, {"delta": -2.0})
    # looking only at overlap[0] would miss UL_SNR
    assert _observable_kpis(num, t, [ev_a], 0, n - 1) == ["RSRP"]
    assert _observable_kpis(num, t, [ev_a, ev_b], 0, n - 1) == ["RSRP", "UL_SNR"]


def test_g4_window_end_time_spans_wlen_steps():
    """end_time must be start + (window_length-1) steps, not 10x that (finding G4)."""
    import pandas as pd

    from src.data.synthetic_inject import _shift_time

    t0 = "2025-06-18 00:40:09.600"
    w0, wlen = 0, 128
    start = pd.Timestamp(_shift_time(t0, w0))
    end = pd.Timestamp(_shift_time(t0, w0 + (wlen - 1)))
    assert (end - start).total_seconds() == pytest.approx(12.7, abs=1e-6)


def test_g4_emitted_windows_have_correct_duration():
    """Artifact check: every generated window spans exactly 12.7 s (finding G4)."""
    import pandas as pd

    root = ROOT / "data" / "raw" / "TelecomTS" / "anomalous" / "synthetic_gen"
    files = sorted(root.rglob("processed/chunked.jsonl"))
    if not files:
        pytest.skip("synthetic_gen corpus not generated")
    bad = 0
    n = 0
    for f in files:
        for line in f.read_text().splitlines():
            rec = json.loads(line)
            n += 1
            dt = (pd.Timestamp(rec["end_time"]) - pd.Timestamp(rec["start_time"])).total_seconds()
            if abs(dt - 12.7) > 1e-6:
                bad += 1
    assert n > 0
    assert bad == 0, f"{bad}/{n} windows have the 10x end_time bug (finding G4)"


def test_g2_generated_labels_exclude_unobservable_kpis():
    """Artifact check: an affected KPI must differ from its pre-injection source.

    The synthetic recording is injected onto ``normal/<rel>`` where ``<rel>`` is
    the same relative path; timestamps are preserved, so the source row at the
    window start is the true pre-injection base. A labelled affected KPI must
    deviate from that base somewhere in the window.
    """
    import pandas as pd

    gen_root = ROOT / "data" / "raw" / "TelecomTS" / "anomalous" / "synthetic_gen"
    src_root = ROOT / "data" / "raw" / "TelecomTS" / "normal"
    files = sorted(gen_root.rglob("processed/chunked.jsonl"))
    if not files:
        pytest.skip("synthetic_gen corpus not generated")

    checked = suspicious = 0
    for f in files:
        rel = f.parent.parent.relative_to(gen_root)          # e.g. mobile/File
        raw_csv = gen_root / rel / "raw" / "metrics.csv"
        src_csv = src_root / rel / "raw" / "metrics.csv"
        if not src_csv.exists():
            continue
        raw = pd.read_csv(raw_csv)
        src = pd.read_csv(src_csv)
        if len(src) != len(raw):
            continue
        for line in f.read_text().splitlines():
            rec = json.loads(line)
            ts = rec["start_time"]
            row0 = raw[raw["timestamp"] == ts]
            if row0.empty:
                continue
            i0 = int(row0.index[0])
            for kpi in rec["anomalies"]["affected_kpis"]:
                post = raw[kpi].to_numpy()[i0 : i0 + 128].astype(float)
                base = src[kpi].to_numpy()[i0 : i0 + 128].astype(float)
                checked += 1
                scale = max(1.0, float(np.max(np.abs(base))))
                if float(np.max(np.abs(post - base))) < 1e-6 * scale:
                    suspicious += 1
    assert checked > 0
    assert suspicious == 0, f"{suspicious}/{checked} labelled KPIs equal their pre-injection base"
