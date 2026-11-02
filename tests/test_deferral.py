import numpy as np
from src.explainability.deferral import attribution_stability, fit_deferral_calibrator, coverage_risk_curve


def test_stability_shape():
    A = np.random.rand(5, 4)
    s = attribution_stability(A)
    assert s.shape == (5,)
    assert ((s >= 0) & (s <= 1)).all()


def test_deferral_curve():
    rng = np.random.default_rng(0)
    A = rng.random((20, 4))
    s = attribution_stability(A)
    ent = rng.random(20)
    mp = rng.random(20)
    correct = (rng.random(20) > 0.5).astype(int)
    clf = fit_deferral_calibrator(s, ent, mp, correct)
    curve = coverage_risk_curve(clf, s, ent, mp, correct)
    assert len(curve) == 20
    assert all(0 <= c <= 1 and 0 <= p <= 1 for c, p in curve)
