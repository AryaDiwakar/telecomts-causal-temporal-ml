"""Regression tests for the train-only scaler (degenerate-IQR fix)."""

import numpy as np
import pytest

from src.preprocessing.prepare import fit_scaler


def test_zero_inflated_channel_does_not_explode():
    """DL_BLER-style channel: 59.5% exact zeros, tiny IQR, large tail."""
    rng = np.random.default_rng(0)
    n, t = 400, 32
    ch = np.zeros((n, t))
    nz = rng.random((n, t)) > 0.595
    ch[nz] = rng.gamma(2.0, 0.05, size=nz.sum())
    ch[nz & (rng.random((n, t)) < 0.01)] += 1.0   # heavy tail
    X = np.concatenate([ch[..., None], rng.normal(size=(n, t, 1))], axis=-1)

    sc = fit_scaler(X, np.arange(n), "robust_zscore")
    scale = np.asarray(sc["scale"])
    scaled = (X - np.asarray(sc["loc"])) / scale

    assert scale[0] > 1e-3, "degenerate IQR must fall back to a real spread estimate"
    assert scaled[..., 0].std() < 5.0, "zero-inflated channel must not dominate"
    assert np.abs(scaled[..., 0]).max() < 200.0


def test_well_behaved_channel_keeps_iqr_scale():
    rng = np.random.default_rng(1)
    X = rng.normal(10.0, 3.0, size=(500, 32, 1))
    sc = fit_scaler(X, np.arange(500), "robust_zscore")
    scale = np.asarray(sc["scale"])[0]
    # IQR/1.349 of a normal ~= sigma
    assert 2.4 < scale < 3.6


def test_still_fits_on_train_only():
    rng = np.random.default_rng(2)
    X = np.concatenate([rng.normal(0, 1, size=(100, 16, 1)),
                        rng.normal(100, 1, size=(100, 16, 1))])
    sc = fit_scaler(X, np.arange(100), "robust_zscore")
    # train half is N(0,1) -> location near 0, not pulled by the held-out half
    assert abs(np.asarray(sc["loc"])[0]) < 0.5


def test_scale_never_degenerate():
    rng = np.random.default_rng(3)
    constant = np.zeros((200, 16, 1))
    const_col = np.ones((200, 16, 1)) * 7.0
    X = np.concatenate([constant, const_col, rng.normal(size=(200, 16, 1))], axis=-1)
    sc = fit_scaler(X, np.arange(200), "robust_zscore")
    assert (np.asarray(sc["scale"]) >= 1e-8).all()


@pytest.mark.parametrize("method", ["zscore", "robust_zscore", "minmax"])
def test_all_methods_return_matching_lengths(method):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(50, 8, 5))
    sc = fit_scaler(X, np.arange(50), method)
    assert len(sc["loc"]) == 5 and len(sc["scale"]) == 5
    assert np.isfinite(sc["loc"]).all() and np.isfinite(sc["scale"]).all()
