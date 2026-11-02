"""Tests for self-supervised pretraining of the TAGN trunk (exp_008)."""

import numpy as np
import pytest
import torch

from src.models.pretrain import (
    DEFAULTS,
    TagnPretrain,
    augment,
    sample_mask,
)
from src.models.tagn import N_NUMERIC, TagnModel, model_config


@pytest.fixture(scope="module")
def tagn():
    torch.manual_seed(0)
    m = TagnModel(model_config())
    m.set_prototypes(torch.eye(N_NUMERIC).unsqueeze(0))  # single identity prototype
    return m


def test_encode_matches_forward_trunk(tagn):
    tagn.eval()
    x = torch.randn(4, 128, 18)
    proto = torch.zeros(4, dtype=torch.long)
    enc = tagn.encode(x, proto)
    assert set(enc) == {"pooled", "h", "node_feats", "fused"}
    assert enc["pooled"].shape == (4, tagn.cfg["d"])
    assert enc["node_feats"].shape == (4, 128, N_NUMERIC, tagn.cfg["d"])

    logits, node_feats, h = tagn(x, proto)
    assert torch.equal(node_feats, enc["node_feats"])
    assert torch.equal(h, enc["h"])
    assert logits["kpi"].shape == (4, N_NUMERIC)


HEAD_PREFIXES = (
    "summary_proj", "summary_head", "summary_residual", "summary_type_head",
    "summary_loc", "summary_scale",
    "anom_head", "type_head", "kpi_head", "ew_head",
    "rca_adv", "ident_head",
)


def test_encoder_state_roundtrip(tagn):
    state = tagn.encoder_state()
    assert len(state) > 0
    assert not any(k.startswith(HEAD_PREFIXES) for k in state)

    other = TagnModel(model_config())
    for p in other.parameters():
        p.data.normal_(0.0, 1.0)  # ensure it differs before loading
    applied = other.load_encoder_state(state)
    assert applied == len(state)

    for k, v in tagn.state_dict().items():
        if k.startswith(HEAD_PREFIXES):
            continue
        if k in ("prototypes", "static_adj"):
            continue
        assert torch.equal(v, other.state_dict()[k]), k


def test_encoder_state_rejects_empty(tagn):
    with pytest.raises(ValueError):
        tagn.load_encoder_state({"definitely.not.a.key": torch.zeros(1)})


def test_sample_mask_always_masks_something():
    rng = np.random.default_rng(0)
    mask = sample_mask(6, 128, 0.35, torch.device("cpu"), rng)
    assert mask.shape == (6, 128, N_NUMERIC)
    assert mask.dtype == torch.bool
    per_window = mask[:, 0].sum(dim=1)
    assert (per_window > 0).all()
    # identical across time steps within a window (channel-level masking)
    assert (mask == mask[:, :1]).all()


def test_augment_only_touches_numeric_channels():
    rng = np.random.default_rng(0)
    x = torch.randn(3, 128, 18)
    out = augment(x, rng, noise_std=0.05, drop=0.15)
    assert out.shape == x.shape
    assert torch.equal(out[:, :, 16:], x[:, :, 16:])


def test_nt_xent_prefers_aligned_pairs():
    torch.manual_seed(0)
    z = torch.randn(16, 32)
    good = TagnPretrain.nt_xent(z, z + 0.01 * torch.randn_like(z), 0.2)
    bad = TagnPretrain.nt_xent(z, z[torch.randperm(16)], 0.2)
    assert good < bad


def test_pretrain_forward_shapes_and_gradients(tagn):
    torch.manual_seed(0)
    pre = TagnPretrain(tagn, dict(DEFAULTS))
    rng = np.random.default_rng(0)
    x = torch.randn(6, 128, 18)
    proto = torch.zeros(6, dtype=torch.long)
    out = pre(x, proto, rng)

    assert out["loss"].requires_grad
    for k in ("recon", "forecast", "contrastive"):
        assert torch.isfinite(out[k])
    out["loss"].backward()
    assert pre.recon_head[0].weight.grad is not None
    assert tagn.node_enc.blocks[0].weight.grad is not None

    # heads must not exist yet: they stay at their supervised init
    assert not hasattr(pre, "anom_head")
