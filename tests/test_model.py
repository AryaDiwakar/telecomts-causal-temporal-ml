"""Focused regression tests for the TAGN graph/node interface."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.classical import extract_features
from src.models.tagn import DilatedCausalConv, GATLayer, NodeEncoder, TagnModel, chunk_summary


def _cfg() -> dict:
    return {
        "input_dim": 18,
        "n_kpis": 16,
        "d": 8,
        "n_layers_t": 1,
        "gat_heads": 2,
        "gat_layers": 1,
        "dropout": 0.0,
        "device": torch.device("cpu"),
        "loss_weights": {"anomaly": 1.0, "type": 1.0, "root_cause": 1.0, "early_warning": 0.5},
    }


def test_node_encoder_keeps_kpis_separate():
    encoder = NodeEncoder(d=4, layers=1).eval()
    baseline = encoder(torch.zeros(1, 8, 16))
    changed = torch.zeros(1, 8, 16)
    changed[:, :, 0] = 1.0
    result = encoder(changed)

    assert result.shape == (1, 8, 16, 4)
    assert torch.allclose(result[:, :, 1], baseline[:, :, 1])
    assert not torch.allclose(result[:, :, 0], baseline[:, :, 0])


def test_chunk_summary_preserves_classical_feature_order():
    x = torch.randn(3, 8, 18)
    summary = chunk_summary(x).numpy()
    classical = extract_features(x.numpy())

    assert np.allclose(summary[:, :176], classical[:, :176], atol=1e-6)


def test_temporal_encoder_is_causal():
    encoder = DilatedCausalConv(in_channels=3, d=4, layers=2).eval()
    prefix = torch.randn(1, 6, 3)
    first = torch.cat([prefix, torch.zeros(1, 4, 3)], dim=1)
    second = first.clone()
    second[:, 6:] = 100.0

    out_first = encoder(first)
    out_second = encoder(second)
    assert torch.allclose(out_first[:, :6], out_second[:, :6], atol=1e-6)


def test_gat_isolated_nodes_do_not_mix():
    layer = GATLayer(d=8, n_heads=2).eval()
    adjacency = torch.zeros(1, 4, 4)
    first = torch.zeros(1, 2, 4, 8)
    second = first.clone()
    second[:, :, 1] = 1.0

    out_first = layer(first, adjacency)
    out_second = layer(second, adjacency)

    assert torch.allclose(out_first[:, :, 0], out_second[:, :, 0])


def test_gat_follows_source_to_target_direction():
    layer = GATLayer(d=2, n_heads=1).eval()
    with torch.no_grad():
        layer.q.weight.zero_()
        layer.q.bias.zero_()
        layer.k.weight.zero_()
        layer.k.bias.zero_()
        layer.v.weight.copy_(torch.eye(2))
        layer.v.bias.zero_()
        layer.out.weight.copy_(torch.eye(2))
        layer.out.bias.zero_()

    x = torch.zeros(1, 1, 3, 2)
    x[:, :, 0] = torch.tensor([1.0, 2.0])
    # The documented convention is adjacency[source, target].
    adj = torch.zeros(1, 3, 3)
    adj[:, 0, 1] = 1.0

    out = layer(x, adj)
    assert not torch.allclose(out[:, :, 1], torch.zeros_like(out[:, :, 1]))
    assert torch.allclose(out[:, :, 2], torch.zeros_like(out[:, :, 2]))


def test_model_emits_distinct_node_states():
    model = TagnModel(_cfg()).eval()
    model.set_prototypes(torch.zeros(1, 16, 16))
    logits, node_states, hidden = model(torch.randn(2, 8, 18), torch.zeros(2, dtype=torch.long))

    assert node_states.shape == (2, 8, 16, 8)
    assert hidden.shape == (2, 8, 8)
    assert logits["kpi"].shape == (2, 16)
    assert not torch.allclose(node_states[:, :, 0], node_states[:, :, 1])


def test_auto_device_prefers_available_accelerator():
    from src.models.tagn import model_config

    device = model_config()["device"]
    if torch.cuda.is_available():
        assert device.type == "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        assert device.type == "mps"
