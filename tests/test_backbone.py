import torch
from src.models.backbone import TemporalBackbone


def test_backbone_shape():
    b = TemporalBackbone(n_channels=18, hidden=32, n_heads=4, n_tcn_layers=2, n_transformer_layers=1)
    x = torch.randn(4, 16, 18)
    out = b(x)
    assert out.shape == (4, 32)
    assert torch.isfinite(out).all()


def test_backbone_grad():
    b = TemporalBackbone(n_channels=6, hidden=16, n_heads=2, n_tcn_layers=1, n_transformer_layers=1)
    x = torch.randn(2, 8, 6, requires_grad=True)
    out = b(x)
    out.sum().backward()
    assert x.grad is not None
