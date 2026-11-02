import torch
from src.models.fusion import MultimodalFusion


def test_fusion_shape():
    m = MultimodalFusion(n_channels=6, hidden=16, n_heads=2, n_context=4, n_graph_nodes=4)
    x = torch.randn(3, 8, 6)
    g = torch.randn(3, 4, 4)
    c = torch.randn(3, 4)
    out = m(x, g, c)
    assert out.shape == (3, 16)
    assert torch.isfinite(out).all()
