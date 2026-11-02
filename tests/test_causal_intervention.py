import torch
from src.models.causal_intervention import intervene, interventional_fidelity


def test_intervene():
    x = torch.ones(2, 4, 3)
    y = intervene(x, [0, 2], value=0.0)
    assert (y[:, :, 0] == 0).all() and (y[:, :, 2] == 0).all()
    assert (y[:, :, 1] == 1).all()


def test_fidelity():
    def afn(x):
        return x.flatten(1).sum(dim=1)
    x = torch.ones(4, 4, 3)
    fid = interventional_fidelity(afn, x, [0], [1])
    assert 0.0 <= fid <= 1.0
