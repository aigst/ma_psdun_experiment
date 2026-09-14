import torch

from ma_psdun.model import Prior


def test_prior_initialization_is_near_identity():
    prior = Prior()
    x = torch.rand(2, 1, 16, 16)
    c = torch.randn(2, 32)
    out = prior(x, c)
    assert (out - x).abs().mean() < 0.08
