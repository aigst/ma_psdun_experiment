import torch

from ma_psdun.core import MeasurementOperator
from ma_psdun.model import Prior, MAPSDUN
from ma_psdun.core import simulate_measurements


def test_measurement_operator_has_unit_scale_energy():
    op = MeasurementOperator(n=64, m=16, seed=1)
    x = torch.randn(4, 64)
    y = op.forward(x)
    assert torch.isfinite(y).all()
    assert y.abs().mean() < 2.0


def test_prior_preserves_input_structure_with_residual_update():
    prior = Prior()
    x = torch.rand(2, 1, 16, 16)
    c = torch.randn(2, 32)
    out = prior(x, c)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert (out - x).abs().mean() < 0.5


def test_model_initialization_does_not_collapse_to_zero():
    op = MeasurementOperator(n=64, m=16, seed=1)
    model = MAPSDUN(op, stages=1)
    raw, dark = simulate_measurements(torch.rand(2, 64), op, torch.tensor([0.1, 1.0]))
    pred, _ = model(raw[:, None], dark[:, None], torch.tensor([[.1,.1,550.],[.1,1.,550.]]), (8, 8))
    assert float(pred.std()) > 1e-3
