import torch

from ma_psdun.core import MeasurementOperator, structured_images
from ma_psdun.eval import image_metrics


def test_zero_mean_measurement_operator_has_spatially_informative_backprojection():
    op = MeasurementOperator(n=32 * 32, m=32 * 32 // 2, seed=123)
    x = structured_images(1, 32, seed=999)
    bp = op.adjoint(op.forward(x)).reshape(1, 1, 32, 32)
    corr = torch.corrcoef(torch.stack([bp.flatten(), x.flatten()]))[0, 1]
    assert float(corr) > 0.05


def test_measurement_operator_uses_binary_random_patterns():
    op = MeasurementOperator(n=64, m=32, seed=1)
    assert set(op.patterns.unique().tolist()) == {0.0, 1.0}
    assert 0.35 < float(op.patterns.mean()) < 0.65


def test_centered_operator_is_a_true_adjoint():
    op = MeasurementOperator(n=64, m=32, seed=1)
    x = torch.randn(3, 64)
    y = torch.randn(3, 32)
    lhs = (op.forward(x) * y).sum()
    rhs = (x * op.adjoint(y)).sum()
    assert torch.allclose(lhs, rhs, rtol=1e-5, atol=1e-5)
