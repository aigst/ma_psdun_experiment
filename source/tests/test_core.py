import torch

from ma_psdun.core import MeasurementOperator, ConditionEncoder, TCM, simulate_measurements
from ma_psdun.model import MAPSDUN


def test_measurement_operator_is_rectangular_and_adjoint_shapes():
    op = MeasurementOperator(n=16, m=4, seed=7, device="cpu")
    assert op.patterns.shape == (4, 16)
    x = torch.rand(2, 16)
    y = op.forward(x)
    assert y.shape == (2, 4)
    assert op.adjoint(y).shape == x.shape


def test_condition_encoder_has_stable_log_scales():
    enc = ConditionEncoder()
    out = enc(torch.tensor([[0.10, 0.001, 400.0], [0.50, 1.0, 1700.0]]))
    assert out.shape == (2, 32)
    assert torch.isfinite(out).all()


def test_tcm_correction_is_bounded_relative_residual():
    tcm = TCM(channels=8, max_relative_residual=0.25)
    raw = torch.randn(2, 1, 12)
    dark = torch.randn(2, 1, 12)
    corrected = tcm(raw, dark)
    baseline = raw - dark
    bound = 0.25 * (baseline.abs() + 1e-3)
    assert torch.all((corrected - baseline).abs() <= bound + 1e-5)


def test_simulated_raw_and_dark_are_per_measurement_sequences():
    op = MeasurementOperator(n=16, m=4, seed=7, device="cpu")
    raw, dark = simulate_measurements(torch.rand(2, 16), op, torch.tensor([0.1, 1.0]))
    assert raw.shape == (2, 4)
    assert dark.shape == (2, 4)
    assert float(raw.mean()) > 0.0


def test_binary_operator_centering_matches_centered_measurements():
    op = MeasurementOperator(n=16, m=8, seed=7, device="cpu")
    x = torch.rand(2, 16)
    raw = op.forward_raw(x)
    assert torch.allclose(op.center_measurement(raw), op.forward(x), atol=0.2)


def test_model_keeps_batch_dimension_in_measurement_consistency():
    op = MeasurementOperator(n=16, m=4, seed=7, device="cpu")
    model = MAPSDUN(op, stages=1)
    raw, dark = simulate_measurements(torch.rand(2, 16), op, torch.tensor([0.1, 1.0]))
    cond = torch.tensor([[0.1, 0.1, 550.0], [0.1, 1.0, 550.0]])
    pred, corrected = model(raw[:, None], dark[:, None], cond, (4, 4))
    assert pred.shape == (2, 1, 4, 4)
    assert corrected.shape == (2, 4)


def test_anisotropic_psf_has_valid_forward_and_adjoint_shapes():
    op = MeasurementOperator(n=16, m=8, seed=4, psf_sigma=0.8, psf_sigma_y=1.2, psf_angle=25.0, image_hw=(4, 4))
    x = torch.rand(2, 16)
    y = op.forward(x)
    assert y.shape == (2, 8)
    assert op.adjoint(y).shape == x.shape
