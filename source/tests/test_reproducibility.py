import torch

from ma_psdun.core import MeasurementOperator, structured_images, simulate_measurements


def test_structured_target_is_deterministic_for_fixed_seed():
    a = structured_images(1, 16, seed=42)
    b = structured_images(1, 16, seed=42)
    assert torch.equal(a, b)


def test_fixed_simulation_is_reproducible():
    op = MeasurementOperator(16, 4, seed=3)
    x = structured_images(1, 4, seed=42)
    raw1, dark1 = simulate_measurements(x, op, torch.tensor([0.1]), seed=9)
    raw2, dark2 = simulate_measurements(x, op, torch.tensor([0.1]), seed=9)
    assert torch.equal(raw1, raw2)
    assert torch.equal(dark1, dark2)
