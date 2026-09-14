import torch

from ma_psdun.core import MeasurementOperator, structured_images, simulate_measurements
from ma_psdun.model import MAPSDUN
from ma_psdun.eval import image_metrics


def test_experiment_condition_is_fixed_when_requested():
    op = MeasurementOperator(16, 8, seed=1)
    x = structured_images(2, 4, seed=3)
    intensity = torch.full((2,), 0.1)
    raw1, dark1 = simulate_measurements(x, op, intensity, seed=10)
    raw2, dark2 = simulate_measurements(x, op, intensity, seed=10)
    assert torch.equal(raw1, raw2)
    assert torch.equal(dark1, dark2)


def test_operator_seed_is_part_of_reproducible_configuration():
    a = MeasurementOperator(32, 16, seed=123)
    b = MeasurementOperator(32, 16, seed=123)
    assert torch.equal(a.patterns, b.patterns)
