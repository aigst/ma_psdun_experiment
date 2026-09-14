import torch
import pytest

from ring_blind_calibrate import apply_calibration


def test_identity_calibration_is_exact():
    value = torch.rand(2, 1, 4, 4)
    assert apply_calibration(value, {"kind": "identity"}) is value


def test_calibrations_remain_in_image_range():
    value = torch.linspace(0, 1, 16).reshape(1, 1, 4, 4)
    specs = [
        {"kind": "gamma", "gamma": 1.2},
        {"kind": "contrast", "low": 0.1, "high": 0.9},
        {"kind": "sigmoid", "threshold": 0.5, "temperature": 0.08},
        {"kind": "unsharp", "amount": 0.5},
    ]
    for spec in specs:
        calibrated = apply_calibration(value, spec)
        assert float(calibrated.min()) >= 0.0
        assert float(calibrated.max()) <= 1.0


def test_unknown_calibration_is_rejected():
    with pytest.raises(ValueError, match="unknown calibration"):
        apply_calibration(torch.zeros(1, 1, 2, 2), {"kind": "not-a-calibration"})
