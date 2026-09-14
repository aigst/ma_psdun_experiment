import numpy as np
import torch

from new_sample_train import _condition, capture_fusion_weights


def _captures():
    return [{"od": f"OD{i}"} for i in range(4)]


def test_capture_fusion_weights_are_finite_and_normalized():
    signals = np.asarray([
        [0.0, 1.0, 0.0, -1.0],
        [0.1, 0.9, 0.1, -0.9],
        [0.2, 0.8, 0.2, -0.8],
        [0.3, 0.7, 0.3, -0.7],
    ], dtype=np.float32)
    for fusion in ("mean", "weighted", "quality", "agreement", "median", "od0"):
        weights = capture_fusion_weights(signals, _captures(), fusion)
        assert weights.shape == (4,)
        assert np.isfinite(weights).all()
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()


def test_od0_fusion_is_explicit_and_does_not_depend_on_signal_values():
    captures = _captures()
    first = capture_fusion_weights(np.zeros((4, 8), dtype=np.float32), captures, "od0")
    second = capture_fusion_weights(np.random.default_rng(4).normal(size=(4, 8)).astype(np.float32), captures, "od0")
    assert np.array_equal(first, np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert np.array_equal(second, first)


def test_agreement_fusion_falls_back_for_single_capture():
    weights = capture_fusion_weights(np.ones((1, 8), dtype=np.float32), [{"od": "OD0"}], "agreement")
    assert np.array_equal(weights, np.asarray([1.0], dtype=np.float32))


def test_stats_condition_uses_target_free_measurement_metadata():
    samples = [
        {"raw": np.zeros(8, dtype=np.float32), "dark": np.zeros(8, dtype=np.float32),
         "stats": np.asarray([-5.0, -6.0], dtype=np.float32)},
        {"raw": np.zeros(8, dtype=np.float32), "dark": np.zeros(8, dtype=np.float32),
         "stats": np.asarray([0.5, -1.0], dtype=np.float32)},
    ]
    cond = _condition([0, 1], samples, torch.device("cpu"), "stats")
    assert cond.shape == (2, 3)
    assert torch.isfinite(cond).all()
    assert not torch.allclose(cond[0], cond[1])
    assert torch.all((cond[:, 1] > 0) & (cond[:, 1] < 1))
    weak = _condition([0, 1], samples, torch.device("cpu"), "stats_weak")
    assert torch.isfinite(weak).all()
    assert torch.max(torch.abs(weak[:, 1] - 0.98)) < 0.02
