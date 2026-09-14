import numpy as np
import torch

from ma_psdun.core import MeasurementOperator
from self_supervised_adapt import adapt_affine, candidate_grid


def test_target_free_affine_adaptation_is_finite_and_preserves_shape():
    op = MeasurementOperator(16, 8, seed=4)
    base = torch.rand(2, 1, 4, 4) * 0.8 + 0.1
    measured = op.forward(base.flatten(1)).detach()
    output, details = adapt_affine(base, measured, op, steps=3, lr=0.01, reg=0.01, tv_weight=0.0, obs_scale=1.0)
    assert output.shape == base.shape
    assert torch.isfinite(output).all()
    assert details["measurement_smooth_l1"] >= 0


def test_candidate_grid_is_predeclared_and_contains_identity():
    candidates = candidate_grid()
    assert len(candidates) > 1
    assert any(item["steps"] == 0 for item in candidates)
