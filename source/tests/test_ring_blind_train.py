import pytest
import torch

from ma_psdun.core import MeasurementOperator
from new_sample_train import dihedral
from ring_blind_train import binary_loss, build_operator_views, validate_split


def test_binary_loss_is_finite_for_clamped_predictions():
    pred = torch.tensor([[[[0.0, 1.0], [0.2, 0.8]]]])
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    assert torch.isfinite(binary_loss(pred, target))


def test_dihedral_operator_augmentation_preserves_measurement_equation():
    size = 4
    patterns = torch.randint(0, 2, (7, size * size), dtype=torch.float32)
    image = torch.rand(3, 1, size, size)
    op = MeasurementOperator(size * size, len(patterns), patterns=patterns)
    baseline = op.forward(image.flatten(1))
    views = build_operator_views(op, size, True)
    for transform, view in enumerate(views):
        op.centered_patterns = view
        transformed = dihedral(image, transform)
        torch.testing.assert_close(op.forward(transformed.flatten(1)), baseline)


def test_split_rejects_circular_text_in_training_or_validation():
    available = ["plain-a", "plain-b", "ring", "test"]
    with pytest.raises(ValueError, match="forbidden circular-text"):
        validate_split(available, ["plain-a"], ["ring"], ["test"], [], ["ring"], True)


def test_split_accepts_disjoint_ring_blind_supervision():
    audit = validate_split(
        ["plain-a", "plain-b", "ring", "test"],
        ["plain-a"],
        ["plain-b"],
        ["test"],
        ["ring"],
        ["ring"],
        True,
    )
    assert audit["ring_blind_supervision"] is True
    assert audit["forbidden_intersection"] == {"train": [], "validation": []}
