import pytest
import torch

from ring_blind_blend import blend_predictions, require_matching_pack, select_candidate


def test_blend_endpoints_are_exact():
    old = torch.rand(2, 1, 4, 4)
    new = torch.rand(2, 1, 4, 4)
    assert torch.equal(blend_predictions(old, new, 1.0), old)
    assert torch.equal(blend_predictions(old, new, 0.0), new)


def test_blend_rejects_invalid_weight_and_shape():
    value = torch.zeros(1, 1, 4, 4)
    with pytest.raises(ValueError, match="old_weight"):
        blend_predictions(value, value, 1.1)
    with pytest.raises(ValueError, match="shape mismatch"):
        blend_predictions(value, torch.zeros(2, 1, 4, 4), 0.5)


def test_pack_targets_must_match_exactly():
    pack = {
        "objects": ["test"],
        "target": torch.zeros(1, 1, 4, 4),
        "audit_objects": ["audit"],
        "audit_target": torch.zeros(1, 1, 4, 4),
    }
    changed = {**pack, "target": torch.ones(1, 1, 4, 4)}
    with pytest.raises(ValueError, match="target differ"):
        require_matching_pack(pack, changed)


def test_selection_can_choose_old_family_endpoint():
    target = torch.linspace(0.05, 0.95, 64).reshape(1, 1, 8, 8)
    old = target.clone()
    new = target.flip(-1)
    selected, rows = select_candidate(old, new, target)
    assert selected["old_weight"] == 1.0
    assert selected["validation"]["ssim"] == pytest.approx(1.0)
    assert len(rows) > 100
