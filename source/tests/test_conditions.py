import pytest
import torch

from ma_psdun.conditions import (
    OD_OPTICAL_DENSITY,
    OD_TRANSMITTANCE,
    condition_tensor,
    od_to_optical_density,
    od_to_transmittance,
)


def test_od_mapping_matches_calibrated_physical_values():
    assert OD_OPTICAL_DENSITY == {"OD0": 0.0, "OD1": 0.3, "OD2": 0.5, "OD3": 1.0}
    assert OD_TRANSMITTANCE == {"OD0": 1.0, "OD1": 0.5, "OD2": 0.32, "OD3": 0.1}
    assert [od_to_transmittance(name) for name in ("OD0", "OD1", "OD2", "OD3")] == [1.0, 0.5, 0.32, 0.1]


def test_condition_tensor_uses_transmittance_as_encoder_intensity():
    cond = condition_tensor(["OD0", "OD1", "OD2", "OD3"])
    assert cond.shape == (4, 3)
    assert torch.allclose(cond[:, 1], torch.tensor([1.0, 0.5, 0.32, 0.1]))
    assert torch.all(cond[:, 0] == 1.0)
    assert torch.all(cond[:, 2] == 550.0)


def test_unknown_od_is_rejected():
    with pytest.raises(ValueError, match="unknown OD label"):
        od_to_transmittance("OD9")
    with pytest.raises(ValueError, match="unknown OD label"):
        od_to_optical_density("OD9")
