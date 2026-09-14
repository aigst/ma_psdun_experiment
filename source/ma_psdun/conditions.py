"""Physical condition metadata shared by real and synthetic MA-PSDUN paths.

The exported directory names are labels, not numerical transmission values.
Keeping the optical-density and transmittance mappings together prevents a
training path from silently treating ``OD1`` as 10% transmission.
"""

from __future__ import annotations

from typing import Iterable

import torch


OD_OPTICAL_DENSITY = {
    "OD0": 0.0,
    "OD1": 0.3,
    "OD2": 0.5,
    "OD3": 1.0,
}

OD_TRANSMITTANCE = {
    "OD0": 1.0,
    "OD1": 0.5,
    "OD2": 0.32,
    "OD3": 0.1,
}

# Backward-compatible name used by older training scripts.  It deliberately
# means the physical transmission supplied to ConditionEncoder, not optical
# density and not an arbitrary decade schedule.
OD_VALUES = OD_TRANSMITTANCE


def od_to_transmittance(label: str) -> float:
    """Return the calibrated transmission for an ``OD0``-style folder name."""
    try:
        return float(OD_TRANSMITTANCE[label])
    except KeyError as exc:
        known = ", ".join(sorted(OD_TRANSMITTANCE))
        raise ValueError(f"unknown OD label {label!r}; expected one of {known}") from exc


def od_to_optical_density(label: str) -> float:
    """Return optical density (``-log10(transmission)``) for a folder name."""
    try:
        return float(OD_OPTICAL_DENSITY[label])
    except KeyError as exc:
        known = ", ".join(sorted(OD_OPTICAL_DENSITY))
        raise ValueError(f"unknown OD label {label!r}; expected one of {known}") from exc


def condition_tensor(labels: Iterable[str], device=None, dtype=torch.float32) -> torch.Tensor:
    """Build the transmission feature used by :class:`ConditionEncoder`.

    The first and third features preserve the historical fixed rate and
    wavelength convention; only the middle feature is calibrated from the OD
    folder name.
    """
    values = [od_to_transmittance(label) for label in labels]
    intensity = torch.tensor(values, device=device, dtype=dtype)
    return torch.stack([torch.ones_like(intensity), intensity,
                        torch.full_like(intensity, 550.0)], dim=-1)


__all__ = [
    "OD_OPTICAL_DENSITY",
    "OD_TRANSMITTANCE",
    "OD_VALUES",
    "od_to_transmittance",
    "od_to_optical_density",
    "condition_tensor",
]
