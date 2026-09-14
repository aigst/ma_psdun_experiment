"""Empirical detector-noise analysis and sequence replay.

The acquisition export stores ``dark, bright, dark, bright, ...`` values in
each ``traindata.txt``.  This module treats the 4096 dark entries as a time
series, removes only a fitted low-order drift for spectral analysis, and keeps
the normalized residual sequence so a simulator can replay its temporal
correlation.  Absolute ADC units are reported but are not silently mixed with
the normalized synthetic image scale; callers choose ``noise_scale`` during
calibration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def load_dark_sequence(path: str | Path) -> np.ndarray:
    """Load and validate the dark entries from an interleaved acquisition."""
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if values.size < 4 or values.size % 2:
        raise ValueError(f"{path}: expected an even interleaved dark/bright sequence")
    dark = values[0::2]
    if not np.isfinite(dark).all():
        raise ValueError(f"{path}: dark sequence contains non-finite values")
    return dark.astype(np.float32)


def detrend_sequence(sequence: Sequence[float], degree: int = 1) -> np.ndarray:
    """Remove a polynomial drift while preserving the acquisition ordering."""
    x = np.asarray(sequence, dtype=np.float64).reshape(-1)
    if x.size < degree + 2:
        raise ValueError("sequence is too short for requested detrending degree")
    grid = np.linspace(-1.0, 1.0, x.size, dtype=np.float64)
    coeff = np.polyfit(grid, x, degree)
    trend = np.polyval(coeff, grid)
    return (x - trend).astype(np.float32)


def power_spectral_density(sequence: Sequence[float], sample_rate: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return one-sided periodogram frequencies and power values."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    residual = np.asarray(sequence, dtype=np.float64).reshape(-1)
    if residual.size < 4:
        raise ValueError("sequence is too short for PSD")
    window = np.hanning(residual.size)
    scale = float(np.sum(window * window) * sample_rate)
    spectrum = np.fft.rfft((residual - residual.mean()) * window)
    psd = (np.abs(spectrum) ** 2 / max(scale, 1e-12)).astype(np.float64)
    frequencies = np.fft.rfftfreq(residual.size, d=1.0 / sample_rate)
    return frequencies.astype(np.float64), psd


def _spectral_slope(frequency: np.ndarray, psd: np.ndarray) -> float | None:
    mask = (frequency > 0) & np.isfinite(psd) & (psd > 0)
    if int(mask.sum()) < 3:
        return None
    x = np.log10(frequency[mask])
    y = np.log10(psd[mask])
    return float(np.polyfit(x, y, 1)[0])


def _spectral_peaks(frequency: np.ndarray, psd: np.ndarray, count: int = 5) -> list[dict]:
    mask = frequency > 0
    if not np.any(mask):
        return []
    indices = np.flatnonzero(mask)
    ranked = indices[np.argsort(psd[indices])[::-1][:count]]
    return [{"bin": int(index), "frequency": float(frequency[index]), "power": float(psd[index])}
            for index in ranked]


def _mains_checks(frequency: np.ndarray, psd: np.ndarray, sample_rate: float) -> dict:
    """Report nearest 50/60 Hz bins when the supplied rate makes them observable."""
    checks = {}
    for line_hz in (50.0, 60.0):
        if line_hz >= sample_rate / 2.0:
            checks[str(int(line_hz))] = None
            continue
        index = int(np.argmin(np.abs(frequency - line_hz)))
        local = psd[max(1, index - 1): min(len(psd), index + 2)]
        start, stop = max(1, index - 5), min(len(psd), index + 6)
        window = psd[start:stop]
        center = index - start
        background = np.delete(window, np.arange(max(0, center - 1), min(len(window), center + 2)))
        checks[str(int(line_hz))] = {
            "nearest_frequency": float(frequency[index]),
            "power": float(psd[index]),
            "local_peak_power": float(local.max()),
            "background_median": float(np.median(background)) if background.size else None,
            "peak_to_background": float(local.max() / max(float(np.median(background)), 1e-12)) if background.size else None,
        }
    return checks


def analyze_dark_sequence(
    sequence: Sequence[float],
    *,
    sample_rate: float = 1.0,
    low_frequency_fraction: float = 0.05,
    keep_sequence: bool = True,
) -> dict:
    """Compute time-domain and PSD diagnostics for one dark acquisition."""
    raw = np.asarray(sequence, dtype=np.float32).reshape(-1)
    if raw.size < 4:
        raise ValueError("dark sequence is too short")
    residual = detrend_sequence(raw)
    frequency, psd = power_spectral_density(residual, sample_rate)
    positive = frequency > 0
    total_power = float(psd[positive].sum())
    nyquist = sample_rate / 2.0
    low_cut = max(sample_rate * low_frequency_fraction, sample_rate / raw.size)
    low_power = float(psd[(frequency > 0) & (frequency <= low_cut)].sum())
    diff = np.diff(raw)
    residual_diff = np.diff(residual)
    slope = _spectral_slope(frequency, psd)
    record = {
        "count": int(raw.size),
        "mean": float(raw.mean()),
        "variance": float(raw.var()),
        "std": float(raw.std()),
        "detrended_std": float(residual.std()),
        "adjacent_diff_std": float(diff.std()),
        "detrended_adjacent_diff_std": float(residual_diff.std()),
        "white_noise_std_estimate": float(residual_diff.std() / np.sqrt(2.0)),
        "low_frequency_cut": float(low_cut),
        "low_frequency_power_fraction": float(low_power / max(total_power, 1e-12)),
        "nyquist": float(nyquist),
        "spectral_slope": slope,
        "one_over_f_indicator": bool(slope is not None and slope < -0.2),
        "mains_frequency_checks": _mains_checks(frequency, psd, sample_rate),
        "spectral_peaks": _spectral_peaks(frequency, psd),
    }
    if keep_sequence:
        scale = max(float(residual.std()), 1e-12)
        record["normalized_residual"] = (residual / scale).astype(np.float32).tolist()
    return record


def analyze_traindata(path: str | Path, **kwargs) -> dict:
    """Analyze one interleaved ``traindata.txt`` file."""
    result = analyze_dark_sequence(load_dark_sequence(path), **kwargs)
    result["source_data"] = str(path)
    return result


def build_noise_profile(
    data_root: str | Path,
    *,
    sample_rate: float = 1.0,
    include_sequences: bool = True,
    max_sequences_per_od: int | None = None,
    include_objects: Sequence[str] | None = None,
) -> dict:
    """Analyze all acquisitions under ``data_root`` and aggregate by OD."""
    root = Path(data_root)
    files = sorted(root.glob("**/OD*/traindata.txt"))
    selected_objects = sorted(set(include_objects)) if include_objects is not None else None
    if selected_objects is not None:
        selected_set = set(selected_objects)
        files = [path for path in files if path.parent.parent.name in selected_set]
    if not files:
        raise FileNotFoundError(f"no OD/traindata.txt files under {root}")
    by_od: dict[str, list[dict]] = {}
    for path in files:
        od = path.parent.name
        record = analyze_traindata(path, sample_rate=sample_rate, keep_sequence=include_sequences)
        by_od.setdefault(od, []).append(record)
    aggregate = {}
    sequences = {}
    for od, records in sorted(by_od.items()):
        selected = records if max_sequences_per_od is None else records[:max_sequences_per_od]
        keys = ("mean", "variance", "std", "detrended_std", "adjacent_diff_std",
                "detrended_adjacent_diff_std", "white_noise_std_estimate",
                "low_frequency_power_fraction")
        aggregate[od] = {
            "count": len(records),
            "metrics_mean": {key: float(np.mean([row[key] for row in records])) for key in keys},
            "metrics_median": {key: float(np.median([row[key] for row in records])) for key in keys},
            "spectral_slope_median": (
                float(np.median([row["spectral_slope"] for row in records if row["spectral_slope"] is not None]))
                if any(row["spectral_slope"] is not None for row in records) else None
            ),
            "one_over_f_fraction": float(np.mean([bool(row["one_over_f_indicator"]) for row in records])),
            "sources": [row["source_data"] for row in records],
        }
        if include_sequences:
            sequences[od] = [row["normalized_residual"] for row in selected]
    profile = {
        "schema": "ma-psdun-empirical-noise-v1",
        "data_root": str(root),
        "sample_rate": float(sample_rate),
        "file_count": len(files),
        "source_objects": sorted({path.parent.parent.name for path in files}),
        "by_od": aggregate,
    }
    if include_sequences:
        profile["sequences"] = sequences
    return profile


def save_noise_profile(profile: Mapping, path: str | Path) -> None:
    Path(path).write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n")


def load_noise_profile(path: str | Path) -> dict:
    profile = json.loads(Path(path).read_text())
    if profile.get("schema") != "ma-psdun-empirical-noise-v1":
        raise ValueError(f"unsupported noise profile schema: {profile.get('schema')!r}")
    if not profile.get("sequences"):
        raise ValueError("noise profile has no replay sequences")
    return profile


def _sequence_tensor(sequence: Sequence[float], length: int, device, dtype) -> torch.Tensor:
    values = torch.as_tensor(sequence, device=device, dtype=dtype).reshape(1, 1, -1)
    if values.shape[-1] != length:
        values = F.interpolate(values, size=length, mode="linear", align_corners=False)
    return values.reshape(length)


def sample_noise_sequences(
    profile: Mapping,
    batch_size: int,
    length: int,
    *,
    generator: torch.Generator,
    device,
    dtype=torch.float32,
    od_labels: Iterable[str] | None = None,
) -> torch.Tensor:
    """Replay normalized empirical sequences with deterministic torch RNG."""
    if batch_size < 1 or length < 1:
        raise ValueError("batch_size and length must be positive")
    sequences = profile.get("sequences", {})
    global_sequences = [sequence for rows in sequences.values() for sequence in rows]
    if not global_sequences:
        raise ValueError("noise profile has no sequences")
    labels = list(od_labels) if od_labels is not None else [None] * batch_size
    if len(labels) != batch_size:
        raise ValueError("od_labels length must match batch_size")
    output = []
    for label in labels:
        candidates = sequences.get(label, global_sequences) if label is not None else global_sequences
        if not candidates:
            candidates = global_sequences
        index = int(torch.randint(len(candidates), (), generator=generator, device=device))
        output.append(_sequence_tensor(candidates[index], length, device, dtype))
    return torch.stack(output)


__all__ = [
    "load_dark_sequence",
    "detrend_sequence",
    "power_spectral_density",
    "analyze_dark_sequence",
    "analyze_traindata",
    "build_noise_profile",
    "save_noise_profile",
    "load_noise_profile",
    "sample_noise_sequences",
]
