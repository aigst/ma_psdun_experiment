"""Fold-local optical-density response fitting and joint sampling."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .conditions import OD_OPTICAL_DENSITY


OD_LABELS = ("OD0", "OD1", "OD2", "OD3")
OD_DENSITIES = np.asarray(
    [OD_OPTICAL_DENSITY[label] for label in OD_LABELS], dtype=np.float64
)


def robust_bucket_response(values: np.ndarray, trim_fraction: float = 0.01) -> float:
    """Return a robust absolute response from interleaved dark/raw buckets."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 8 or values.size % 2:
        raise ValueError(f"expected an even dark/raw sequence, got {values.size} values")
    if not 0.0 <= trim_fraction < 0.25:
        raise ValueError(f"trim_fraction must be in [0, 0.25), got {trim_fraction}")
    signal = values[1::2] - values[0::2]
    signal = signal[np.isfinite(signal)]
    if signal.size < 4:
        raise ValueError("not enough finite dark-subtracted buckets")
    if trim_fraction:
        low, high = np.quantile(signal, [trim_fraction, 1.0 - trim_fraction])
        signal = signal[(signal >= low) & (signal <= high)]
    response = float(signal.mean())
    if not math.isfinite(response) or response <= 0.0:
        raise ValueError(f"dark-subtracted response must be finite and positive, got {response}")
    return response


def fit_exponential_curve(
    responses: Sequence[float],
    densities: Sequence[float] = OD_DENSITIES,
    *,
    k_min: float = 0.01,
    k_max: float = 16.0,
    grid_size: int = 8192,
) -> dict:
    """Fit ``S(D) = A exp(-kD) + b`` with ``A,k>0`` and ``b>=0``.

    Only four OD measurements are available, so a dense one-dimensional grid
    over ``k`` is more reproducible than an unconstrained nonlinear optimizer.
    For every candidate ``k``, constrained least squares solves ``A`` and ``b``.
    """
    y = np.asarray(responses, dtype=np.float64)
    d = np.asarray(densities, dtype=np.float64)
    if y.shape != d.shape or y.ndim != 1 or y.size < 3:
        raise ValueError(f"responses and densities must be matching 1-D arrays, got {y.shape} and {d.shape}")
    if not np.isfinite(y).all() or np.any(y <= 0.0):
        raise ValueError(f"responses must be finite and positive, got {y.tolist()}")
    if not np.isfinite(d).all() or np.any(d < 0.0):
        raise ValueError(f"densities must be finite and non-negative, got {d.tolist()}")
    if not 0.0 < k_min < k_max or grid_size < 128:
        raise ValueError("invalid k search range")

    response_scale = float(y[0])
    y_normalized = y / response_scale
    k_grid = np.geomspace(k_min, k_max, int(grid_size), dtype=np.float64)
    x = np.exp(-k_grid[:, None] * d[None, :])
    x_mean = x.mean(axis=1)
    y_mean = float(y_normalized.mean())
    centered_x = x - x_mean[:, None]
    denominator = np.sum(centered_x * centered_x, axis=1)
    a = np.sum(centered_x * (y_normalized - y_mean)[None, :], axis=1) / np.maximum(denominator, 1e-18)
    b = y_mean - a * x_mean

    boundary = b < 0.0
    if np.any(boundary):
        xb = x[boundary]
        a[boundary] = (xb @ y_normalized) / np.maximum(np.sum(xb * xb, axis=1), 1e-18)
        b[boundary] = 0.0
    valid = (a > 0.0) & (b >= 0.0) & np.isfinite(a) & np.isfinite(b)
    prediction = a[:, None] * x + b[:, None]
    mse = np.mean((prediction - y_normalized[None, :]) ** 2, axis=1)
    mse[~valid] = np.inf
    index = int(np.argmin(mse))
    if not np.isfinite(mse[index]):
        raise RuntimeError("no feasible exponential attenuation fit")

    fitted_a = float(a[index])
    fitted_k = float(k_grid[index])
    fitted_b = float(b[index])
    fitted = fitted_a * np.exp(-fitted_k * d) + fitted_b
    fitted_s0 = fitted_a + fitted_b
    simulation_a = fitted_a / fitted_s0
    simulation_b = fitted_b / fitted_s0
    simulation_fitted = simulation_a * np.exp(-fitted_k * d) + simulation_b
    residual = y_normalized - fitted
    total_variation = float(np.sum((y_normalized - y_normalized.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual * residual)) / max(total_variation, 1e-18)
    return {
        "A": fitted_a * response_scale,
        "k": fitted_k,
        "b": fitted_b * response_scale,
        "response_scale_adc": response_scale,
        "A_normalized": simulation_a,
        "b_normalized": simulation_b,
        "measured_responses_adc": y.tolist(),
        "measured_responses_normalized": y_normalized.tolist(),
        "fitted_responses_adc": (fitted * response_scale).tolist(),
        "fitted_responses_normalized": simulation_fitted.tolist(),
        "rmse_normalized": float(np.sqrt(mse[index])),
        "r2": r2,
        "monotonic_measurement": bool(np.all(np.diff(y) <= 0.0)),
    }


def fit_attenuation_distribution(
    data_root: str | Path,
    object_names: Sequence[str],
    *,
    trim_fraction: float = 0.01,
) -> dict:
    """Fit one response curve per object and return its joint empirical law."""
    root = Path(data_root)
    names = sorted(set(object_names))
    if not names:
        raise ValueError("object_names must not be empty")
    fits = []
    for name in names:
        paths = [root / name / label / "traindata.txt" for label in OD_LABELS]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{name} is missing OD captures: {missing}")
        responses = [robust_bucket_response(np.loadtxt(path), trim_fraction) for path in paths]
        row = fit_exponential_curve(responses)
        row.update({
            "object": name,
            "source_data": [str(path) for path in paths],
        })
        fits.append(row)

    population = [
        {
            "object": row["object"],
            "A": float(row["A_normalized"]),
            "k": float(row["k"]),
            "b": float(row["b_normalized"]),
        }
        for row in fits
    ]
    values = np.asarray([[row["A"], row["k"], row["b"]] for row in population])
    return {
        "schema": "ma-psdun-attenuation-exp-v1",
        "model": "S(D) = A * exp(-k * D) + b",
        "od_labels": list(OD_LABELS),
        "optical_densities": OD_DENSITIES.tolist(),
        "response_estimator": {
            "signal": "raw-dark",
            "statistic": "trimmed_mean",
            "trim_fraction_each_tail": float(trim_fraction),
        },
        "normalization": (
            "Raw ADC fits are retained for audit. Sampling parameters are divided "
            "by each fitted S(0)=A+b because acquisition batches use incompatible ADC units."
        ),
        "source_objects": names,
        "fits": fits,
        "sampling_population": population,
        "sampling_summary": {
            "count": len(population),
            "mean": dict(zip(("A", "k", "b"), values.mean(axis=0).tolist())),
            "std": dict(zip(("A", "k", "b"), values.std(axis=0).tolist())),
            "sampling": "joint object-level empirical bootstrap",
        },
    }


def sample_joint_attenuation(
    profile: Mapping,
    batch_size: int,
    *,
    generator: torch.Generator,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    amplitude_jitter_log_std: float = 0.04,
) -> dict[str, torch.Tensor | list[str]]:
    """Bootstrap complete ``A/k/b`` tuples instead of sampling marginals."""
    if profile.get("schema") != "ma-psdun-attenuation-exp-v1":
        raise ValueError(f"unsupported attenuation profile: {profile.get('schema')!r}")
    population = profile.get("sampling_population", [])
    if not population:
        raise ValueError("attenuation profile has no sampling population")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    params = torch.tensor(
        [[row["A"], row["k"], row["b"]] for row in population],
        device=device,
        dtype=dtype,
    )
    indices = torch.randint(len(population), (batch_size,), generator=generator, device=device)
    sampled = params.index_select(0, indices)
    if amplitude_jitter_log_std > 0.0:
        gain = torch.exp(
            torch.randn(batch_size, generator=generator, device=device, dtype=dtype)
            * float(amplitude_jitter_log_std)
        )
        sampled[:, 0] *= gain
        sampled[:, 2] *= gain
    return {
        "A": sampled[:, 0],
        "k": sampled[:, 1],
        "b": sampled[:, 2],
        "source_index": indices,
        "source_object": [population[int(i)]["object"] for i in indices.detach().cpu()],
    }


__all__ = [
    "OD_DENSITIES",
    "OD_LABELS",
    "fit_attenuation_distribution",
    "fit_exponential_curve",
    "robust_bucket_response",
    "sample_joint_attenuation",
]
