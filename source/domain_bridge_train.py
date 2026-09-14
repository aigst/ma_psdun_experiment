"""Existing-data-only MA-PSDUN sim-to-real bridge experiment.

The script is deliberately object-level: the locked test objects never
enter model selection.  Each fold estimates nuisance ranges from its training
objects, pretrains on calibrated synthetic measurements, fine-tunes only the
measurement adapter and the last prior block, then optionally applies the
unlabelled obj12 consistency update and validation-selected TTO.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ma_psdun.core import MeasurementOperator
from ma_psdun.attenuation import (
    OD_DENSITIES,
    OD_LABELS,
    fit_attenuation_distribution,
    sample_joint_attenuation,
)
from ma_psdun.conditions import OD_TRANSMITTANCE
from ma_psdun.noise import (
    build_noise_profile,
    load_noise_profile,
    sample_noise_sequences,
    save_noise_profile,
)
from ma_psdun.eval import image_metrics
from ma_psdun.model import MAPSDUN
from new_sample_train import (
    _batch,
    _gaussian_smooth,
    _moving_average,
    _normalize,
    _weighted_loss,
    load_sample,
    metrics,
    read_patterns,
)


# The current deleted-object protocol keeps obj18/obj20 locked for the final
# comparison.  Older 25-object runs can still opt into the former split via
# ``--test-objects`` without changing this default.
TEST_OBJECTS = ["obj18-20260901", "obj20-20260901"]
OD_LEVELS = tuple(OD_TRANSMITTANCE[name] for name in ("OD0", "OD1", "OD2", "OD3"))
DEFAULT_OD_WEIGHTS = (0.50, 0.30, 0.15, 0.05)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def objects_from_samples(samples):
    return sorted({s["object"] for s in samples})


def split_folds(objects: list[str], seed: int = 20260902,
                test_objects: list[str] | tuple[str, ...] = TEST_OBJECTS,
                n_folds: int = 4):
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    test_objects = list(test_objects)
    missing_test = sorted(set(test_objects) - set(objects))
    if missing_test:
        raise ValueError(f"test objects are missing from labelled data: {missing_test}")
    eligible = [x for x in objects if x not in set(test_objects)]
    if len(eligible) < n_folds:
        raise ValueError(f"need at least one eligible object per fold, got {len(eligible)} for {n_folds} folds")
    # Use a fixed object-only permutation.  The fold sizes differ by at most
    # one when a caller uses a non-divisible dataset; the realised composition
    # is recorded in split.json.
    rng = random.Random(seed)
    rng.shuffle(eligible)
    base, remainder = divmod(len(eligible), n_folds)
    folds = []
    start = 0
    for fold_id in range(n_folds):
        size = base + (1 if fold_id < remainder else 0)
        folds.append(sorted(eligible[start:start + size]))
        start += size
    return folds


def load_unlabelled(root: str | Path, pattern_path: str | Path, preprocess: str,
                    detrend_width: int, gauss_sigma: float,
                    preserve_od_amplitude: bool = False):
    """Load obj12 measurements without reading its misspelled truth file."""
    root = Path(root)
    pattern = read_patterns(pattern_path)
    obj = root / "obj12-20260901"
    if not obj.is_dir():
        raise FileNotFoundError(obj)
    out = []
    for od_dir in sorted(obj.glob("OD*")):
        path = od_dir / "traindata.txt"
        values = np.loadtxt(path, dtype=np.float32)
        if values.size != pattern.shape[0] * 2:
            raise ValueError(f"{path}: {values.size} values, expected {pattern.shape[0] * 2}")
        dark, raw = values[0::2], values[1::2]
        if preserve_od_amplitude:
            raw_n = _preprocess_shared_signal(raw, dark, preprocess, detrend_width, gauss_sigma)
            dark_n = np.zeros_like(raw_n)
        else:
            raw_n, dark_n = _normalize(raw, dark, preprocess, detrend_width, gauss_sigma)
        out.append({"object": obj.name, "od": od_dir.name,
                    "raw": raw_n.astype(np.float32), "dark": dark_n.astype(np.float32),
                    "source_data": str(path)})
    if len(out) != 4:
        raise ValueError(f"obj12 expected four OD captures, got {len(out)}")
    raw = np.stack([x["raw"] for x in out]).astype(np.float32)
    dark = np.stack([x["dark"] for x in out]).astype(np.float32)
    if preserve_od_amplitude:
        scale = max(float((raw - dark).reshape(-1).std()), 1e-8)
        raw /= scale
        dark /= scale
    return {"object": obj.name, "od": "MULTI", "raw": raw, "dark": dark,
            "source_data": [x["source_data"] for x in out]}


def _preprocess_shared_signal(raw: np.ndarray, dark: np.ndarray, preprocess: str,
                              detrend_width: int, gauss_sigma: float) -> np.ndarray:
    """Remove per-capture baseline without applying a per-OD scale."""
    signal = np.asarray(raw, dtype=np.float64) - np.asarray(dark, dtype=np.float64)
    if preprocess in {"zscore", "difference"}:
        residual = signal - signal.mean()
    elif preprocess == "robust":
        residual = signal - np.median(signal)
    elif preprocess in {"detrend", "detrend_robust", "gauss_detrend", "gauss_detrend_robust"}:
        trend = (_gaussian_smooth(signal, gauss_sigma)
                 if preprocess.startswith("gauss") else
                 _moving_average(signal, detrend_width))
        residual = signal - trend
        residual -= np.median(residual) if preprocess.endswith("robust") else residual.mean()
    else:
        raise ValueError(f"unknown preprocess mode: {preprocess}")
    if not np.isfinite(residual).all():
        raise ValueError("shared OD preprocessing produced non-finite values")
    return residual.astype(np.float32)


def preserve_multi_od_amplitude(samples, preprocess: str, detrend_width: int,
                                gauss_sigma: float):
    """Reload each OD capture and normalize the four-channel stack once."""
    converted = []
    for sample in samples:
        paths = [Path(path) for path in sample["source_data"]]
        by_label = {path.parent.name: path for path in paths}
        if set(by_label) != set(OD_LABELS):
            raise ValueError(f"{sample['object']} does not have exactly {OD_LABELS}: {sorted(by_label)}")
        rows = []
        for label in OD_LABELS:
            values = np.loadtxt(by_label[label], dtype=np.float64)
            rows.append(_preprocess_shared_signal(
                values[1::2], values[0::2], preprocess, detrend_width, gauss_sigma,
            ))
        unscaled = np.stack(rows).astype(np.float32)
        shared_scale = max(float(unscaled.reshape(-1).std()), 1e-8)
        item = copy.deepcopy(sample)
        item["raw"] = (unscaled / shared_scale).astype(np.float32)
        item["dark"] = np.zeros_like(item["raw"], dtype=np.float32)
        per_od_std = unscaled.std(axis=1)
        item["shared_od_preprocess"] = {
            "shared_scale": shared_scale,
            "per_od_std_before_scale": per_od_std.astype(float).tolist(),
            "per_od_std_ratio_to_od0": (per_od_std / max(float(per_od_std[0]), 1e-12)).astype(float).tolist(),
        }
        converted.append(item)
    return converted


def fit_nuisance(train_samples, patterns: np.ndarray) -> dict:
    """Estimate low-dimensional nuisance ranges from training objects only.

    These are deliberately robust proxies.  The export has no repeated frame
    at identical pattern/time, so they are not claimed as physical absolute
    calibration.
    """
    noise, drift, scales = [], [], []
    for sample in train_samples:
        y = np.asarray(sample["raw"] - sample["dark"], dtype=np.float32)
        if y.ndim == 2:
            for row in y:
                noise.append(float(np.std(np.diff(row)) / math.sqrt(2.0)))
                smooth = np.convolve(np.pad(row, (63, 63), mode="reflect"), np.ones(127, dtype=np.float32) / 127, mode="valid")
                drift.append(float(np.std(smooth)))
                scales.append(float(np.std(row)))
        else:
            noise.append(float(np.std(np.diff(y)) / math.sqrt(2.0)))
            smooth = np.convolve(np.pad(y, (63, 63), mode="reflect"), np.ones(127, dtype=np.float32) / 127, mode="valid")
            drift.append(float(np.std(smooth)))
            scales.append(float(np.std(y)))
    # Label gradient energy is a fold-local PSF proxy; labels from val/test are
    # never touched here.
    grad = []
    for sample in train_samples:
        label = np.asarray(sample["label"], dtype=np.float32)
        grad.append(float(np.mean(np.abs(np.diff(label, axis=0))) + np.mean(np.abs(np.diff(label, axis=1)))))
    grad_energy = float(np.median(grad)) if grad else 0.2
    psf_sigma = float(np.clip(0.85 / max(grad_energy, 0.15), 0.45, 1.25))
    rel_noise = float(np.clip(np.median(noise) / max(np.median(scales), 1e-6), 0.01, 0.12))
    drift_std = float(np.clip(np.median(drift) / max(np.median(scales), 1e-6), 0.005, 0.10))
    photon_peak = float(np.clip(1.0 / max(rel_noise * rel_noise, 1e-5), 180.0, 900.0))
    return {
        "psf_sigma": psf_sigma,
        "psf_sigma_range": [max(0.35, psf_sigma * 0.8), min(1.6, psf_sigma * 1.2)],
        "relative_noise_std": rel_noise,
        "drift_std": drift_std,
        "photon_peak": photon_peak,
        "readout_noise_std": float(np.clip(np.median(noise) * 0.35, 0.002, 0.04)),
        "dark_noise_std": float(np.clip(np.median(noise) * 0.50, 0.003, 0.06)),
        "gain_range": [0.92, 1.08],
        "shift_pixels": 0.35,
        "nonlinearity": 0.015,
        "source_objects": sorted({s["object"] for s in train_samples}),
        "note": "fold-local proxy estimates; no repeated identical-frame calibration exists",
    }


def _gaussian_blur(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # A small separable kernel is sufficient for the measured 64x64 optical
    # bandwidth and keeps the simulator differentiable and cheap.
    radius = 3
    grid = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (grid[None, :] / sigma[:, None].clamp_min(0.15)) ** 2)
    kernel = kernel / kernel.sum(dim=1, keepdim=True)
    # Per-sample kernels are grouped by a scalar approximation to avoid a
    # Python loop in the large synthetic pretraining stream.
    k = kernel.mean(dim=0)
    k2 = k[:, None] * k[None, :]
    return F.conv2d(x, k2[None, None], padding=radius)


def synthetic_batch(train_samples, op: MeasurementOperator, nuisance: dict,
                    batch_size: int, device: torch.device, seed: int,
                    noise_profile=None, noise_mode: str = "poisson",
                    attenuation_profile=None, attenuation_jitter: float = 0.04,
                    return_metadata: bool = False):
    g = torch.Generator(device=device).manual_seed(seed)
    targets = []
    target_sources = []
    for i in range(batch_size):
        if train_samples and torch.rand((), generator=g, device=device).item() < 0.65:
            source_index = int(torch.randint(len(train_samples), (), generator=g, device=device))
            src = train_samples[source_index]["label"]
            t = torch.from_numpy(src)[None, None].to(device)
            if torch.rand((), generator=g, device=device).item() < 0.5:
                t = t.flip(-1)
            t = torch.rot90(t, int(torch.randint(4, (), generator=g, device=device)), (-2, -1))
            target_sources.append(train_samples[source_index]["object"])
        else:
            # Procedural targets prevent the network from memorising the 16
            # labelled objects while retaining sparse smooth structure.
            from ma_psdun.core import structured_images
            t = structured_images(1, 64, seed + i).reshape(1, 1, 64, 64).to(device)
            target_sources.append("procedural")
        targets.append(t)
    target = torch.cat(targets, dim=0).clamp(0, 1)
    sigma = torch.empty(batch_size, device=device).uniform_(
        nuisance["psf_sigma_range"][0], nuisance["psf_sigma_range"][1], generator=g)
    x = _gaussian_blur(target, sigma)
    # Differentiable subpixel shift sampled from the fold-local bounded range.
    shift = torch.empty(batch_size, 2, device=device).uniform_(-0.35, 0.35, generator=g)
    theta = torch.zeros(batch_size, 2, 3, device=device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1.0
    theta[:, :, 2] = shift / 32.0
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    ideal = op.forward_raw(x.flatten(1))
    m = ideal.shape[-1]
    tline = torch.linspace(-0.5, 0.5, m, device=device)[None, None, :]
    raw_channels, dark_channels = [], []
    attenuation = None
    if attenuation_profile is not None:
        attenuation = sample_joint_attenuation(
            attenuation_profile, batch_size, generator=g, device=device,
            dtype=ideal.dtype, amplitude_jitter_log_std=attenuation_jitter,
        )
        a = attenuation["A"]
        k = attenuation["k"]
        b = attenuation["b"]
        densities = torch.as_tensor(OD_DENSITIES, device=device, dtype=ideal.dtype)
        od_responses = a[:, None] * torch.exp(-k[:, None] * densities[None, :]) + b[:, None]
        ideal_shape = ideal / ideal.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    else:
        alpha = torch.empty(batch_size, device=device).uniform_(0.85, 1.15, generator=g)
        gain = torch.empty(batch_size, device=device).uniform_(0.92, 1.08, generator=g)
        od_responses = torch.stack([
            torch.full((batch_size,), level, device=device).pow(alpha) * gain
            for level in OD_LEVELS
        ], dim=1)
        ideal_shape = ideal
    drift_amp = torch.empty(batch_size, device=device).uniform_(0.5, 1.5, generator=g) * nuisance["drift_std"]
    for od_index, level in enumerate(OD_LEVELS):
        level_t = torch.full((batch_size, 1), level, device=device)
        signal = ideal_shape * od_responses[:, od_index, None]
        drift = 1.0 + drift_amp[:, None] * tline[:, 0]
        signal = signal * drift
        # Mild detector nonlinearity and an OD-dependent dark floor.
        signal = signal + nuisance["nonlinearity"] * signal.square()
        dark = 0.01 * (1.0 + 0.2 * tline[:, 0]) * (1.0 + level_t * 0.1)
        if noise_mode not in {"none", "poisson", "empirical", "hybrid"}:
            raise ValueError(f"unknown noise mode: {noise_mode}")
        if noise_mode in {"empirical", "hybrid"} and noise_profile is None:
            raise ValueError("noise_profile is required for empirical or hybrid noise")
        if noise_mode in {"poisson", "hybrid"}:
            rate = (signal + dark).clamp_min(0.0) * nuisance["photon_peak"]
            photon = torch.poisson(rate, generator=g) / nuisance["photon_peak"]
            readout = torch.randn(photon.shape, generator=g, device=device) * nuisance["readout_noise_std"]
            raw_obs = photon + readout
            dark_obs = dark + torch.randn(dark.shape, generator=g, device=device) * nuisance["dark_noise_std"]
        else:
            raw_obs, dark_obs = signal, dark
        if noise_mode in {"empirical", "hybrid"}:
            labels = [name for name in ("OD0", "OD1", "OD2", "OD3") if abs(OD_TRANSMITTANCE[name] - level) < 1e-8]
            label = labels[0] if labels else None
            replay_args = {"generator": g, "device": device, "dtype": signal.dtype,
                           "od_labels": [label] * batch_size}
            raw_obs = raw_obs + nuisance["noise_scale"] * sample_noise_sequences(noise_profile, batch_size, m, **replay_args)
            dark_obs = dark_obs + nuisance["dark_noise_scale"] * sample_noise_sequences(noise_profile, batch_size, m, **replay_args)
        raw_channels.append(raw_obs.float())
        dark_channels.append(dark_obs.float())
    raw = torch.stack(raw_channels, dim=1)
    dark = torch.stack(dark_channels, dim=1)
    cond = torch.tensor([[1.0, 1.0, 550.0]] * batch_size, device=device)
    if not return_metadata:
        return raw, dark, target, cond
    metadata = {
        "A": (attenuation["A"] if attenuation is not None else gain).detach(),
        "k": (attenuation["k"] if attenuation is not None else alpha * math.log(10.0)).detach(),
        "b": (attenuation["b"] if attenuation is not None else torch.zeros_like(gain)).detach(),
        "od_responses": od_responses.detach(),
        "attenuation_source_index": (
            attenuation["source_index"].detach()
            if attenuation is not None else torch.full((batch_size,), -1, device=device, dtype=torch.long)
        ),
        "attenuation_source_object": (
            attenuation["source_object"] if attenuation is not None else ["legacy"] * batch_size
        ),
        "target_source_object": target_sources,
    }
    return raw, dark, target, cond, metadata


def generate_synthetic_dataset(train_samples, op: MeasurementOperator, nuisance: dict,
                               attenuation_profile: dict, count: int, batch_size: int,
                               device: torch.device, seed: int, *, noise_profile=None,
                               noise_mode: str = "poisson", attenuation_jitter: float = 0.04):
    """Materialize a deterministic fold-local synthetic dataset on CPU."""
    if count < 1:
        raise ValueError("synthetic dataset count must be positive")
    tensors = {key: [] for key in (
        "raw", "dark", "target", "cond", "A", "k", "b", "od_responses",
        "attenuation_source_index",
    )}
    attenuation_sources = []
    target_sources = []
    for start in range(0, count, batch_size):
        current = min(batch_size, count - start)
        raw, dark, target, cond, metadata = synthetic_batch(
            train_samples, op, nuisance, current, device, seed + start,
            noise_profile=noise_profile, noise_mode=noise_mode,
            attenuation_profile=attenuation_profile,
            attenuation_jitter=attenuation_jitter, return_metadata=True,
        )
        for key, value in (("raw", raw), ("dark", dark), ("target", target), ("cond", cond)):
            tensors[key].append(value.detach().cpu())
        for key in ("A", "k", "b", "od_responses", "attenuation_source_index"):
            tensors[key].append(metadata[key].detach().cpu())
        attenuation_sources.extend(metadata["attenuation_source_object"])
        target_sources.extend(metadata["target_source_object"])
    packed = {key: torch.cat(parts, dim=0) for key, parts in tensors.items()}
    packed.update({
        "schema": "ma-psdun-synthetic-exp-attenuation-v1",
        "seed": int(seed),
        "count": int(count),
        "train_objects": sorted({sample["object"] for sample in train_samples}),
        "attenuation_source_object": attenuation_sources,
        "target_source_object": target_sources,
    })
    validate_synthetic_dataset(packed, attenuation_profile)
    return packed


def validate_synthetic_dataset(dataset: dict, attenuation_profile: dict) -> None:
    if dataset.get("schema") != "ma-psdun-synthetic-exp-attenuation-v1":
        raise ValueError(f"unsupported synthetic dataset schema: {dataset.get('schema')!r}")
    count = int(dataset["count"])
    tensor_keys = (
        "raw", "dark", "target", "cond", "A", "k", "b", "od_responses",
        "attenuation_source_index",
    )
    for key in tensor_keys:
        value = dataset.get(key)
        if not isinstance(value, torch.Tensor) or value.shape[0] != count:
            raise ValueError(f"synthetic dataset field {key!r} has invalid shape")
        if key != "attenuation_source_index" and not torch.isfinite(value).all():
            raise ValueError(f"synthetic dataset field {key!r} contains non-finite values")
    if dataset["raw"].shape != dataset["dark"].shape or dataset["raw"].shape[1] != 4:
        raise ValueError("synthetic raw/dark tensors must be matching four-OD stacks")
    if dataset["od_responses"].shape != (count, 4):
        raise ValueError("synthetic od_responses must have shape [count, 4]")
    expected = dataset["A"][:, None] * torch.exp(
        -dataset["k"][:, None] * torch.as_tensor(OD_DENSITIES, dtype=dataset["A"].dtype)[None, :]
    ) + dataset["b"][:, None]
    if not torch.allclose(dataset["od_responses"], expected, rtol=1e-5, atol=1e-6):
        raise ValueError("saved OD responses do not follow the sampled exponential curve")
    allowed = set(attenuation_profile["source_objects"])
    if not set(dataset["attenuation_source_object"]).issubset(allowed):
        raise ValueError("synthetic dataset sampled attenuation from a non-training object")
    population = attenuation_profile["sampling_population"]
    indices = dataset["attenuation_source_index"]
    if int(indices.min()) < 0 or int(indices.max()) >= len(population):
        raise ValueError("synthetic attenuation source index is out of range")
    expected_sources = [population[int(index)]["object"] for index in indices]
    if list(dataset["attenuation_source_object"]) != expected_sources:
        raise ValueError("synthetic attenuation source index/object mismatch")
    if len(dataset["target_source_object"]) != count:
        raise ValueError("synthetic target source manifest length does not match count")


def save_reload_synthetic_dataset(dataset: dict, path: Path,
                                  attenuation_profile: dict) -> tuple[dict, dict]:
    """Save, independently reload, and byte-check all generated tensors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, path)
    digest = sha256_file(path)
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    validate_synthetic_dataset(reloaded, attenuation_profile)
    tensor_keys = (
        "raw", "dark", "target", "cond", "A", "k", "b", "od_responses",
        "attenuation_source_index",
    )
    exact = (
        all(torch.equal(dataset[key], reloaded[key]) for key in tensor_keys)
        and dataset["attenuation_source_object"] == reloaded["attenuation_source_object"]
        and dataset["target_source_object"] == reloaded["target_source_object"]
    )
    if not exact:
        raise RuntimeError("synthetic dataset changed after save/reload")
    audit = {
        "status": "verified",
        "schema": reloaded["schema"],
        "path": str(path),
        "sha256": digest,
        "count": int(reloaded["count"]),
        "tensor_shapes": {key: list(reloaded[key].shape) for key in tensor_keys},
        "exact_tensor_reload": True,
        "train_objects": reloaded["train_objects"],
        "attenuation_source_objects": sorted(set(reloaded["attenuation_source_object"])),
    }
    return reloaded, audit


def synthetic_dataset_batch(dataset: dict, batch_size: int, device: torch.device,
                            seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(int(dataset["count"]), (batch_size,), generator=generator)
    return tuple(dataset[key].index_select(0, indices).to(device) for key in (
        "raw", "dark", "target", "cond",
    ))


def make_model(op, args, stages=None, shared_prior=None, lowpass=None, prior_scale=None):
    stages = args.stages if stages is None else stages
    shared_prior = args.shared_prior if shared_prior is None else shared_prior
    lowpass = args.lowpass_kernel if lowpass is None else lowpass
    prior_scale = args.prior_residual_scale if prior_scale is None else prior_scale
    model = MAPSDUN(op, stages=stages, backprojection_gain_init=4.0,
                    lowpass_kernel=lowpass, od_channels=4,
                    prior_residual_scale=prior_scale,
                    shared_prior=shared_prior,
                    preserve_od_amplitude=getattr(args, "preserve_od_amplitude", False)).to(op.patterns.device)
    model.tcm.od_logits.data.copy_(torch.log(torch.tensor(DEFAULT_OD_WEIGHTS, device=op.patterns.device)))
    return model


def real_batch(samples, names, device):
    indices = [i for i, s in enumerate(samples) if s["object"] in set(names)]
    if not indices:
        raise ValueError(f"empty object split {names}")
    raw, dark, target, cond = _batch(samples, indices, device, 0, "constant")
    return raw, dark, target, cond, [samples[i]["object"] for i in indices]


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def eval_batch(model, batch, op, use_od3=True):
    raw, dark, target, cond, names = batch
    with torch.no_grad():
        if use_od3:
            pred, y = model(raw, dark, cond, (64, 64))
        else:
            old = model.tcm.od_logits.detach().clone()
            model.tcm.od_logits.data[-1] = -20.0
            masked_raw, masked_dark = raw.clone(), dark.clone()
            masked_raw[:, 3] = 0.0; masked_dark[:, 3] = 0.0
            pred, y = model(masked_raw, masked_dark, cond, (64, 64))
            model.tcm.od_logits.data.copy_(old)
        overall = metrics(pred, target, y, op)
    rows = []
    for j, name in enumerate(names):
        one = metrics(pred[j:j + 1], target[j:j + 1], y[j:j + 1], op)
        one["object"] = name
        rows.append(one)
    return overall, rows, pred.detach().cpu(), target.detach().cpu()


def fine_tune(model, train_batch, val_batch, train_samples, nuisance, op, args,
              out: Path, seed: int, noise_profile=None, synthetic_dataset=None,
              attenuation_profile=None):
    # Freeze the high-capacity prior.  Only the physical adapter and final
    # prior block can absorb the small real-domain shift.
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("tcm") or name in {"backprojection_gain", "backprojection_bias", "rho_logits"}
                         or name.startswith(f"priors.{args.stages - 1}."))
    parameters = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(parameters, lr=args.finetune_lr, weight_decay=args.weight_decay)
    best = -float("inf"); best_state = None; log = []
    for epoch in range(args.finetune_epochs):
        model.train()
        if synthetic_dataset is None:
            syn = synthetic_batch(
                train_samples, op, nuisance, args.synthetic_batch, op.patterns.device,
                seed + epoch + 1000, noise_profile=noise_profile,
                noise_mode=args.noise_mode, attenuation_profile=attenuation_profile,
                attenuation_jitter=args.attenuation_jitter,
            )
        else:
            syn = synthetic_dataset_batch(
                synthetic_dataset, args.synthetic_batch, op.patterns.device,
                seed + epoch + 1000,
            )
        pred, y = model(train_batch[0], train_batch[1], train_batch[3], (64, 64))
        real_loss = _weighted_loss(pred, train_batch[2], y, op, args.consistency_weight,
                                   ssim_weight=args.ssim_weight, tv_weight=args.tv_weight,
                                   edge_weight=args.edge_weight)
        sp, sy, st, sc = syn
        syn_pred, syn_y = model(sp, sy, sc, (64, 64))
        syn_loss = _weighted_loss(syn_pred, st, syn_y, op, args.consistency_weight,
                                  ssim_weight=args.ssim_weight, tv_weight=args.tv_weight,
                                  edge_weight=args.edge_weight)
        loss = (real_loss + args.synthetic_mix * syn_loss) / (1.0 + args.synthetic_mix)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0); opt.step()
        model.eval()
        vm, _, _, _ = eval_batch(model, val_batch, op, True)
        rec = {"epoch": epoch, "loss": float(loss.detach()), "real_loss": float(real_loss.detach()),
               "synthetic_loss": float(syn_loss.detach()), **vm}
        log.append(rec)
        if vm["ssim"] > best:
            best = vm["ssim"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % args.log_every == 0 or epoch == args.finetune_epochs - 1:
            print(json.dumps(rec), flush=True)
    if best_state is None:
        raise RuntimeError("no validation checkpoint")
    model.load_state_dict(best_state)
    save_json(out / "validation.json", log)
    torch.save({"model": model.state_dict(), "best_validation_ssim": best,
                "seed": seed, "stage": "real_finetune"}, out / "checkpoint_best.pt")
    return model, best


def adapt_obj12(model, obj12, val_batch, op, args, out: Path):
    before_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    before_vm, _, _, _ = eval_batch(model, val_batch, op, True)
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("tcm") or name in {"backprojection_gain", "backprojection_bias", "rho_logits"})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.uda_lr, weight_decay=1e-6)
    raw = torch.from_numpy(obj12["raw"])[None].to(op.patterns.device)
    dark = torch.from_numpy(obj12["dark"])[None].to(op.patterns.device)
    cond = torch.tensor([[1.0, 1.0, 550.0]], device=op.patterns.device)
    obs = raw.sub(dark).mean(dim=1)
    obs = (obs - obs.mean(dim=-1, keepdim=True)) / obs.std(dim=-1, keepdim=True).clamp_min(1e-4) * 0.35
    rows = []
    for step in range(args.uda_steps):
        model.train()
        noise = torch.randn_like(raw) * args.uda_input_noise
        p1, y1 = model(raw + noise, dark, cond, (64, 64))
        p2, y2 = model(raw - noise, dark, cond, (64, 64))
        consistency = (op.forward(p1.flatten(1)) - obs).abs().mean()
        temporal = (p1 - p2).abs().mean()
        od_consistency = (y1 - y2).abs().mean()
        loss = consistency + args.uda_temporal_weight * temporal + args.uda_od_weight * od_consistency
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        rows.append({"step": step, "loss": float(loss.detach()), "measurement": float(consistency.detach()),
                     "temporal": float(temporal.detach()), "od": float(od_consistency.detach())})
    after_vm, _, _, _ = eval_batch(model, val_batch, op, True)
    accepted = after_vm["ssim"] >= before_vm["ssim"] - args.uda_val_tolerance
    if not accepted:
        model.load_state_dict(before_state)
        after_vm = before_vm
    save_json(out / "obj12_uda.json", {"accepted": accepted, "before_validation": before_vm,
              "after_validation": after_vm, "curve": rows,
              "note": "obj12 used without labels; validation only gates rollback"})
    return model, accepted


def tto_one(model, sample_batch, op, steps: int, args):
    saved = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if name.startswith("tcm") or name in {"backprojection_gain", "backprojection_bias", "rho_logits"}:
            p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    initial = [p.detach().clone() for p in params]
    opt = torch.optim.Adam(params, lr=args.tto_lr)
    raw, dark, _, cond, _ = sample_batch
    obs = raw.sub(dark).mean(dim=1)
    obs = (obs - obs.mean(dim=-1, keepdim=True)) / obs.std(dim=-1, keepdim=True).clamp_min(1e-4) * 0.35
    curve = []
    for step in range(steps):
        pred, y = model(raw, dark, cond, (64, 64))
        data = (op.forward(pred.flatten(1)) - obs).abs().mean()
        tv = (pred[:, :, 1:] - pred[:, :, :-1]).abs().mean() + (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean()
        reg = sum((p - p0).square().mean() for p, p0 in zip(params, initial))
        loss = data + args.tto_tv_weight * tv + args.tto_reg_weight * reg
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            curve.append({"step": step, "loss": float(loss.detach()), "measurement": float(data.detach()), "tv": float(tv.detach())})
    with torch.no_grad():
        pred, y = model(raw, dark, cond, (64, 64))
    result = {"pred": pred.detach().cpu(), "curve": curve, "measurement_l1": float((op.forward(pred.flatten(1)) - obs).abs().mean())}
    model.load_state_dict(saved)
    return result


def run_fold(args, fold_id: int, arch_name: str = "baseline", extra_name: str | None = None):
    seed = args.seed + fold_id * 101
    seed_everything(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    patterns, samples = load_sample(args.data_root, args.patterns, 64, args.preprocess, "multi", "lanczos", args.detrend_width, args.gauss_sigma)
    if args.preserve_od_amplitude:
        samples = preserve_multi_od_amplitude(
            samples, args.preprocess, args.detrend_width, args.gauss_sigma,
        )
    objects = objects_from_samples(samples)
    test_objects = list(args.test_objects)
    folds = split_folds(objects, args.split_seed, test_objects, args.num_folds)
    if fold_id < 0 or fold_id >= args.num_folds:
        raise ValueError(f"fold must be 0..{args.num_folds - 1}, got {fold_id}")
    val_objects = folds[fold_id]
    train_objects = [x for x in objects if x not in set(test_objects) and x not in val_objects]
    out = Path(args.out_dir) / (extra_name or f"fold{fold_id}_{arch_name}")
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "run_config.json", vars(args))
    save_json(out / "split.json", {"fold": fold_id, "num_folds": args.num_folds,
              "train_objects": train_objects, "validation_objects": val_objects,
              "test_objects": test_objects,
              "all_labelled_objects": objects,
              "test_lock": "labels unused until final evaluation",
              "preserve_od_amplitude": bool(args.preserve_od_amplitude)})
    train_set = set(train_objects)
    train_samples = [sample for sample in samples if sample["object"] in train_set]
    train_batch = real_batch(samples, train_objects, device)
    val_batch = real_batch(samples, val_objects, device)
    test_batch = real_batch(samples, test_objects, device)
    obj12 = load_unlabelled(
        Path(args.data_root), args.patterns, args.preprocess, args.detrend_width,
        args.gauss_sigma, preserve_od_amplitude=args.preserve_od_amplitude,
    )
    nuisance = fit_nuisance(train_samples, patterns)
    attenuation_profile = None
    if args.fit_attenuation:
        attenuation_profile = fit_attenuation_distribution(
            args.data_root, train_objects, trim_fraction=args.attenuation_trim_fraction,
        )
        save_json(out / "attenuation_fit.json", attenuation_profile)

    noise_profile_path = args.noise_profile
    if args.fold_local_noise:
        if args.noise_profile:
            raise ValueError("--fold-local-noise cannot be combined with --noise-profile")
        noise_profile = build_noise_profile(
            args.data_root, include_sequences=True, include_objects=train_objects,
        )
        noise_profile_path = str(out / "noise_profile.json")
        save_noise_profile(noise_profile, noise_profile_path)
        noise_profile = load_noise_profile(noise_profile_path)
    else:
        noise_profile = load_noise_profile(args.noise_profile) if args.noise_profile else None
    if args.noise_mode != "poisson" and noise_profile is None:
        if args.noise_mode != "none":
            raise ValueError("--noise-profile or --fold-local-noise is required for empirical or hybrid noise")
    nuisance.update({"noise_mode": args.noise_mode, "noise_profile": noise_profile_path,
                     "noise_scale": args.noise_scale, "dark_noise_scale": args.dark_noise_scale})
    save_json(out / "nuisance.json", nuisance)

    expected_train = set(train_objects)
    attenuation_sources = set(attenuation_profile["source_objects"]) if attenuation_profile else set()
    noise_sources = set(noise_profile.get("source_objects", [])) if noise_profile else set()
    split_audit = {
        "status": "verified",
        "train_objects": train_objects,
        "validation_objects": val_objects,
        "test_objects": test_objects,
        "attenuation_source_objects": sorted(attenuation_sources),
        "noise_source_objects": sorted(noise_sources),
        "attenuation_exactly_train_only": attenuation_sources == expected_train if attenuation_profile else None,
        "noise_exactly_train_only": noise_sources == expected_train if args.fold_local_noise else None,
        "forbidden_overlap": sorted((attenuation_sources | noise_sources) & (set(val_objects) | set(test_objects))),
    }
    if attenuation_profile and attenuation_sources != expected_train:
        raise RuntimeError("attenuation fit did not use exactly the fold training objects")
    if args.fold_local_noise and noise_sources != expected_train:
        raise RuntimeError("noise profile did not use exactly the fold training objects")
    if split_audit["forbidden_overlap"]:
        raise RuntimeError(f"validation/test leakage in simulator: {split_audit['forbidden_overlap']}")
    save_json(out / "simulator_split_audit.json", split_audit)
    op = MeasurementOperator(
        patterns.shape[1], patterns.shape[0], patterns=torch.from_numpy(patterns), device=device,
        psf_sigma=args.psf_sigma, psf_sigma_y=args.psf_sigma_y, psf_angle=args.psf_angle,
        image_hw=(64, 64),
    )
    synthetic_dataset = None
    if args.synthetic_dataset_size:
        if attenuation_profile is None:
            raise ValueError("--synthetic-dataset-size requires --fit-attenuation")
        generated = generate_synthetic_dataset(
            train_samples, op, nuisance, attenuation_profile,
            args.synthetic_dataset_size, args.synthetic_generation_batch, device, seed,
            noise_profile=noise_profile, noise_mode=args.noise_mode,
            attenuation_jitter=args.attenuation_jitter,
        )
        synthetic_dataset, dataset_audit = save_reload_synthetic_dataset(
            generated, out / "synthetic_dataset.pt", attenuation_profile,
        )
        save_json(out / "synthetic_dataset_audit.json", dataset_audit)
        manifest = []
        for index in range(int(synthetic_dataset["count"])):
            manifest.append({
                "index": index,
                "A": float(synthetic_dataset["A"][index]),
                "k": float(synthetic_dataset["k"][index]),
                "b": float(synthetic_dataset["b"][index]),
                "od_responses": synthetic_dataset["od_responses"][index].tolist(),
                "attenuation_source_object": synthetic_dataset["attenuation_source_object"][index],
                "target_source_object": synthetic_dataset["target_source_object"][index],
            })
        save_json(out / "synthetic_manifest.json", {
            "schema": synthetic_dataset["schema"],
            "dataset_sha256": dataset_audit["sha256"],
            "count": len(manifest),
            "samples": manifest,
        })

    if arch_name == "shared8":
        stages, shared, lowpass, prior = 8, True, 3, 0.50
    elif arch_name == "lowpass5":
        stages, shared, lowpass, prior = 12, False, 5, 0.50
    else:
        stages, shared, lowpass, prior = args.stages, args.shared_prior, args.lowpass_kernel, args.prior_residual_scale
    model = make_model(op, args, stages, shared, lowpass, prior)
    if synthetic_dataset is None:
        # Retain the legacy manifest for old online-simulation invocations.
        manifest_rng = np.random.default_rng(seed)
        manifest = [{"seed": int(seed + i), "psf_sigma": float(manifest_rng.uniform(*nuisance["psf_sigma_range"])),
                     "shift_x": float(manifest_rng.uniform(-0.35, 0.35)), "shift_y": float(manifest_rng.uniform(-0.35, 0.35)),
                     "gain": float(manifest_rng.uniform(*nuisance["gain_range"])), "photon_peak": nuisance["photon_peak"],
                     "drift_std": nuisance["drift_std"], "nonlinearity": nuisance["nonlinearity"],
                     "noise_mode": args.noise_mode,
                     "operator_psf_sigma": args.psf_sigma,
                     "operator_psf_sigma_y": args.psf_sigma_y,
                     "operator_psf_angle": args.psf_angle} for i in range(args.manifest_count)]
        save_json(out / "synthetic_parameter_manifest.json", manifest)
    pretrain_log = []
    pretrain_steps = args.nas_pretrain_steps if arch_name != "baseline" and args.nas_mode else args.pretrain_steps
    pretrain_params = [p for p in model.parameters() if p.requires_grad]
    pre_opt = torch.optim.AdamW(pretrain_params, lr=args.pretrain_lr, weight_decay=args.weight_decay)
    for step in range(pretrain_steps):
        model.train()
        if synthetic_dataset is None:
            raw, dark, target, cond = synthetic_batch(
                train_samples, op, nuisance, args.synthetic_batch, device, seed + step,
                noise_profile=noise_profile, noise_mode=args.noise_mode,
                attenuation_profile=attenuation_profile,
                attenuation_jitter=args.attenuation_jitter,
            )
        else:
            raw, dark, target, cond = synthetic_dataset_batch(
                synthetic_dataset, args.synthetic_batch, device, seed + step,
            )
        pred, y = model(raw, dark, cond, (64, 64))
        loss = _weighted_loss(pred, target, y, op, args.consistency_weight,
                              ssim_weight=args.ssim_weight, tv_weight=args.tv_weight, edge_weight=args.edge_weight)
        pre_opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(pretrain_params, 1.0); pre_opt.step()
        if step % max(1, args.log_every) == 0 or step == pretrain_steps - 1:
            with torch.no_grad():
                sm = image_metrics(pred, target)
            rec = {"step": step, "loss": float(loss.detach()), **sm}
            pretrain_log.append(rec); print(json.dumps({"phase": "pretrain", **rec}), flush=True)
    save_json(out / "pretrain.json", {"steps": pretrain_steps, "curve": pretrain_log, "nuisance": nuisance})
    torch.save({"model": model.state_dict(), "seed": seed, "stage": "synthetic_pretrain"}, out / "checkpoint_pretrain.pt")
    model, best_val = fine_tune(model, train_batch, val_batch,
                                train_samples, nuisance, op, args, out, seed,
                                noise_profile=noise_profile,
                                synthetic_dataset=synthetic_dataset,
                                attenuation_profile=attenuation_profile)
    before_uda, _, _, _ = eval_batch(model, val_batch, op, True)
    seed_everything(seed + 200_000)
    model, uda_accepted = adapt_obj12(model, obj12, val_batch, op, args, out)
    val_after_uda, _, _, _ = eval_batch(model, val_batch, op, True)
    torch.save({
        "model": model.state_dict(),
        "best_validation_ssim": best_val,
        "validation_after_uda": val_after_uda,
        "seed": seed,
        "uda_seed": seed + 200_000,
        "uda_accepted": uda_accepted,
        "stage": "post_obj12_uda",
    }, out / "checkpoint_final.pt")
    test_full, test_rows, test_pred, test_target = eval_batch(model, test_batch, op, True)
    test_od02, test_rows_od02, test_pred_od02, _ = eval_batch(model, test_batch, op, False)
    save_json(out / "test_before_tto.json", {"full": test_full, "od0_od2": test_od02,
              "by_object": test_rows, "by_object_od0_od2": test_rows_od02,
              "uda_accepted": uda_accepted, "validation_before_uda": before_uda,
              "validation_after_uda": val_after_uda})
    torch.save({"pred": test_pred, "pred_od0_od2": test_pred_od02, "target": test_target,
                "objects": test_objects}, out / "test_predictions.pt")
    # Select TTO steps using validation only.  The selection itself never reads
    # a test label.
    tto_candidates = [args.tto_steps // 2, args.tto_steps]
    val_scores = []
    for steps in tto_candidates:
        preds = []
        for j, _ in enumerate(val_batch[4]):
            one = (val_batch[0][j:j + 1], val_batch[1][j:j + 1], val_batch[2][j:j + 1], val_batch[3][j:j + 1], [val_batch[4][j]])
            result = tto_one(model, one, op, steps, args)
            preds.append(result["pred"])
        vp = torch.cat(preds, dim=0)
        score = float(image_metrics(vp, val_batch[2].detach().cpu())["ssim"])
        val_scores.append({"steps": steps, "ssim": score})
    selected_steps = max(val_scores, key=lambda x: x["ssim"])["steps"]
    tto_rows = []
    tto_preds = []
    for j, name in enumerate(test_batch[4]):
        one = (test_batch[0][j:j + 1], test_batch[1][j:j + 1], test_batch[2][j:j + 1], test_batch[3][j:j + 1], [name])
        result = tto_one(model, one, op, selected_steps, args)
        tto_preds.append(result["pred"])
        p = result["pred"].to(device)
        with torch.no_grad():
            tm = image_metrics(p, test_batch[2][j:j + 1])
        tto_rows.append({"object": name, **tm, "measurement_l1": result["measurement_l1"], "curve": result["curve"]})
    tto_pred = torch.cat(tto_preds, dim=0)
    tto_mean = image_metrics(tto_pred, test_batch[2].detach().cpu())
    save_json(out / "tto.json", {"validation_selection": val_scores, "selected_steps": selected_steps,
              "test_mean": tto_mean, "test_by_object": tto_rows})
    preview = np.concatenate([test_target.numpy()[:, 0], test_pred.numpy()[:, 0], tto_pred.numpy()[:, 0]], axis=2)
    preview = preview.reshape(-1, preview.shape[-1])
    Image.fromarray(np.rint(np.clip(preview, 0, 1) * 255).astype(np.uint8), mode="L").save(out / "preview_target_pred_tto.png")
    save_json(out / "summary.json", {"fold": fold_id, "arch": arch_name, "best_validation_ssim": best_val,
              "validation_after_uda": val_after_uda, "test_before_tto": test_full,
              "test_od0_od2_before_tto": test_od02, "test_tto": tto_mean,
              "tto_selected_steps": selected_steps, "uda_accepted": uda_accepted,
              "status": "completed"})
    (out / "DONE").write_text("completed\n")
    print(json.dumps({"fold": fold_id, "arch": arch_name, "test_ssim": test_full["ssim"],
                      "test_od0_od2_ssim": test_od02["ssim"], "tto_ssim": float(tto_mean["ssim"]),
                      "uda_accepted": uda_accepted}, ensure_ascii=False), flush=True)


def aggregate_ensemble(args):
    root = Path(args.out_dir)
    dirs = sorted(root.glob("fold*_baseline")) + sorted(root.glob("extra_*"))
    dirs = [d for d in dirs if (d / "test_predictions.pt").exists()]
    if args.ensemble_size < 1:
        raise ValueError(f"ensemble-size must be positive, got {args.ensemble_size}")
    if len(dirs) < args.ensemble_size:
        raise RuntimeError(f"need {args.ensemble_size} model predictions, found {len(dirs)}: {dirs}")
    chosen = dirs[:args.ensemble_size]
    packs = [torch.load(d / "test_predictions.pt", map_location="cpu", weights_only=False) for d in chosen]
    pred = torch.stack([p["pred"] for p in packs]).mean(0)
    pred02 = torch.stack([p["pred_od0_od2"] for p in packs]).mean(0)
    target = packs[0]["target"]
    full = image_metrics(pred, target); od02 = image_metrics(pred02, target)
    rows = []
    for j, name in enumerate(packs[0]["objects"]):
        rows.append({"object": name, "full": image_metrics(pred[j:j + 1], target[j:j + 1]),
                     "od0_od2": image_metrics(pred02[j:j + 1], target[j:j + 1])})
    out = root / "ensemble"
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "ensemble.json", {"models": [str(x) for x in chosen], "size": len(chosen),
              "full": full, "od0_od2": od02, "by_object": rows,
              "note": "mean of fold/seed models; test labels used only for this final evaluation"})
    torch.save({"pred": pred, "pred_od0_od2": pred02, "target": target, "objects": packs[0]["objects"]}, out / "predictions.pt")
    preview = np.concatenate([target.numpy()[:, 0], pred.numpy()[:, 0], pred02.numpy()[:, 0]], axis=2)
    preview = preview.reshape(-1, preview.shape[-1])
    Image.fromarray(np.rint(np.clip(preview, 0, 1) * 255).astype(np.uint8), mode="L").save(out / "preview_target_ensemble_od02.png")
    (out / "DONE").write_text("completed\n")
    print(json.dumps({"ensemble_size": len(chosen), "ssim": full["ssim"], "od0_od2_ssim": od02["ssim"]}, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fold", "ensemble"], default="fold")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--num-folds", type=int, default=4)
    p.add_argument("--test-objects", nargs="+", default=TEST_OBJECTS)
    p.add_argument("--arch", choices=["baseline", "shared8", "lowpass5"], default="baseline")
    p.add_argument("--extra-name", default="")
    p.add_argument("--nas-mode", action="store_true")
    p.add_argument("--data-root", required=False, default="/sci_persistent_storage/ma_psdun_v2_20260830/data/sample/sample")
    p.add_argument("--patterns", required=False, default="/sci_persistent_storage/ma_psdun_v2_20260830/data/4096.tif")
    p.add_argument("--out-dir", required=False, default="/sci_persistent_storage/ma_psdun_v2_20260830/domain_bridge_20260902")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--preprocess", default="detrend", choices=["zscore", "robust", "difference", "detrend", "detrend_robust", "gauss_detrend", "gauss_detrend_robust"])
    p.add_argument("--detrend-width", type=int, default=31); p.add_argument("--gauss-sigma", type=float, default=40.0)
    p.add_argument("--stages", type=int, default=12); p.add_argument("--lowpass-kernel", type=int, default=3)
    p.add_argument("--prior-residual-scale", type=float, default=0.50); p.add_argument("--shared-prior", action="store_true")
    p.add_argument("--psf-sigma", type=float, default=0.0)
    p.add_argument("--psf-sigma-y", type=float, default=None)
    p.add_argument("--psf-angle", type=float, default=0.0)
    p.add_argument("--pretrain-steps", type=int, default=120); p.add_argument("--nas-pretrain-steps", type=int, default=50)
    p.add_argument("--pretrain-lr", type=float, default=1e-4); p.add_argument("--finetune-epochs", type=int, default=80)
    p.add_argument("--finetune-lr", type=float, default=3e-5); p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--synthetic-batch", type=int, default=8); p.add_argument("--synthetic-mix", type=float, default=0.5)
    p.add_argument("--noise-profile", default="", help="JSON profile produced by tools/analyze_noise.py")
    p.add_argument("--noise-mode", choices=["none", "poisson", "empirical", "hybrid"], default="poisson")
    p.add_argument("--fold-local-noise", action="store_true",
                   help="build empirical noise only from this fold's training objects")
    p.add_argument("--noise-scale", type=float, default=0.005); p.add_argument("--dark-noise-scale", type=float, default=0.005)
    p.add_argument("--fit-attenuation", action="store_true",
                   help="fit S(D)=A*exp(-kD)+b from this fold's raw training captures")
    p.add_argument("--attenuation-trim-fraction", type=float, default=0.01)
    p.add_argument("--attenuation-jitter", type=float, default=0.04)
    p.add_argument("--preserve-od-amplitude", action="store_true",
                   help="center each OD separately but apply one shared four-OD scale")
    p.add_argument("--synthetic-dataset-size", type=int, default=0,
                   help="materialize this many synthetic samples before training; zero keeps legacy online generation")
    p.add_argument("--synthetic-generation-batch", type=int, default=32)
    p.add_argument("--consistency-weight", type=float, default=0.01); p.add_argument("--ssim-weight", type=float, default=0.2)
    p.add_argument("--tv-weight", type=float, default=0.0); p.add_argument("--edge-weight", type=float, default=0.05)
    p.add_argument("--uda-steps", type=int, default=60); p.add_argument("--uda-lr", type=float, default=1e-4)
    p.add_argument("--uda-input-noise", type=float, default=0.01); p.add_argument("--uda-temporal-weight", type=float, default=0.1)
    p.add_argument("--uda-od-weight", type=float, default=0.1); p.add_argument("--uda-val-tolerance", type=float, default=0.01)
    p.add_argument("--tto-steps", type=int, default=100); p.add_argument("--tto-lr", type=float, default=3e-4)
    p.add_argument("--tto-tv-weight", type=float, default=0.02); p.add_argument("--tto-reg-weight", type=float, default=0.01)
    p.add_argument("--manifest-count", type=int, default=64); p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--split-seed", type=int, default=20260902); p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ensemble-size", type=int, default=5)
    args = p.parse_args()
    if args.attenuation_jitter < 0.0:
        p.error("--attenuation-jitter must be non-negative")
    if args.synthetic_dataset_size < 0:
        p.error("--synthetic-dataset-size must be non-negative")
    if args.synthetic_generation_batch < 1:
        p.error("--synthetic-generation-batch must be positive")
    if args.mode == "ensemble":
        aggregate_ensemble(args)
    else:
        run_fold(args, args.fold, args.arch, args.extra_name or None)


if __name__ == "__main__":
    main()
