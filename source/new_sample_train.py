"""Train/evaluate MA-PSDUN on the 2026-09-01 sample export.

The export contains one object directory per target, four OD captures per
object, 4096 dark/bright pairs per capture, and a variably sized ``ground
truth.png`` (the original export also contains the misspelled ``groud
truth.png``).  This script keeps object-level splits and resizes labels to the
64x64 reconstruction grid used by the 4096-frame DMD pattern.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ma_psdun.core import MeasurementOperator
from ma_psdun.conditions import (
    OD_OPTICAL_DENSITY,
    OD_TRANSMITTANCE,
    OD_VALUES,
    condition_tensor,
)
from ma_psdun.eval import image_metrics
from ma_psdun.labels import preprocess_label
from ma_psdun.model import MAPSDUN, loss_fn


# Fixed inverse-noise weighting for the four repeated OD captures.  These
# weights are intentionally conservative: OD0 carries most signal, while the
# lower-intensity captures still contribute complementary measurements.
OD_FUSION_WEIGHTS = {"OD0": 0.4, "OD1": 0.3, "OD2": 0.2, "OD3": 0.1}


def _measurement_stats(raw: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """Extract target-free acquisition statistics before normalization.

    The reconstruction input is intentionally normalized for numerical
    stability, but these two log-scale measurements retain useful information
    about object brightness and detector noise.  They are computed only from
    the dark-subtracted buckets and never inspect the target image.
    """
    signal = np.asarray(raw, dtype=np.float64) - np.asarray(dark, dtype=np.float64)
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    return np.asarray([
        np.log10(max(abs(mean), 1e-12)),
        np.log10(max(std, 1e-12)),
    ], dtype=np.float32)


def capture_fusion_weights(signals: np.ndarray, captures: list[dict], fusion: str) -> np.ndarray:
    """Return deterministic, target-free weights for one object's OD captures.

    ``signals`` are already independently normalized capture sequences.  The
    quality variants only use agreement among these sequences, so they remain
    valid at inference time for unlabeled objects and cannot consume target
    pixels during model selection.
    """
    count = int(signals.shape[0])
    if count < 1:
        raise ValueError("at least one capture is required for fusion")
    if fusion == "od0":
        weights = np.zeros(count, dtype=np.float32)
        weights[0] = 1.0
        return weights
    if fusion == "weighted":
        weights = np.asarray(
            [OD_FUSION_WEIGHTS.get(item["od"], 1.0) for item in captures],
            dtype=np.float32,
        )
    elif fusion == "quality":
        # A leave-one-out median estimates the shared pattern response while
        # remaining robust to one low-SNR or drifting capture.
        consensus = np.median(signals, axis=0)
        residual_var = np.mean((signals - consensus[None, :]) ** 2, axis=1)
        weights = 1.0 / (residual_var + 0.25)
    elif fusion == "agreement":
        if count == 1:
            return np.ones(1, dtype=np.float32)
        corr = np.corrcoef(signals)
        score = (np.nan_to_num(corr, nan=0.0).sum(axis=1) - 1.0) / (count - 1)
        # A small floor prevents a single accidental anti-correlation from
        # deleting an otherwise useful channel completely.
        weights = np.maximum(score, 0.05)
    elif fusion == "median":
        # The caller treats this sentinel as equal weights plus a median below.
        return np.full(count, 1.0 / count, dtype=np.float32)
    else:
        weights = np.ones(count, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.ones(count, dtype=np.float32)
    return weights / weights.sum()


def read_patterns(path: str | Path) -> np.ndarray:
    tif = Image.open(path)
    out = []
    for i in range(getattr(tif, "n_frames", 1)):
        tif.seek(i)
        frame = np.asarray(tif, dtype=np.float32)
        lo, hi = float(frame.min()), float(frame.max())
        out.append((frame > (lo + hi) * 0.5).astype(np.float32).reshape(-1))
    return np.stack(out)


def _truth_file(obj: Path) -> Path | None:
    def is_truth_name(path: Path) -> bool:
        name = path.stem.lower()
        # The export contains both ``groud truth`` and the more corrupted
        # ``groud trurth``/``groud trurh`` spellings.  Keep the fallback
        # narrow so unrelated acquisition files are not treated as labels.
        return (
            "truth" in name
            or "ground" in name
            or ("groud" in name and "trur" in name)
        )

    candidates = sorted(p for p in obj.iterdir() if p.is_file() and is_truth_name(p))
    return candidates[0] if candidates else None


def _resize_label(path: Path, size: int, resample: str = "lanczos", crop: str = "center",
                  contrast: str = "percentile", percentile_low: float = 1.0,
                  percentile_high: float = 99.0, return_audit: bool = False):
    label, audit = preprocess_label(path, size, resample=resample, crop=crop,
                                    contrast=contrast, percentile_low=percentile_low,
                                    percentile_high=percentile_high)
    return (label, audit) if return_audit else label


def _moving_average(x: np.ndarray, width: int = 127) -> np.ndarray:
    """Reflect-padded low-pass estimate for acquisition drift removal."""
    if width % 2 == 0:
        raise ValueError("moving-average width must be odd")
    pad = width // 2
    kernel = np.full(width, 1.0 / width, dtype=np.float32)
    return np.convolve(np.pad(x, (pad, pad), mode="reflect"), kernel, mode="valid")


def _gaussian_smooth(x: np.ndarray, sigma: float = 40.0) -> np.ndarray:
    """Reflect-padded Gaussian low-pass without a scipy runtime dependency."""
    radius = max(3, int(4.0 * sigma))
    grid = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (grid / float(sigma)) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(x, (radius, radius), mode="reflect"), kernel, mode="valid")


def _normalize(raw: np.ndarray, dark: np.ndarray, mode: str,
               detrend_width: int = 127, gauss_sigma: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    signal = raw - dark
    if mode == "zscore":
        center = float(signal.mean())
        scale = max(float(signal.std()), 1e-8)
        return (raw - center) / scale, (dark - center) / scale
    if mode == "robust":
        center = float(np.median(signal))
        mad = float(np.median(np.abs(signal - center))) * 1.4826
        scale = max(mad, float(signal.std()) * 0.25, 1e-8)
        return (raw - center) / scale, (dark - center) / scale
    if mode == "difference":
        center = float(signal.mean())
        scale = max(float(signal.std()), 1e-8)
        y = (signal - center) / scale
        return y, np.zeros_like(y)
    if mode in {"detrend", "detrend_robust", "gauss_detrend", "gauss_detrend_robust"}:
        robust = mode.endswith("robust")
        if mode.startswith("gauss"):
            smoother = lambda z: _gaussian_smooth(z, sigma=gauss_sigma)
        else:
            smoother = lambda z: _moving_average(z, width=detrend_width)
        y = signal - smoother(signal)
        center = float(np.median(y) if robust else y.mean())
        if robust:
            mad = float(np.median(np.abs(y - center))) * 1.4826
            scale = max(mad, float(y.std()) * 0.25, 1e-8)
        else:
            scale = max(float(y.std()), 1e-8)
        y = (y - center) / scale
        return y, np.zeros_like(y)
    raise ValueError(f"unknown preprocess mode: {mode}")


def load_sample(
    root: str | Path,
    pattern_path: str | Path,
    size: int = 64,
    preprocess: str = "zscore",
    fusion: str = "none",
    label_resample: str = "lanczos",
    detrend_width: int = 127,
    gauss_sigma: float = 40.0,
    label_crop: str = "center",
    label_contrast: str = "percentile",
    label_percentile_low: float = 1.0,
    label_percentile_high: float = 99.0,
):
    root = Path(root)
    pattern = read_patterns(pattern_path)
    objects = []
    for obj in sorted(root.iterdir()):
        if not obj.is_dir() or not any(obj.glob("OD*")):
            continue
        truth = _truth_file(obj)
        if truth is None:
            continue
        label, label_audit = _resize_label(
            truth, size, label_resample, label_crop, label_contrast,
            label_percentile_low, label_percentile_high, return_audit=True,
        )
        for od_dir in sorted(obj.glob("OD*")):
            data_path = od_dir / "traindata.txt"
            if not data_path.exists():
                continue
            values = np.loadtxt(data_path, dtype=np.float32)
            if values.size != pattern.shape[0] * 2:
                raise ValueError(f"{data_path}: {values.size} values, expected {pattern.shape[0] * 2}")
            dark, raw = values[0::2], values[1::2]
            stats = _measurement_stats(raw, dark)
            raw_n, dark_n = _normalize(raw, dark, preprocess, detrend_width, gauss_sigma)
            objects.append({
                "object": obj.name,
                "od": od_dir.name,
                "raw": raw_n.astype(np.float32),
                "dark": dark_n.astype(np.float32),
                "stats": stats,
                "label": label.astype(np.float32),
                "source_truth": str(truth),
                "source_data": str(data_path),
                "label_audit": label_audit,
            })
    if not objects:
        raise RuntimeError(f"no labelled OD captures found under {root}")
    if fusion != "none":
        fused = []
        for name in sorted({s["object"] for s in objects}):
            captures = [s for s in objects if s["object"] == name]
            signals = np.stack([s["raw"] - s["dark"] for s in captures])
            weights = capture_fusion_weights(signals, captures, fusion)
            stats = np.mean(np.stack([s.get("stats", _measurement_stats(s["raw"], s["dark"])) for s in captures]), axis=0)
            if fusion == "median":
                baseline = np.median(signals, axis=0)
            else:
                baseline = np.average(signals, axis=0, weights=weights)
            baseline = (baseline - baseline.mean()) / max(float(baseline.std()), 1e-8)
            fused.append({
                "object": name,
                "od": "FUSED",
                "raw": baseline.astype(np.float32),
                "dark": np.zeros_like(baseline, dtype=np.float32),
                "stats": stats.astype(np.float32),
                "label": captures[0]["label"],
                "source_truth": captures[0]["source_truth"],
                "source_data": [s["source_data"] for s in captures],
                "label_audit": captures[0]["label_audit"],
            })
        if fusion in {"mean", "weighted", "quality", "agreement", "median", "od0"}:
            objects = fused
        elif fusion == "multi":
            grouped = []
            for name in sorted({s["object"] for s in objects}):
                captures = [s for s in objects if s["object"] == name]
                grouped.append({
                    "object": name,
                    "od": "MULTI",
                    "raw": np.stack([s["raw"] for s in captures]).astype(np.float32),
                    "dark": np.stack([s["dark"] for s in captures]).astype(np.float32),
                    "stats": np.stack([s.get("stats", _measurement_stats(s["raw"], s["dark"])) for s in captures]).astype(np.float32),
                    "label": captures[0]["label"],
                    "source_truth": captures[0]["source_truth"],
                    "source_data": [s["source_data"] for s in captures],
                    "label_audit": captures[0]["label_audit"],
                })
            objects = grouped
        else:
            objects = objects + fused
    return pattern, objects


def dihedral(x: torch.Tensor, transform: int) -> torch.Tensor:
    # 0..3 rotations; 4..7 horizontal flip followed by a rotation.
    if transform >= 4:
        x = x.flip(-1)
        transform -= 4
    return torch.rot90(x, transform, (-2, -1))


def _condition(indices, samples, device, mode: str = "od") -> torch.Tensor:
    if mode == "constant":
        intensity = torch.ones(len(indices), device=device)
    elif mode in {"stats", "stats_weak"}:
        # ConditionEncoder accepts the historical (rate, intensity,
        # wavelength) triple.  Keep the physical rate fixed and encode the
        # target-free log signal mean/std in the two remaining slots.
        rows = []
        for i in indices:
            stats = np.asarray(samples[i].get("stats", _measurement_stats(samples[i]["raw"], samples[i]["dark"])), dtype=np.float32)
            if stats.ndim > 1:
                stats = stats.mean(axis=0)
            rows.append(stats)
        stats = torch.from_numpy(np.stack(rows)).to(device=device, dtype=torch.float32)
        if mode == "stats":
            # Keep the feature in the valid positive intensity domain while
            # avoiding hard saturation for the high-range legacy captures.
            intensity = torch.sigmoid(stats[:, 0] + 4.0).clamp(1e-4, 1.0 - 1e-4)
            wavelength = 1050.0 + 650.0 * torch.tanh(stats[:, 1] + 5.0)
        else:
            # A weak gate keeps the trained constant-condition prior nearly
            # unchanged while allowing a small, target-free domain cue.
            intensity = (0.98 + 0.015 * torch.tanh((stats[:, 0] + 4.0) / 2.0)).clamp(1e-4, 1.0 - 1e-4)
            wavelength = 550.0 + 80.0 * torch.tanh(stats[:, 1] + 5.0)
        return torch.stack([torch.ones_like(intensity), intensity, wavelength], dim=-1)
    elif mode == "measured":
        # The absolute ADC units vary by capture; OD is the stable condition
        # available in the export and is therefore used as the condition.
        labels = [samples[i]["od"] for i in indices]
        return condition_tensor(labels, device=device)
    else:
        labels = [samples[i]["od"] for i in indices]
        return condition_tensor(labels, device=device)
    rate = torch.ones_like(intensity)
    return torch.stack([rate, intensity, torch.full_like(intensity, 550.0)], dim=-1)


def _batch(samples, indices, device, transform: int, condition_mode: str):
    raw_np = np.stack([samples[i]["raw"] for i in indices])
    dark_np = np.stack([samples[i]["dark"] for i in indices])
    raw = torch.from_numpy(raw_np).to(device)
    dark = torch.from_numpy(dark_np).to(device)
    if raw.ndim == 2:
        raw, dark = raw[:, None], dark[:, None]
    target = torch.from_numpy(np.stack([samples[i]["label"] for i in indices])).to(device)[:, None]
    target = dihedral(target, transform)
    cond = _condition(indices, samples, device, condition_mode)
    return raw, dark, target, cond


def _drop_od_channels(raw: torch.Tensor, dark: torch.Tensor, probability: float):
    """Randomly hide lower-quality OD channels during training."""
    if probability <= 0.0 or raw.ndim != 3 or raw.shape[1] <= 1:
        return raw, dark
    keep = (torch.rand((raw.shape[0], raw.shape[1], 1), device=raw.device) >= probability).to(raw.dtype)
    keep[:, 0] = 1.0
    return raw * keep, dark * keep


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.flatten(), b.flatten()
    return float(torch.corrcoef(torch.stack([x, y]))[0, 1])


def _weighted_loss(
    pred,
    target,
    y,
    op,
    consistency_weight: float,
    binary_weight: float = 0.0,
    ssim_weight: float = 0.2,
    tv_weight: float = 0.01,
    edge_weight: float = 0.0,
    foreground_weight: float = 0.0,
):
    # Normalize the pixel weights so changing foreground emphasis does not
    # silently change the overall loss scale or effective learning rate.
    pixel_weight = (1.0 + foreground_weight * target).detach()
    pixel_weight = pixel_weight / pixel_weight.mean().clamp_min(1e-6)
    image_l1 = ((pred - target).abs() * pixel_weight).mean()
    image_mse = ((pred - target).square() * pixel_weight).mean()
    mu_x = F.avg_pool2d(pred, 7, 1, 3)
    mu_y = F.avg_pool2d(target, 7, 1, 3)
    vx = F.avg_pool2d(pred * pred, 7, 1, 3) - mu_x * mu_x
    vy = F.avg_pool2d(target * target, 7, 1, 3) - mu_y * mu_y
    cov = F.avg_pool2d(pred * target, 7, 1, 3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x.square() + mu_y.square() + c1) * (vx + vy + c2))
    consistency = (op.forward(pred.flatten(1)) - y).abs().mean()
    tv = (pred[:, :, 1:] - pred[:, :, :-1]).abs().mean() + (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean()
    edge = ((pred[:, :, 1:] - pred[:, :, :-1]) - (target[:, :, 1:] - target[:, :, :-1])).abs().mean()
    edge = edge + ((pred[:, :, :, 1:] - pred[:, :, :, :-1]) - (target[:, :, :, 1:] - target[:, :, :, :-1])).abs().mean()
    binary = F.binary_cross_entropy(pred.clamp(1e-4, 1 - 1e-4), target)
    return (image_l1 + 0.5 * image_mse + ssim_weight * (1.0 - ssim.mean())
            + consistency_weight * consistency + tv_weight * tv
            + edge_weight * edge + binary_weight * binary)


def metrics(pred, target, y, op):
    out = image_metrics(pred, target)
    out.update({"corr": _corr(pred, target), "pred_std": float(pred.std()), "target_std": float(target.std()),
                "measurement_l1": float((op.forward(pred.flatten(1)) - y).abs().mean())})
    return out


def orientation_scan(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    patterns, samples = load_sample(
        args.data_root, args.patterns, args.size, args.preprocess, args.fusion,
        args.label_resample, args.detrend_width, args.gauss_sigma,
        args.label_crop, args.label_contrast,
        args.label_percentile_low, args.label_percentile_high,
    )
    op = MeasurementOperator(patterns.shape[1], patterns.shape[0], patterns=torch.from_numpy(patterns), device=device)
    # A normalized adjoint is a diagnostic only; the learned network is still
    # trained for every candidate in the multi-card sweep.
    rows = []
    with torch.no_grad():
        for transform in range(8):
            vals = []
            for s in samples:
                raw = torch.from_numpy(s["raw"])[None].to(device)
                dark = torch.from_numpy(s["dark"])[None].to(device)
                y = raw - dark
                x = op.adjoint(y).reshape(1, 1, args.size, args.size)
                x = (x - x.amin()) / (x.amax() - x.amin() + 1e-6)
                target = dihedral(torch.from_numpy(s["label"])[None, None].to(device), transform)
                vals.append({"corr": _corr(x, target), "mse": float((x-target).square().mean())})
            rows.append({"transform": transform, "corr": float(np.mean([v["corr"] for v in vals])), "mse": float(np.mean([v["mse"] for v in vals]))})
    out = Path(args.exp_dir or "experiments/new-sample-orientation")
    out.mkdir(parents=True, exist_ok=True)
    (out / "orientation_scan.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    patterns, samples = load_sample(
        args.data_root, args.patterns, args.size, args.preprocess, args.fusion,
        args.label_resample, args.detrend_width, args.gauss_sigma,
        args.label_crop, args.label_contrast,
        args.label_percentile_low, args.label_percentile_high,
    )
    all_objects = sorted({s["object"] for s in samples})
    test_objects = [x for x in args.test_objects.split(",") if x]
    val_objects = [x for x in args.val_objects.split(",") if x]
    if args.fit_all:
        # Full-fit is an explicit deployment mode for producing reconstructions
        # of every labelled object.  Its score is a training-fit diagnostic,
        # not an independent generalization estimate.
        test_objects = list(all_objects)
        val_objects = list(all_objects)
        args.train_all_nontest = True
    if not test_objects:
        test_objects = all_objects[-max(2, len(all_objects) // 5):]
    if not val_objects:
        val_objects = [x for x in all_objects if x not in test_objects][-2:]
    # For a final deployment fit, use every labelled object except the
    # explicitly held-out test objects.  Validation remains available for
    # diagnostics, but overlaps the fit and therefore must not select a
    # checkpoint in this mode.
    if args.fit_all:
        train_objects = list(all_objects)
    elif args.train_objects:
        requested = [x.strip() for x in args.train_objects.split(",") if x.strip()]
        unknown = sorted(set(requested) - set(all_objects))
        if unknown:
            raise ValueError(f"unknown --train-objects: {unknown}; available={all_objects}")
        train_objects = [x for x in requested if x not in set(test_objects)]
    else:
        train_objects = ([x for x in all_objects if x not in set(test_objects)]
                         if args.train_all_nontest else
                         [x for x in all_objects if x not in set(test_objects + val_objects)])
    groups = {"train": [i for i,s in enumerate(samples) if s["object"] in train_objects],
              "val": [i for i,s in enumerate(samples) if s["object"] in val_objects],
              "test": [i for i,s in enumerate(samples) if s["object"] in test_objects]}
    if not all(groups.values()):
        raise ValueError(f"empty split: {groups}, objects={all_objects}")
    op = MeasurementOperator(patterns.shape[1], patterns.shape[0], patterns=torch.from_numpy(patterns), device=device)
    od_channels = int(samples[0]["raw"].shape[0]) if np.asarray(samples[0]["raw"]).ndim == 2 else 1
    model = MAPSDUN(op, stages=args.stages, backprojection_gain_init=args.gain_init, lowpass_kernel=args.lowpass_kernel, od_channels=od_channels, prior_residual_scale=args.prior_residual_scale, shared_prior=args.shared_prior, adaptive_fusion_scale=args.adaptive_fusion_scale).to(device)
    if args.od_weights:
        if od_channels == 1:
            raise ValueError("--od-weights requires --fusion multi")
        vals = [float(x) for x in args.od_weights.split(",") if x.strip()]
        if len(vals) != od_channels or any(x < 0 for x in vals) or sum(vals) <= 0:
            raise ValueError(f"--od-weights must contain {od_channels} non-negative values")
        weights = torch.tensor(vals, device=device, dtype=model.tcm.od_logits.dtype)
        model.tcm.od_logits.data.copy_(torch.log(weights.clamp_min(1e-8)))
        model.tcm.od_logits.requires_grad_(False)
    if args.od_init_weights:
        if od_channels == 1:
            raise ValueError("--od-init-weights requires --fusion multi")
        vals = [float(x) for x in args.od_init_weights.split(",") if x.strip()]
        if len(vals) != od_channels or any(x < 0 for x in vals) or sum(vals) <= 0:
            raise ValueError(f"--od-init-weights must contain {od_channels} non-negative values")
        weights = torch.tensor(vals, device=device, dtype=model.tcm.od_logits.dtype)
        model.tcm.od_logits.data.copy_(torch.log(weights.clamp_min(1e-8)))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out = Path(args.exp_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy(); cfg.update({"objects": all_objects, "train_objects": train_objects, "val_objects": val_objects, "test_objects": test_objects,
                                         "split_sizes": {k: len(v) for k,v in groups.items()}, "device_resolved": str(device),
                                         "pattern_sha256": hashlib.sha256(Path(args.patterns).read_bytes()).hexdigest()})
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out / "split.json").write_text(json.dumps({k: [samples[i]["object"] + "/" + samples[i]["od"] for i in v] for k,v in groups.items()}, indent=2) + "\n")
    audits = {sample["object"]: sample.get("label_audit") for sample in samples}
    (out / "label_preprocessing_audit.json").write_text(
        json.dumps(audits, indent=2, ensure_ascii=False) + "\n"
    )
    train_indices = groups["train"]
    # The export mixes old bar captures and the newer obj captures.  Repeating
    # new-domain objects changes only their loss weight while preserving the
    # object-level split and the physical measurement for each sample.
    if args.repeat_new > 1:
        train_indices = [i for i in train_indices
                         for _ in range(args.repeat_new if samples[i]["object"].startswith("obj") else 1)]
    train_batch = _batch(samples, train_indices, device, args.transform, args.condition_mode)
    val_batch = _batch(samples, groups["val"], device, args.transform, args.condition_mode)
    best_ssim = -float("inf"); best_mse = float("inf"); log=[]; t0=time.time()
    for epoch in range(args.epochs):
        model.train(); train_raw, train_dark = _drop_od_channels(train_batch[0], train_batch[1], args.od_dropout); pred,y=model(train_raw,train_dark,train_batch[3],(args.size,args.size)); loss=_weighted_loss(pred,train_batch[2],y,op,args.consistency_weight,args.binary_weight,args.ssim_weight,args.tv_weight,args.edge_weight,args.foreground_weight)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval()
        with torch.no_grad():
            vp,vy=model(val_batch[0],val_batch[1],val_batch[3],(args.size,args.size)); vm=metrics(vp,val_batch[2],vy,op)
        rec={**vm,"epoch":epoch,"train_loss":float(loss.detach()),"elapsed_s":time.time()-t0}; log.append(rec)
        if epoch % args.log_every == 0 or epoch == args.epochs-1: print(json.dumps(rec),flush=True)
        if vm["ssim"] > best_ssim:
            best_ssim=vm["ssim"]; torch.save({"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict(),"validation":vm},out/"checkpoint_best.pt")
        if vm["mse"] < best_mse:
            best_mse=vm["mse"]; torch.save({"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict(),"validation":vm},out/"checkpoint_best_mse.pt")
        if epoch % args.save_every == 0 or epoch == args.epochs-1: torch.save({"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict()},out/"checkpoint_latest.pt")
    (out/"validation.jsonl").write_text("\n".join(json.dumps(x) for x in log)+"\n")
    ck_path = out / ("checkpoint_latest.pt" if (args.train_all_nontest or args.fit_all) else "checkpoint_best.pt")
    ck=torch.load(ck_path,map_location=device,weights_only=False); model.load_state_dict(ck["model"]); model.eval()
    test_batch=_batch(samples,groups["test"],device,args.transform,args.condition_mode)
    with torch.no_grad(): tp,ty=model(test_batch[0],test_batch[1],test_batch[3],(args.size,args.size)); tm=metrics(tp,test_batch[2],ty,op)
    tm.update({"checkpoint_epoch":int(ck["epoch"]),"checkpoint_selection":"final_epoch" if (args.train_all_nontest or args.fit_all) else "best_validation_ssim","status":"completed","test_objects":test_objects,"train_objects":train_objects,"evaluation_note":"full-fit training diagnostic" if args.fit_all else "held-out object evaluation"})
    (out/"test.json").write_text(json.dumps(tm,indent=2)+"\n"); torch.save({"pred":tp.cpu(),"target":test_batch[2].cpu()},out/"test_samples.pt")
    rows=[]
    for j,obj in enumerate(test_objects):
        idx=[k for k,i in enumerate(groups["test"]) if samples[i]["object"]==obj]
        if idx:
            p=tp[idx]; t=test_batch[2][idx]; rows.append({"object":obj,"metrics":metrics(p,t,ty[idx],op),"od": [samples[groups["test"][i]]["od"] for i in idx]})
    (out/"test_by_object.json").write_text(json.dumps(rows,indent=2)+"\n")
    # Rows contain target | prediction for each test capture.
    canvas=[]
    for j in range(tp.shape[0]):
        a=test_batch[2][j,0].detach().cpu().numpy(); b=tp[j,0].detach().cpu().numpy(); gap=np.zeros((args.size,2),np.float32); canvas.append(np.concatenate([a,gap,b],1))
    Image.fromarray(np.rint(np.clip(np.concatenate(canvas,0),0,1)*255).astype(np.uint8),mode="L").save(out/"preview_target_pred.png")
    (out/"DONE").write_text("completed\n")
    print(json.dumps(tm),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "orientation-scan"], default="train")
    p.add_argument("--data-root", required=True)
    p.add_argument("--patterns", required=True)
    p.add_argument("--exp-dir", default="experiments/new-sample")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--preprocess", choices=["zscore", "robust", "difference", "detrend", "detrend_robust", "gauss_detrend", "gauss_detrend_robust"], default="zscore")
    p.add_argument("--detrend-width", type=int, default=127)
    p.add_argument("--gauss-sigma", type=float, default=40.0)
    p.add_argument("--fusion", choices=["none", "mean", "weighted", "quality", "agreement", "median", "od0", "both", "multi"], default="none")
    p.add_argument("--label-resample", choices=["nearest", "bilinear", "bicubic", "lanczos"], default="lanczos")
    p.add_argument("--label-crop", choices=["none", "center", "foreground"], default="center")
    p.add_argument("--label-contrast", choices=["none", "minmax", "percentile"], default="percentile")
    p.add_argument("--label-percentile-low", type=float, default=1.0)
    p.add_argument("--label-percentile-high", type=float, default=99.0)
    p.add_argument("--transform", type=int, default=0)
    p.add_argument("--condition-mode", choices=["od", "constant", "measured"], default="od")
    p.add_argument("--stages", type=int, default=8)
    p.add_argument("--gain-init", type=float, default=4.0)
    p.add_argument("--lowpass-kernel", type=int, default=7)
    p.add_argument("--prior-residual-scale", type=float, default=0.05)
    p.add_argument("--shared-prior", action="store_true")
    p.add_argument("--adaptive-fusion-scale", type=float, default=0.0)
    p.add_argument("--fit-all", action="store_true")
    p.add_argument("--train-all-nontest", action="store_true")
    p.add_argument("--train-objects", default="")
    p.add_argument("--repeat-new", type=int, default=1)
    p.add_argument("--od-weights", default="")
    p.add_argument("--od-init-weights", default="")
    p.add_argument("--od-dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--consistency-weight", type=float, default=0.05)
    p.add_argument("--binary-weight", type=float, default=0.0)
    p.add_argument("--ssim-weight", type=float, default=0.2)
    p.add_argument("--tv-weight", type=float, default=0.01)
    p.add_argument("--edge-weight", type=float, default=0.0)
    p.add_argument("--foreground-weight", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--test-objects", default="")
    p.add_argument("--val-objects", default="")
    a = p.parse_args()
    if a.repeat_new < 1:
        raise ValueError("--repeat-new must be >= 1")
    (orientation_scan if a.mode=="orientation-scan" else train)(a)


if __name__ == "__main__": main()
