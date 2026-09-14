#!/usr/bin/env python3
"""Estimate a small anisotropic Gaussian PSF from labelled bar targets.

This is a calibration artifact, not a model score.  It fits the blur kernel
against measured bucket sequences while allowing one gain and offset per
capture, then writes the fitted parameters and per-object residuals for an
auditable sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ma_psdun.core import MeasurementOperator
from new_sample_train import load_sample, read_patterns


def gaussian_kernel(log_sigma_x, log_sigma_y, angle_deg, radius=8):
    device, dtype = log_sigma_x.device, log_sigma_x.dtype
    grid = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    angle = angle_deg * (math.pi / 180.0)
    rx = xx * torch.cos(angle) + yy * torch.sin(angle)
    ry = -xx * torch.sin(angle) + yy * torch.cos(angle)
    sx, sy = log_sigma_x.exp().clamp(0.15, 4.0), log_sigma_y.exp().clamp(0.15, 4.0)
    kernel = torch.exp(-0.5 * (rx.square() / sx.square() + ry.square() / sy.square()))
    return (kernel / kernel.sum()).reshape(1, 1, 2 * radius + 1, 2 * radius + 1)


def fit_psf(patterns: np.ndarray, samples: list[dict], *, device: str, steps: int, lr: float) -> dict:
    if not samples:
        raise ValueError("no calibration samples")
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    pattern_tensor = torch.from_numpy(patterns).to(dev)
    centered = pattern_tensor - pattern_tensor.mean(dim=0, keepdim=True)
    scale = 1.0 / max(patterns.shape[0], 1) ** 0.5
    target = torch.from_numpy(np.stack([s["label"] for s in samples])).to(dev)[:, None]
    observed = torch.from_numpy(np.stack([s["raw"] - s["dark"] for s in samples])).to(dev)
    log_sx = torch.tensor(math.log(1.0), device=dev, requires_grad=True)
    log_sy = torch.tensor(math.log(1.0), device=dev, requires_grad=True)
    angle = torch.tensor(0.0, device=dev, requires_grad=True)
    log_gain = torch.zeros(len(samples), device=dev, requires_grad=True)
    bias = torch.zeros(len(samples), device=dev, requires_grad=True)
    optimizer = torch.optim.Adam([log_sx, log_sy, angle, log_gain, bias], lr=lr)
    history = []
    for step in range(steps):
        kernel = gaussian_kernel(log_sx, log_sy, angle)
        blurred = F.conv2d(target, kernel, padding=kernel.shape[-1] // 2).flatten(1)
        predicted = blurred @ centered.T * scale
        calibrated = predicted * log_gain.exp()[:, None] + bias[:, None]
        fit = F.smooth_l1_loss(calibrated, observed)
        regularizer = 1e-3 * (log_gain.square().mean() + bias.square().mean())
        loss = fit + regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_sx.clamp_(math.log(0.15), math.log(4.0))
            log_sy.clamp_(math.log(0.15), math.log(4.0))
            angle.clamp_(-90.0, 90.0)
        if step == 0 or step == steps - 1 or step % max(1, steps // 10) == 0:
            history.append({
                "step": step,
                "loss": float(loss.detach()),
                "fit": float(fit.detach()),
                "sigma_x": float(log_sx.detach().exp()),
                "sigma_y": float(log_sy.detach().exp()),
                "angle_deg": float(angle.detach()),
            })
    with torch.no_grad():
        kernel = gaussian_kernel(log_sx, log_sy, angle)
        blurred = F.conv2d(target, kernel, padding=kernel.shape[-1] // 2).flatten(1)
        predicted = blurred @ centered.T * scale
        calibrated = predicted * log_gain.exp()[:, None] + bias[:, None]
        residual = calibrated - observed
    by_object = []
    for i, sample in enumerate(samples):
        by_object.append({
            "object": sample["object"],
            "od": sample["od"],
            "measurement_l1": float(residual[i].abs().mean()),
            "measurement_rmse": float(residual[i].square().mean().sqrt()),
        })
    return {
        "status": "completed",
        "method": "label_forward_model_fit_with_per_capture_affine_calibration",
        "sample_count": len(samples),
        "sigma_x": float(log_sx.detach().exp()),
        "sigma_y": float(log_sy.detach().exp()),
        "angle_deg": float(angle.detach()),
        "kernel_radius": 8,
        "history": history,
        "by_object": by_object,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--preprocess", default="detrend")
    parser.add_argument("--objects", default="")
    args = parser.parse_args()
    patterns = read_patterns(args.patterns)
    _, loaded = load_sample(args.data_root, args.patterns, args.size, args.preprocess, "mean", "bilinear")
    requested = {x.strip() for x in args.objects.split(",") if x.strip()}
    samples = [s for s in loaded if s["object"].startswith("bar") and (not requested or s["object"] in requested)]
    result = fit_psf(patterns, samples, device=args.device, steps=args.steps, lr=args.lr)
    result.update({
        "data_root": args.data_root,
        "patterns": args.patterns,
        "objects": sorted({s["object"] for s in samples}),
        "source_sha256": hashlib.sha256(Path(args.patterns).read_bytes()).hexdigest(),
        "config": vars(args),
        "usage": "pass sigma_x/sigma_y/angle_deg to MeasurementOperator for the simulator",
    })
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ("sample_count", "sigma_x", "sigma_y", "angle_deg")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
