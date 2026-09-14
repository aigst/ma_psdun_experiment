"""Classical regularized reconstruction diagnostic for the real 64x64 data.

The acquisition is full rate (4096 binary patterns for 4096 pixels), so a
regularized inverse is a useful control for the learned unrolled network.  No
test labels are used to solve or calibrate the reconstruction: ridge/smoothing
parameters are supplied on the command line and output calibration is selected
on the validation objects only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ma_psdun.core import MeasurementOperator
from ma_psdun.eval import image_metrics
from new_sample_train import load_sample


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1])


def metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    out = image_metrics(pred, target)
    out["corr"] = corr(pred, target)
    out["pred_std"] = float(pred.std())
    out["target_std"] = float(target.std())
    return out


def positive_laplacian(x: torch.Tensor, size: int) -> torch.Tensor:
    image = x.reshape(-1, 1, size, size)
    kernel = image.new_tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    return F.conv2d(image, kernel.reshape(1, 1, 3, 3), padding=1).flatten(1)


def batched_cg(
    rhs: torch.Tensor,
    matrix: torch.Tensor,
    ridge: float,
    smooth: float,
    size: int,
    iterations: int,
    tolerance: float,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Solve (A^T A + ridge I + smooth D^T D)x=A^T y for many RHS."""

    def apply(x: torch.Tensor) -> torch.Tensor:
        out = (x @ matrix.T) @ matrix
        if ridge:
            out = out + ridge * x
        if smooth:
            out = out + smooth * positive_laplacian(x, size)
        return out

    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rr = (r * r).sum(dim=1, keepdim=True)
    initial = rr.sqrt().clamp_min(1e-12)
    history = []
    for step in range(iterations):
        ap = apply(p)
        alpha = rr / (p * ap).sum(dim=1, keepdim=True).clamp_min(1e-12)
        x = x + alpha * p
        r = r - alpha * ap
        rr_next = (r * r).sum(dim=1, keepdim=True)
        relative = rr_next.sqrt() / initial
        if step % 10 == 0 or step == iterations - 1:
            history.append({
                "iteration": step,
                "relative_residual_mean": float(relative.mean()),
                "relative_residual_max": float(relative.max()),
            })
        if float(relative.max()) <= tolerance:
            break
        beta = rr_next / rr.clamp_min(1e-20)
        p = r + beta * p
        rr = rr_next
    return x, history


def fuse_measurements(raw: torch.Tensor, dark: torch.Tensor, mode: str) -> torch.Tensor:
    y = raw - dark
    if y.ndim == 2:
        return y
    if mode == "mean":
        return y.mean(dim=1)
    if mode == "od0":
        return y[:, 0]
    if mode == "weighted":
        weights = y.new_tensor([0.4, 0.3, 0.2, 0.1]).reshape(1, 4, 1)
        return (y * weights).sum(dim=1)
    if mode == "snr":
        # Detrended channels are unit variance.  Correlation between OD
        # captures estimates their shared signal without using any labels.
        centered = y - y.mean(dim=-1, keepdim=True)
        reference = centered[:, :1]
        similarity = (centered * reference).mean(dim=-1).clamp_min(0.0)
        weights = similarity / similarity.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (y * weights[..., None]).sum(dim=1)
    raise ValueError(f"unknown fusion mode: {mode}")


def normalize_feature(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "raw":
        return x
    if mode == "zscore":
        return (x - x.mean(dim=(2, 3), keepdim=True)) / x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    if mode == "robust":
        flat = x.flatten(1)
        lo = torch.quantile(flat, 0.01, dim=1, keepdim=True)
        hi = torch.quantile(flat, 0.99, dim=1, keepdim=True)
        return ((flat - lo) / (hi - lo).clamp_min(1e-6)).reshape_as(x)
    raise ValueError(f"unknown feature normalization: {mode}")


def calibrate_output(
    feature: torch.Tensor,
    target: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit affine scale on train and select a local SSIM grid on validation."""
    xf = feature[train_idx].flatten()
    yf = target[train_idx].flatten()
    x_mean, y_mean = xf.mean(), yf.mean()
    slope = ((xf - x_mean) * (yf - y_mean)).mean() / (xf.var(unbiased=False) + 1e-8)
    intercept = y_mean - slope * x_mean

    best = (-float("inf"), float(slope), float(intercept))
    slope_grid = torch.linspace(0.7, 1.3, 13, device=feature.device) * slope
    intercept_grid = torch.linspace(-0.12, 0.12, 17, device=feature.device) + intercept
    for candidate_slope in slope_grid:
        for candidate_intercept in intercept_grid:
            pred = (feature[val_idx] * candidate_slope + candidate_intercept).clamp(0.0, 1.0)
            score = image_metrics(pred, target[val_idx])["ssim"]
            if score > best[0]:
                best = (score, float(candidate_slope), float(candidate_intercept))
    output = (feature * best[1] + best[2]).clamp(0.0, 1.0)
    return output, {
        "validation_ssim": best[0],
        "slope": best[1],
        "intercept": best[2],
        "train_mse_slope": float(slope),
        "train_mse_intercept": float(intercept),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preprocess", default="detrend")
    parser.add_argument("--detrend-width", type=int, default=31)
    parser.add_argument("--fusion", choices=["mean", "weighted", "od0", "snr"], default="mean")
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--smooth", type=float, default=0.0)
    parser.add_argument("--cg-iters", type=int, default=120)
    parser.add_argument("--cg-tolerance", type=float, default=1e-5)
    parser.add_argument("--test-objects", default="obj18-20260901,obj19-20260901,obj20-20260901")
    parser.add_argument("--val-objects", default="obj15-20260901,obj16-20260901,obj17-20260901")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    patterns, samples = load_sample(
        args.data_root,
        args.patterns,
        64,
        args.preprocess,
        "multi",
        "bilinear",
        args.detrend_width,
        40.0,
    )
    objects = sorted({sample["object"] for sample in samples})
    test_objects = [value for value in args.test_objects.split(",") if value]
    val_objects = [value for value in args.val_objects.split(",") if value]
    train_objects = [value for value in objects if value not in set(test_objects + val_objects)]
    groups = {
        "train": [i for i, sample in enumerate(samples) if sample["object"] in train_objects],
        "val": [i for i, sample in enumerate(samples) if sample["object"] in val_objects],
        "test": [i for i, sample in enumerate(samples) if sample["object"] in test_objects],
    }
    if not all(groups.values()):
        raise ValueError(f"empty split: {groups}")

    operator = MeasurementOperator(
        patterns.shape[1],
        patterns.shape[0],
        patterns=torch.from_numpy(patterns),
        device=device,
    )
    raw = torch.from_numpy(np.stack([sample["raw"] for sample in samples])).to(device)
    dark = torch.from_numpy(np.stack([sample["dark"] for sample in samples])).to(device)
    target = torch.from_numpy(np.stack([sample["label"] for sample in samples])).to(device)[:, None]
    y = fuse_measurements(raw, dark, args.fusion)
    y = y - y.mean(dim=-1, keepdim=True)

    started = time.time()
    rhs = y @ operator.centered_patterns * operator.scale
    solution, cg_history = batched_cg(
        rhs,
        operator.centered_patterns * operator.scale,
        args.ridge,
        args.smooth,
        64,
        args.cg_iters,
        args.cg_tolerance,
    )
    image = solution.reshape(-1, 1, 64, 64)
    index = {key: torch.tensor(value, device=device) for key, value in groups.items()}

    candidates = {}
    for feature_mode in ("raw", "zscore", "robust"):
        feature = normalize_feature(image, feature_mode)
        pred, calibration = calibrate_output(feature, target, index["train"], index["val"])
        candidates[feature_mode] = (calibration["validation_ssim"], pred, calibration)
    selected_mode, (_, prediction, calibration) = max(candidates.items(), key=lambda item: item[1][0])

    result = {
        "status": "completed",
        "ridge": args.ridge,
        "smooth": args.smooth,
        "fusion": args.fusion,
        "feature_mode": selected_mode,
        "calibration": calibration,
        "cg_history": cg_history,
        "elapsed_s": time.time() - started,
        "train": metrics(prediction[index["train"]], target[index["train"]]),
        "validation": metrics(prediction[index["val"]], target[index["val"]]),
        "test": metrics(prediction[index["test"]], target[index["test"]]),
        "test_objects": test_objects,
        "train_objects": train_objects,
    }
    result["test_by_object"] = [
        {
            "object": object_name,
            "metrics": metrics(prediction[i : i + 1], target[i : i + 1]),
        }
        for object_name in test_objects
        for i, sample in enumerate(samples)
        if sample["object"] == object_name
    ]

    out = Path(args.exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(
        {"pred": prediction[index["test"]].cpu(), "target": target[index["test"]].cpu()},
        out / "test_samples.pt",
    )
    canvas = []
    for i in index["test"].tolist():
        truth = target[i, 0].detach().cpu().numpy()
        pred = prediction[i, 0].detach().cpu().numpy()
        canvas.append(np.concatenate([truth, np.zeros((64, 2), np.float32), pred], axis=1))
    preview = np.rint(np.clip(np.concatenate(canvas, axis=0), 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(out / "preview_target_pred.png")
    (out / "DONE").write_text("completed\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
