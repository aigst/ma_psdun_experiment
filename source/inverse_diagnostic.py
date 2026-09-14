"""Target-free-selection diagnostic for a direct inverse of the real pattern.

The script solves a regularized normal equation for the centered 0/1 pattern,
then selects fusion, ridge, spatial smoothing and an affine output calibration
on one object-disjoint validation fold.  The locked test objects are only
reported after selection; this is a diagnostic/control, not a replacement for
the audited MA-PSDUN model unless it wins under the same protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ma_psdun.core import MeasurementOperator
from ma_psdun.eval import image_metrics
from new_sample_train import load_sample


TEST = ["obj18-20260901", "obj19-20260901", "obj20-20260901"]
AUDIT = ["obj15-20260901", "obj16-20260901"]
VAL = ["bar1-20260830", "bar5-20260830", "obj11-20260901", "obj24-20260903"]


def fuse(y: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "od0":
        return y[:, 0]
    if mode == "mean":
        return y.mean(1)
    if mode == "weighted":
        w = y.new_tensor([0.4, 0.3, 0.2, 0.1]).view(1, 4, 1)
        return (y * w).sum(1)
    if mode == "agreement":
        ref = y[:, :1]
        centered = y - y.mean(-1, keepdim=True)
        score = (centered * (ref - ref.mean(-1, keepdim=True))).mean(-1).clamp_min(0)
        w = score / score.sum(1, keepdim=True).clamp_min(1e-6)
        return (y * w[:, :, None]).sum(1)
    raise ValueError(mode)


def spatial(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "raw":
        return x
    if mode == "lowpass3":
        return F.avg_pool2d(x, 3, 1, 1)
    if mode == "lowpass5":
        return F.avg_pool2d(x, 5, 1, 2)
    if mode == "unsharp":
        blur = F.avg_pool2d(x, 3, 1, 1)
        return x + 0.5 * (x - blur)
    raise ValueError(mode)


def fit_affine(feature: torch.Tensor, target: torch.Tensor, train: torch.Tensor,
               val: torch.Tensor) -> tuple[torch.Tensor, dict]:
    xf, yf = feature[train].flatten(), target[train].flatten()
    xm, ym = xf.mean(), yf.mean()
    slope = ((xf - xm) * (yf - ym)).mean() / (xf.var(unbiased=False) + 1e-8)
    intercept = ym - slope * xm
    best = (-float("inf"), float(slope), float(intercept))
    # Output calibration is deliberately selected on validation only.
    for scale in torch.linspace(0.7, 1.3, 13, device=feature.device):
        for shift in torch.linspace(-0.12, 0.12, 17, device=feature.device):
            s, b = slope * scale, intercept + shift
            pred = (feature[val] * s + b).clamp(0, 1)
            score = image_metrics(pred, target[val])["ssim"]
            if score > best[0]:
                best = (score, float(s), float(b))
    return (feature * best[1] + best[2]).clamp(0, 1), {
        "validation_ssim": best[0], "slope": best[1], "intercept": best[2],
    }


def score(pred: torch.Tensor, target: torch.Tensor) -> dict:
    out = image_metrics(pred, target)
    out["corr"] = float(torch.corrcoef(torch.stack([pred.flatten(), target.flatten()]))[0, 1])
    out["pred_std"] = float(pred.std())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--patterns", required=True)
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ridges", default="0,0.001,0.003,0.01,0.03,0.1,0.3,1")
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    patterns, samples = load_sample(args.data_root, args.patterns, 64, "detrend", "multi", "bilinear", 31, 40.0)
    names = [s["object"] for s in samples]
    objects = sorted(set(names))
    train_names = [x for x in objects if x not in set(TEST + AUDIT + VAL)]
    groups = {k: torch.tensor([i for i, n in enumerate(names) if n in set(v)], device=device)
              for k, v in {"train": train_names, "val": VAL, "test": TEST, "audit": AUDIT}.items()}
    op = MeasurementOperator(patterns.shape[1], patterns.shape[0], patterns=torch.from_numpy(patterns), device=device)
    A = op.centered_patterns * op.scale
    raw = torch.from_numpy(np.stack([s["raw"] for s in samples])).to(device)
    dark = torch.from_numpy(np.stack([s["dark"] for s in samples])).to(device)
    y = (raw - dark)
    target = torch.from_numpy(np.stack([s["label"] for s in samples])).to(device)[:, None]
    rhs = torch.stack([fuse(y, mode) @ A for mode in ("od0", "mean", "weighted", "agreement")])
    results, best = [], None
    eye = torch.eye(A.shape[1], device=device, dtype=A.dtype)
    for ridge in [float(x) for x in args.ridges.split(",") if x.strip()]:
        normal = A.T @ A + ridge * eye
        for fi, mode in enumerate(("od0", "mean", "weighted", "agreement")):
            solution = torch.linalg.solve(normal, rhs[fi].T).T.reshape(-1, 1, 64, 64)
            for sm in ("raw", "lowpass3", "lowpass5", "unsharp"):
                feature = spatial(solution, sm)
                calibrated, cal = fit_affine(feature, target, groups["train"], groups["val"])
                row = {"ridge": ridge, "fusion": mode, "spatial": sm,
                       "calibration": cal,
                       "validation": score(calibrated[groups["val"]], target[groups["val"]]),
                       "test": score(calibrated[groups["test"]], target[groups["test"]]),
                       "audit": score(calibrated[groups["audit"]], target[groups["audit"]])}
                results.append(row)
                if best is None or row["validation"]["ssim"] > best["validation"]["ssim"]:
                    best = row
    out = Path(args.exp_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"status": "completed", "groups": {k: v.tolist() for k, v in groups.items()}, "best": best, "rows": results}, indent=2) + "\n")
    print(json.dumps({"best": best, "test": best["test"], "audit": best["audit"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
