"""Physical linear-inverse control and OOF-selected blend with MA-PSDUN."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ma_psdun.core import MeasurementOperator
from ma_psdun.eval import image_metrics
from new_sample_train import load_sample, read_patterns


def laplacian(x: torch.Tensor, size: int) -> torch.Tensor:
    image = x.reshape(-1, 1, size, size)
    kernel = x.new_tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    return F.conv2d(image, kernel.reshape(1, 1, 3, 3), padding=1).flatten(1)


def cg(op, y: torch.Tensor, ridge: float, smooth: float, size: int, steps: int) -> torch.Tensor:
    rhs = op.adjoint(y)

    def apply(x):
        value = op.adjoint(op.forward(x)) + ridge * x
        if smooth:
            value = value + smooth * laplacian(x, size)
        return value

    x = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    rr = (residual * residual).sum(1, keepdim=True)
    for _ in range(steps):
        ad = apply(direction)
        denom = (direction * ad).sum(1, keepdim=True).clamp_min(1e-12)
        alpha = rr / denom
        x = x + alpha * direction
        residual = residual - alpha * ad
        rr_next = (residual * residual).sum(1, keepdim=True)
        beta = rr_next / rr.clamp_min(1e-20)
        direction = residual + beta * direction
        rr = rr_next
    return x.reshape(-1, 1, size, size)


def normalize(x: torch.Tensor, mode: str) -> torch.Tensor:
    flat = x.flatten(1)
    if mode == "minmax":
        lo, hi = flat.amin(1, keepdim=True), flat.amax(1, keepdim=True)
        return ((flat - lo) / (hi - lo).clamp_min(1e-6)).reshape_as(x)
    if mode == "zscore":
        z = (flat - flat.mean(1, keepdim=True)) / flat.std(1, keepdim=True).clamp_min(1e-6)
        return torch.sigmoid(z).reshape_as(x)
    if mode == "clip":
        lo = torch.quantile(flat, 0.01, dim=1, keepdim=True)
        hi = torch.quantile(flat, 0.99, dim=1, keepdim=True)
        return ((flat - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1).reshape_as(x)
    raise ValueError(mode)


def affine_fit(feature: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    xf, yf = feature.flatten(), target.flatten()
    xm, ym = xf.mean(), yf.mean()
    slope = ((xf - xm) * (yf - ym)).mean() / (xf.var(unbiased=False) + 1e-8)
    intercept = ym - slope * xm
    return (feature * slope + intercept).clamp(0, 1), float(slope), float(intercept)


def measurements(samples, objects, device):
    by_name = {s["object"]: s for s in samples}
    rows = []
    for name in objects:
        item = by_name[name]
        signal = np.asarray(item["raw"], np.float32) - np.asarray(item["dark"], np.float32)
        if signal.ndim == 2:
            signal = signal.mean(0)
        rows.append(0.35 * signal)
    return torch.from_numpy(np.stack(rows)).to(device)


def pack_metrics(pred, target, objects):
    return {"overall": image_metrics(pred, target), "by_object": [
        {"object": name, "metrics": image_metrics(pred[i:i + 1], target[i:i + 1])}
        for i, name in enumerate(objects)
    ]}


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--patterns", required=True)
    p.add_argument("--oof-pack", required=True)
    p.add_argument("--selection-pack", required=True)
    p.add_argument("--final-pack", required=True)
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--psf-sigma", type=float, default=1.0)
    p.add_argument("--preprocess", default="detrend")
    p.add_argument("--detrend-width", type=int, default=127)
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("classical PSF inverse requires CUDA")
    out = Path(args.exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    patterns = read_patterns(args.patterns)
    _, samples = load_sample(args.data_root, args.patterns, args.size, args.preprocess, "multi", "bilinear", args.detrend_width, 40.0)
    op = MeasurementOperator(patterns.shape[1], patterns.shape[0], patterns=torch.from_numpy(patterns), device=device,
                             psf_sigma=args.psf_sigma, image_hw=(args.size, args.size))
    oof = torch.load(args.oof_pack, map_location="cpu", weights_only=False)
    sel = torch.load(args.selection_pack, map_location="cpu", weights_only=False)
    final = torch.load(args.final_pack, map_location="cpu", weights_only=False)
    objects = list(oof["objects"])
    base_oof = sel["predictions"]["conv3"].float().to(device)
    target_oof = sel["target"].float().to(device)
    oof_y = measurements(samples, objects, device)
    test_objects, audit_objects = list(final["objects"]), list(final["audit_objects"])
    test_y, audit_y = measurements(samples, test_objects, device), measurements(samples, audit_objects, device)
    all_rows = []
    for ridge in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        for smooth in (0.0, 0.01, 0.1):
            print(json.dumps({"phase": "cg", "ridge": ridge, "smooth": smooth}), flush=True)
            ridge_oof = cg(op, oof_y, ridge, smooth, args.size, 80)
            for mode in ("minmax", "zscore", "clip"):
                feature = normalize(ridge_oof, mode)
                calibrated, slope, intercept = affine_fit(feature, target_oof)
                for weight in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
                    blend = ((1.0 - weight) * base_oof + weight * calibrated).clamp(0, 1)
                    all_rows.append({
                        "ridge": ridge, "smooth": smooth, "feature": mode, "weight": weight,
                        "slope": slope, "intercept": intercept, "metrics": image_metrics(blend, target_oof),
                    })
    all_rows.sort(key=lambda row: (row["metrics"]["ssim"], -row["metrics"]["mse"]), reverse=True)
    chosen = all_rows[0]
    ridge_oof = cg(op, oof_y, chosen["ridge"], chosen["smooth"], args.size, 80)
    ridge_test = cg(op, test_y, chosen["ridge"], chosen["smooth"], args.size, 80)
    ridge_audit = cg(op, audit_y, chosen["ridge"], chosen["smooth"], args.size, 80)
    feature_oof = normalize(ridge_oof, chosen["feature"])
    feature_test = normalize(ridge_test, chosen["feature"])
    feature_audit = normalize(ridge_audit, chosen["feature"])
    calibrated_test = (feature_test * chosen["slope"] + chosen["intercept"]).clamp(0, 1)
    calibrated_audit = (feature_audit * chosen["slope"] + chosen["intercept"]).clamp(0, 1)
    base_test, base_audit = final["pred"].float().to(device), final["audit_pred"].float().to(device)
    blend_test = ((1.0 - chosen["weight"]) * base_test + chosen["weight"] * calibrated_test).clamp(0, 1)
    blend_audit = ((1.0 - chosen["weight"]) * base_audit + chosen["weight"] * calibrated_audit).clamp(0, 1)
    result = {
        "status": "completed",
        "protocol": "OOF-selected physical PSF ridge feature and blend; test labels excluded from selection",
        "selected": chosen,
        "oof_top10": all_rows[:10],
        "primary_test_base": pack_metrics(base_test.cpu(), final["target"], test_objects),
        "primary_test": pack_metrics(blend_test.cpu(), final["target"], test_objects),
        "audit_base": pack_metrics(base_audit.cpu(), final["audit_target"], audit_objects),
        "audit": pack_metrics(blend_audit.cpu(), final["audit_target"], audit_objects),
        "test_used_for_selection": False,
        "oof_objects": objects,
        "sha256": {"oof_pack": sha256(Path(args.oof_pack)), "selection_pack": sha256(Path(args.selection_pack)), "final_pack": sha256(Path(args.final_pack)), "patterns": sha256(Path(args.patterns))},
    }
    (out / "classical_blend.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    torch.save({"pred": blend_test.cpu(), "target": final["target"], "objects": test_objects,
                "audit_pred": blend_audit.cpu(), "audit_target": final["audit_target"], "audit_objects": audit_objects,
                "base_pred": base_test.cpu(), "selected": chosen}, out / "predictions.pt")
    print(json.dumps({"selected": chosen, "base_test_ssim": result["primary_test_base"]["overall"]["ssim"],
                      "test_ssim": result["primary_test"]["overall"]["ssim"], "audit_ssim": result["audit"]["overall"]["ssim"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
