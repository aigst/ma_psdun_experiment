"""Run a trained new-sample checkpoint on labelled and unlabelled objects."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ma_psdun.core import MeasurementOperator
from ma_psdun.model import MAPSDUN
from new_sample_train import _normalize, load_sample, read_patterns


def object_captures(root: Path, name: str, preprocess: str, width: int, sigma: float, multi: bool = False):
    rows = []
    for od in sorted((root / name).glob("OD*")):
        values = np.loadtxt(od / "traindata.txt", dtype=np.float32)
        dark, raw = values[0::2], values[1::2]
        raw_n, dark_n = _normalize(raw, dark, preprocess, width, sigma)
        rows.append((raw_n.astype(np.float32), dark_n.astype(np.float32)))
    if len(rows) != 4:
        raise ValueError(f"{name}: expected four OD captures, got {len(rows)}")
    if multi:
        # Multi-OD checkpoints consume all four normalized captures.  Keep the
        # same channel layout for unlabelled objects as load_sample(..., multi).
        return np.stack([raw for raw, _ in rows]).astype(np.float32), np.stack([dark for _, dark in rows]).astype(np.float32)
    signals = np.stack([raw - dark for raw, dark in rows])
    baseline = signals.mean(axis=0)
    baseline = (baseline - baseline.mean()) / max(float(baseline.std()), 1e-8)
    return baseline.astype(np.float32), np.zeros_like(baseline, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--patterns", required=True)
    p.add_argument("--checkpoint-dirs", required=True)
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--contrast-low", type=float, default=0.0)
    p.add_argument("--contrast-high", type=float, default=1.0)
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root = Path(args.data_root)
    config = json.loads((Path(args.checkpoint_dirs.split(",")[0]) / "config.json").read_text())
    ns = Namespace(**config)
    patterns_np = read_patterns(args.patterns)
    op = MeasurementOperator(
        patterns_np.shape[1], patterns_np.shape[0], patterns=torch.from_numpy(patterns_np),
        device=device, psf_sigma=float(config.get("psf_sigma", 0.0)),
        psf_sigma_y=config.get("psf_sigma_y"), psf_angle=float(config.get("psf_angle", 0.0)),
        image_hw=(int(config.get("size", 64)), int(config.get("size", 64))),
    )
    _, labelled = load_sample(args.data_root, args.patterns, ns.size, ns.preprocess, ns.fusion, ns.label_resample, ns.detrend_width, ns.gauss_sigma)
    labelled_by_name = {x["object"]: x for x in labelled}
    objects = sorted([d.name for d in root.iterdir() if d.is_dir() and any(d.glob("OD*"))])
    raw_rows, dark_rows = [], []
    labels = []
    has_label = []
    for name in objects:
        if name in labelled_by_name:
            item = labelled_by_name[name]
            raw, dark = np.asarray(item["raw"], np.float32), np.asarray(item["dark"], np.float32)
            labels.append(item["label"])
            has_label.append(True)
        else:
            raw, dark = object_captures(
                root,
                name,
                ns.preprocess,
                ns.detrend_width,
                ns.gauss_sigma,
                multi=ns.fusion == "multi",
            )
            labels.append(None)
            has_label.append(False)
        raw_rows.append(raw); dark_rows.append(dark)
    raw = torch.from_numpy(np.stack(raw_rows)).to(device)
    dark = torch.from_numpy(np.stack(dark_rows)).to(device)
    if raw.ndim == 2:
        raw, dark = raw[:, None], dark[:, None]
    cond = torch.tensor([[1.0, 1.0, 550.0]] * len(objects), device=device)
    outputs = []
    for directory in args.checkpoint_dirs.split(","):
        d = Path(directory)
        checkpoint_path = d / "checkpoint_latest.pt"
        if not checkpoint_path.exists():
            checkpoint_path = d / "checkpoint_final.pt"
        ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
        od_channels = int(config.get("od_channels", 4 if config.get("fusion") == "multi" else 1))
        model = MAPSDUN(op, stages=ns.stages, backprojection_gain_init=ns.gain_init,
                         lowpass_kernel=ns.lowpass_kernel,
                         od_channels=od_channels,
                         prior_residual_scale=ns.prior_residual_scale,
                         shared_prior=ns.shared_prior,
                         adaptive_fusion_scale=ns.adaptive_fusion_scale).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        with torch.no_grad():
            pred, _ = model(raw, dark, cond, (ns.size, ns.size))
        outputs.append(pred.cpu())
        del model
    pred = torch.stack(outputs).mean(0)
    if not (0.0 <= args.contrast_low < args.contrast_high):
        raise ValueError("contrast bounds must satisfy 0 <= low < high")
    pred = ((pred - args.contrast_low) / (args.contrast_high - args.contrast_low)).clamp(0, 1)
    out = Path(args.exp_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"pred": pred, "objects": objects, "has_label": has_label,
                "labels": labels, "member_dirs": args.checkpoint_dirs.split(",")}, out / "predictions_all.pt")
    # A compact row-per-object preview; labelled rows show target | prediction,
    # while unlabelled rows show a blank target panel | prediction.
    size = ns.size; scale = 4; label_width = 168; tile = size * scale; gap = 8; header = 44
    canvas = Image.new("L", (label_width + 2 * tile + gap, header + len(objects) * (tile + gap) - gap), 0)
    draw = ImageDraw.Draw(canvas); font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    draw.text((label_width + 4, 12), "target/blank", fill=255, font=font)
    draw.text((label_width + tile + gap + 4, 12), "prediction", fill=255, font=font)
    for i, name in enumerate(objects):
        y = header + i * (tile + gap)
        draw.text((4, y + tile // 2 - 8), name, fill=255, font=font)
        if labels[i] is not None:
            target = Image.fromarray(np.rint(np.clip(labels[i], 0, 1) * 255).astype(np.uint8)).resize((tile, tile), Image.Resampling.NEAREST)
        else:
            target = Image.new("L", (tile, tile), 0)
        output = Image.fromarray(np.rint(np.clip(pred[i, 0].numpy(), 0, 1) * 255).astype(np.uint8)).resize((tile, tile), Image.Resampling.NEAREST)
        canvas.paste(target, (label_width, y)); canvas.paste(output, (label_width + tile + gap, y))
    canvas.save(out / "preview_all_objects.png")
    summary = {"status": "completed", "objects": objects,
               "labelled_objects": [n for n, ok in zip(objects, has_label) if ok],
               "unlabelled_objects": [n for n, ok in zip(objects, has_label) if not ok],
               "member_count": len(outputs), "checkpoint_epoch": int(ck["epoch"]),
               "contrast": {"low": args.contrast_low, "high": args.contrast_high},
               "preview": "preview_all_objects.png"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
