#!/usr/bin/env python3
"""Run a trained MA-PSDUN checkpoint over every converted object."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from ma_psdun.conditions import condition_tensor
from ma_psdun.core import MeasurementOperator
from new_sample_train import _measurement_stats, _normalize, read_patterns
from ring_blind_train import make_model


def predict(args):
    cfg = json.loads((Path(args.checkpoint).parent / "config.json").read_text())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pattern = read_patterns(args.patterns)
    op = MeasurementOperator(
        pattern.shape[1], pattern.shape[0], patterns=torch.from_numpy(pattern), device=device,
        psf_sigma=float(cfg.get("psf_sigma", 0.0)), psf_sigma_y=cfg.get("psf_sigma_y"),
        psf_angle=float(cfg.get("psf_angle", 0.0)), image_hw=(int(cfg.get("size", 64)), int(cfg.get("size", 64))),
    )
    fusion = cfg.get("fusion", "multi")
    od_channels = int(cfg.get("od_channels", 4 if fusion == "multi" else 1))
    model = make_model(op, argparse.Namespace(**cfg), od_channels).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    root = Path(args.data_root)
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    size = int(cfg.get("size", 64))
    predictions, records = [], []
    for item in manifest["objects"]:
        object_id = item["object_id"]
        raws, darks, stats = [], [], []
        for od in ("OD0", "OD1", "OD2", "OD3"):
            values = np.loadtxt(root / object_id / od / "traindata.txt", dtype=np.float32)
            dark, raw = values[0::2], values[1::2]
            stats.append(_measurement_stats(raw, dark))
            raw_n, dark_n = _normalize(raw, dark, cfg.get("preprocess", "detrend"), int(cfg.get("detrend_width", 31)), float(cfg.get("gauss_sigma", 40.0)))
            raws.append(raw_n); darks.append(dark_n)
        if fusion == "multi":
            raw_input = np.stack(raws)
            dark_input = np.stack(darks)
        else:
            baseline = np.mean(np.stack(raws) - np.stack(darks), axis=0)
            baseline = (baseline - baseline.mean()) / max(float(baseline.std()), 1e-8)
            raw_input = baseline[None]
            dark_input = np.zeros_like(raw_input)
        raw_t = torch.from_numpy(raw_input[None]).to(device)
        dark_t = torch.from_numpy(dark_input[None]).to(device)
        mode = cfg.get("condition_mode", "constant")
        if mode == "constant":
            cond = torch.tensor([[1.0, 1.0, 550.0]], device=device)
        elif mode in {"stats", "stats_weak"}:
            s = torch.from_numpy(np.mean(np.stack(stats), axis=0)[None]).to(device=device, dtype=torch.float32)
            if mode == "stats":
                intensity = torch.sigmoid(s[:, 0] + 4.0).clamp(1e-4, 1.0 - 1e-4)
                wavelength = 1050.0 + 650.0 * torch.tanh(s[:, 1] + 5.0)
            else:
                intensity = (0.98 + 0.015 * torch.tanh((s[:, 0] + 4.0) / 2.0)).clamp(1e-4, 1.0 - 1e-4)
                wavelength = 550.0 + 80.0 * torch.tanh(s[:, 1] + 5.0)
            cond = torch.stack([torch.ones_like(intensity), intensity, wavelength], dim=-1)
        else:
            cond = condition_tensor(["OD0"], device=device)
        with torch.no_grad():
            pred, _ = model(raw_t, dark_t, cond, (size, size))
        image = pred[0, 0].detach().cpu().numpy().astype(np.float32)
        predictions.append(image)
        records.append({"object_id": object_id, "source_name": item["source_name"], "labelled": item["labelled"]})

    arr = np.stack(predictions)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    torch.save({"pred": torch.from_numpy(arr), "records": records}, out / "deployment_predictions.pt")
    for image, record in zip(arr, records):
        Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8), mode="L").save(out / f"{record['object_id']}.png")
    tile = 5; scale = 4
    canvas = Image.new("L", (tile * size * scale, ((len(arr) + tile - 1) // tile) * size * scale), 0)
    for idx, image in enumerate(arr):
        block = Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8), mode="L").resize((size * scale, size * scale), Image.Resampling.NEAREST)
        canvas.paste(block, ((idx % tile) * size * scale, (idx // tile) * size * scale))
    canvas.save(out / "deployment_montage.png")
    (out / "deployment_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "completed", "objects": len(records), "unlabelled": [r["object_id"] for r in records if not r["labelled"]]}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True); p.add_argument("--patterns", required=True)
    p.add_argument("--checkpoint", required=True); p.add_argument("--output", required=True); p.add_argument("--device", default="cuda:0")
    predict(p.parse_args())


if __name__ == "__main__":
    main()
