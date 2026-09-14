#!/usr/bin/env python3
"""Full labelled-object fit used only for deployment image generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ma_psdun.core import MeasurementOperator
from ma_psdun.model import MAPSDUN
from new_sample_train import _batch, _weighted_loss, dihedral, load_sample
from ring_blind_train import make_model, metrics


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def train(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    patterns, samples = load_sample(
        args.data_root, args.patterns, args.size, args.preprocess, args.fusion,
        args.label_resample, args.detrend_width, args.gauss_sigma,
        args.label_crop, args.label_contrast, args.label_percentile_low,
        args.label_percentile_high,
    )
    objects = sorted({s["object"] for s in samples})
    op = MeasurementOperator(
        patterns.shape[1], patterns.shape[0], patterns=torch.from_numpy(patterns),
        device=device, psf_sigma=args.psf_sigma, psf_sigma_y=args.psf_sigma_y,
        psf_angle=args.psf_angle, image_hw=(args.size, args.size),
    )
    od_channels = int(np.asarray(samples[0]["raw"]).shape[0]) if np.asarray(samples[0]["raw"]).ndim == 2 else 1
    model = make_model(op, args, od_channels).to(device)
    if args.od_weights:
        vals = [float(x) for x in args.od_weights.split(",")]
        if len(vals) != od_channels:
            raise ValueError("od_weights length does not match channels")
        model.tcm.od_logits.data.copy_(torch.log(torch.tensor(vals, device=device).clamp_min(1e-8)))
        model.tcm.od_logits.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batch = _batch(samples, list(range(len(samples))), device, 0, args.condition_mode)
    out = Path(args.exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({
        "device_resolved": str(device), "objects": objects, "object_count": len(objects),
        "od_channels": od_channels, "pattern_sha256": sha256(Path(args.patterns)),
    })
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    history = []
    started = time.time()
    best = None
    if args.dihedral_augmentation:
        pattern_images = op.centered_patterns.reshape(op.centered_patterns.shape[0], 1, args.size, args.size)
        views = [dihedral(pattern_images, transform).flatten(1).contiguous() for transform in range(8)]
    else:
        views = [op.centered_patterns]
    base_view = views[0]
    for epoch in range(args.epochs):
        model.train()
        transform = epoch % 8 if args.dihedral_augmentation else 0
        op.centered_patterns = views[transform]
        target = dihedral(batch[2], transform)
        pred, y = model(batch[0], batch[1], batch[3], (args.size, args.size))
        loss = _weighted_loss(
            pred, target, y, op, args.consistency_weight,
            binary_weight=args.binary_weight, ssim_weight=args.ssim_weight,
            tv_weight=args.tv_weight, edge_weight=args.edge_weight,
            foreground_weight=args.foreground_weight,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        op.centered_patterns = base_view
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                current, cy = model(batch[0], batch[1], batch[3], (args.size, args.size))
                row = metrics(current, batch[2], cy, op)
            row.update({"epoch": epoch, "train_loss": float(loss.detach()), "elapsed_s": time.time() - started})
            history.append(row)
            print(json.dumps(row), flush=True)
        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            torch.save({"epoch": epoch, "model": model.state_dict()}, out / "checkpoint_latest.pt")
    torch.save({"epoch": args.epochs - 1, "model": model.state_dict()}, out / "checkpoint_final.pt")
    (out / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out / "checkpoint_final.sha256").write_text(f"{sha256(out / 'checkpoint_final.pt')}  checkpoint_final.pt\n")
    (out / "FULL_FIT_DONE").write_text("completed\n")
    print(json.dumps({"status": "completed", "objects": len(objects), "checkpoint": str(out / "checkpoint_final.pt")}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True); p.add_argument("--patterns", required=True)
    p.add_argument("--exp-dir", required=True); p.add_argument("--device", default="cuda:0")
    p.add_argument("--size", type=int, default=64); p.add_argument("--preprocess", default="detrend")
    p.add_argument("--detrend-width", type=int, default=31); p.add_argument("--gauss-sigma", type=float, default=40.0)
    p.add_argument("--psf-sigma", type=float, default=0.0); p.add_argument("--psf-sigma-y", type=float, default=None); p.add_argument("--psf-angle", type=float, default=0.0)
    p.add_argument("--fusion", choices=["mean", "weighted", "quality", "agreement", "median", "od0", "multi"], default="multi")
    p.add_argument("--label-resample", choices=["bilinear", "lanczos"], default="bilinear")
    p.add_argument("--label-crop", choices=["none", "center", "foreground"], default="center")
    p.add_argument("--label-contrast", choices=["none", "minmax", "percentile"], default="percentile")
    p.add_argument("--label-percentile-low", type=float, default=1.0); p.add_argument("--label-percentile-high", type=float, default=99.0)
    p.add_argument("--condition-mode", choices=["constant", "od", "stats", "stats_weak"], default="constant")
    p.add_argument("--stages", type=int, default=12); p.add_argument("--gain-init", type=float, default=4.0)
    p.add_argument("--lowpass-kernel", type=int, default=3); p.add_argument("--prior-residual-scale", type=float, default=0.5)
    p.add_argument("--shared-prior", action="store_true"); p.add_argument("--adaptive-fusion-scale", type=float, default=0.0)
    p.add_argument("--dihedral-augmentation", action="store_true")
    p.add_argument("--od-weights", default=""); p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--consistency-weight", type=float, default=0.01); p.add_argument("--binary-weight", type=float, default=0.0)
    p.add_argument("--ssim-weight", type=float, default=0.2); p.add_argument("--tv-weight", type=float, default=0.0)
    p.add_argument("--edge-weight", type=float, default=0.05); p.add_argument("--foreground-weight", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=202609091); p.add_argument("--log-every", type=int, default=50); p.add_argument("--save-every", type=int, default=100)
    train(p.parse_args())


if __name__ == "__main__":
    main()
