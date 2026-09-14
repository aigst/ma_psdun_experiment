"""Regenerate saved domain-bridge validation predictions and contact sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from domain_bridge_train import (
    MeasurementOperator,
    eval_batch,
    load_sample,
    make_model,
    real_batch,
)


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def gray_tile(array: np.ndarray, size: int) -> Image.Image:
    values = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(values, mode="L").resize((size, size), Image.Resampling.NEAREST).convert("RGB")


def error_tile(target: np.ndarray, pred: np.ndarray, size: int) -> Image.Image:
    error = np.clip(np.abs(target - pred) / 0.75, 0.0, 1.0)
    # A compact blue -> cyan -> yellow heat map, implemented without plotting dependencies.
    r = np.clip(2.4 * error - 0.45, 0.0, 1.0)
    g = np.clip(2.0 * error, 0.0, 1.0)
    b = np.clip(1.25 - 1.5 * error, 0.0, 1.0)
    rgb = np.rint(np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB").resize((size, size), Image.Resampling.NEAREST)


def draw_preview(path: Path, fold: int, names: list[str], target: torch.Tensor,
                 pred: torch.Tensor, metrics: dict, rows: list[dict]) -> None:
    tile = 256
    left = 238
    gap = 22
    top = 116
    row_gap = 54
    right = 26
    width = left + 3 * tile + 2 * gap + right
    height = top + len(names) * (tile + row_gap) + 10
    image = Image.new("RGB", (width, height), "#f7f8f6")
    draw = ImageDraw.Draw(image)
    heading = font(27, bold=True)
    body = font(18)
    small = font(15)
    draw.text((24, 18), f"Fold {fold} validation reconstructions", fill="#172525", font=heading)
    draw.text(
        (24, 60),
        f"5 held-out objects | mean SSIM {metrics['ssim']:.6f} | PSNR {metrics['psnr']:.3f} dB",
        fill="#53605e",
        font=body,
    )
    headers = ["Target", "Reconstruction", "Absolute error"]
    for index, label in enumerate(headers):
        x = left + index * (tile + gap)
        draw.text((x, 91), label, fill="#172525", font=small)

    target_np = target.numpy()[:, 0]
    pred_np = pred.numpy()[:, 0]
    for index, name in enumerate(names):
        y = top + index * (tile + row_gap)
        draw.rounded_rectangle((16, y, left - 20, y + tile), radius=6, fill="#e9eeeb")
        draw.text((28, y + 28), name, fill="#172525", font=body)
        draw.text((28, y + 78), f"SSIM  {rows[index]['ssim']:.6f}", fill="#146b68", font=small)
        draw.text((28, y + 108), f"PSNR  {rows[index]['psnr']:.3f} dB", fill="#53605e", font=small)
        draw.text((28, y + 138), f"MSE   {rows[index]['mse']:.6f}", fill="#53605e", font=small)
        tiles = [
            gray_tile(target_np[index], tile),
            gray_tile(pred_np[index], tile),
            error_tile(target_np[index], pred_np[index], tile),
        ]
        for col, tile_image in enumerate(tiles):
            x = left + col * (tile + gap)
            image.paste(tile_image, (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline="#cad2cd", width=2)
    image.save(path, optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=2e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cpu")
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns, samples = load_sample(
        args.data_root,
        args.patterns,
        64,
        "detrend",
        "multi",
        "lanczos",
        127,
        40.0,
    )
    operator = MeasurementOperator(
        patterns.shape[1],
        patterns.shape[0],
        patterns=torch.from_numpy(patterns),
        device=device,
        psf_sigma=0.3987402916,
        psf_sigma_y=0.2762783468,
        psf_angle=-1.292086482,
        image_hw=(64, 64),
    )
    model_args = SimpleNamespace(
        stages=12,
        shared_prior=False,
        lowpass_kernel=3,
        prior_residual_scale=0.5,
    )

    summary = []
    for fold in range(4):
        fold_dir = results_root / f"fold{fold}_baseline"
        split = json.loads((fold_dir / "split.json").read_text())
        names = split["validation_objects"]
        checkpoint = torch.load(fold_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
        model = make_model(operator, model_args)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        batch = real_batch(samples, names, device)
        aggregate, rows, pred, target = eval_batch(model, batch, operator, use_od3=True)
        recorded = float(checkpoint["best_validation_ssim"])
        delta = float(aggregate["ssim"] - recorded)
        verified = abs(delta) <= args.tolerance
        record = {
            "fold": fold,
            "objects": names,
            "aggregate": aggregate,
            "by_object": rows,
            "checkpoint_best_validation_ssim": recorded,
            "ssim_delta": delta,
            "verified_within_tolerance": verified,
            "tolerance": args.tolerance,
            "checkpoint": str(fold_dir / "checkpoint_best.pt"),
            "protocol": {
                "device": "cpu",
                "image_size": 64,
                "preprocess": "detrend",
                "detrend_width": 127,
                "gauss_sigma": 40.0,
                "od_mode": "multi",
                "label_resize": "lanczos",
                "psf_sigma_x": 0.3987402916,
                "psf_sigma_y": 0.2762783468,
                "psf_angle": -1.292086482,
            },
        }
        save_json(output_dir / f"fold{fold}_validation_metrics.json", record)
        torch.save(
            {"pred": pred, "target": target, "objects": names, "aggregate": aggregate},
            output_dir / f"fold{fold}_validation_predictions.pt",
        )
        draw_preview(
            output_dir / f"fold{fold}_validation_preview.png",
            fold,
            names,
            target,
            pred,
            aggregate,
            rows,
        )
        summary.append(record)
        print(json.dumps({"fold": fold, "ssim": aggregate["ssim"], "recorded": recorded,
                          "delta": delta, "verified": verified}), flush=True)

    save_json(output_dir / "validation_regeneration_summary.json", summary)
    if not all(item["verified_within_tolerance"] for item in summary):
        raise RuntimeError("one or more folds did not reproduce the recorded validation SSIM")


if __name__ == "__main__":
    main()
