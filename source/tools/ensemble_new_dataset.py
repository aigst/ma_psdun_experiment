#!/usr/bin/env python3
"""Average independent MA-PSDUN deployments and score labelled objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ma_psdun.eval import image_metrics
from ma_psdun.labels import preprocess_label


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dir", nargs="+", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--size", type=int, default=64)
    args = p.parse_args()

    sources = [Path(x) for x in args.pred_dir]
    payloads = [torch.load(x / "deployment_predictions.pt", map_location="cpu", weights_only=False) for x in sources]
    records = payloads[0]["records"]
    pred_stack = torch.stack([item["pred"].float() for item in payloads])
    pred = pred_stack.mean(0).clamp(0.0, 1.0)
    pred_std = pred_stack.std(0, unbiased=False)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"pred": pred, "pred_std": pred_std, "records": records, "sources": [str(x) for x in sources]}, out / "deployment_predictions.pt")

    tile = 5
    scale = 4
    canvas = Image.new("L", (tile * args.size * scale, ((len(records) + tile - 1) // tile) * args.size * scale), 0)
    for idx, (image, record) in enumerate(zip(pred.numpy(), records)):
        png = Image.fromarray(np.rint(image * 255).astype(np.uint8), mode="L")
        png.save(out / f"{record['object_id']}.png")
        canvas.paste(png.resize((args.size * scale, args.size * scale), Image.Resampling.NEAREST), ((idx % tile) * args.size * scale, (idx // tile) * args.size * scale))
    canvas.save(out / "deployment_montage.png")
    unlabelled = [(image, record) for image, record in zip(pred.numpy(), records) if not record["labelled"]]
    ucanvas = Image.new("L", (2 * args.size * scale, ((len(unlabelled) + 1) // 2) * args.size * scale), 0)
    for idx, (image, record) in enumerate(unlabelled):
        block = Image.fromarray(np.rint(image * 255).astype(np.uint8), mode="L").resize((args.size * scale, args.size * scale), Image.Resampling.NEAREST)
        ucanvas.paste(block, ((idx % 2) * args.size * scale, (idx // 2) * args.size * scale))
    ucanvas.save(out / "unlabelled_montage.png")

    root = Path(args.data_root)
    rows = []
    for image, record in zip(pred, records):
        row = {"object_id": record["object_id"], "source_name": record["source_name"], "labelled": bool(record["labelled"])}
        row["seed_pixel_std_mean"] = float(pred_std[len(rows)].mean())
        row["seed_pixel_std_max"] = float(pred_std[len(rows)].max())
        obj = root / record["object_id"]
        truth = obj / "ground_truth.png"
        if truth.exists():
            target, _ = preprocess_label(truth, args.size, resample="bilinear", crop="center", contrast="percentile", percentile_low=1.0, percentile_high=99.0)
            target_t = torch.from_numpy(target).float()[None, None]
            row.update(image_metrics(image[None, None], target_t))
            row["corr"] = float(torch.corrcoef(torch.stack([image.flatten(), target_t.flatten()]))[0, 1])
        rows.append(row)
    (out / "deployment_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (out / "deployment_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    labelled = [x for x in rows if x["labelled"] and "ssim" in x]
    summary = {"objects": len(rows), "unlabelled": [x["object_id"] for x in rows if not x["labelled"]], "labelled_mean_ssim_delivery_only": float(np.mean([x["ssim"] for x in labelled])) if labelled else None}
    (out / "deployment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
