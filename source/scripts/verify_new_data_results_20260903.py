"""Independently verify saved metrics and render the final comparison preview."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ma_psdun.eval import image_metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_pack(pack: dict) -> dict:
    return {
        "primary_test": {
            "overall": image_metrics(pack["pred"], pack["target"]),
            "by_object": {
                name: image_metrics(pack["pred"][i:i + 1], pack["target"][i:i + 1])
                for i, name in enumerate(pack["objects"])
            },
        },
        "audit_ring_holdout": {
            "overall": image_metrics(pack["audit_pred"], pack["audit_target"]),
            "by_object": {
                name: image_metrics(pack["audit_pred"][i:i + 1], pack["audit_target"][i:i + 1])
                for i, name in enumerate(pack["audit_objects"])
            },
        },
    }


def assert_metrics(actual: dict, expected: dict, tolerance: float = 1e-7) -> None:
    for group in ("primary_test", "audit_ring_holdout"):
        for key in ("mse", "psnr", "ssim"):
            difference = abs(actual[group]["overall"][key] - expected[group]["overall"][key])
            if difference > tolerance:
                raise RuntimeError(f"metric mismatch for {group}.{key}: {difference}")
        expected_rows = {
            row["object"]: row["metrics"] for row in expected[group]["by_object"]
        }
        if set(actual[group]["by_object"]) != set(expected_rows):
            raise RuntimeError(f"object mismatch in {group}")
        for name, metrics in actual[group]["by_object"].items():
            for key in ("mse", "psnr", "ssim"):
                difference = abs(metrics[key] - expected_rows[name][key])
                if difference > tolerance:
                    raise RuntimeError(f"metric mismatch for {group}.{name}.{key}: {difference}")


def render_comparison(path: Path, packs: list[tuple[str, dict]]) -> None:
    reference = packs[0][1]
    names = reference["objects"] + reference["audit_objects"]
    targets = torch.cat([reference["target"], reference["audit_target"]])
    predictions = [torch.cat([pack["pred"], pack["audit_pred"]]) for _, pack in packs]
    scale = 4
    tile = targets.shape[-1] * scale
    label_width = 96
    gap = 8
    header_height = 44
    columns = ["target"] + [label for label, _ in packs]
    canvas = Image.new(
        "L",
        (label_width + len(columns) * tile + (len(columns) - 1) * gap, header_height + len(names) * (tile + gap) - gap),
        color=0,
    )
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    header_font = ImageFont.truetype(font_path, 18)
    row_font = ImageFont.truetype(font_path, 16)
    for column, label in enumerate(columns):
        draw.text((label_width + column * (tile + gap) + 4, 12), label, fill=255, font=header_font)

    for row, name in enumerate(names):
        y = header_height + row * (tile + gap)
        draw.text((4, y + tile // 2 - 8), name.split("-")[0], fill=255, font=row_font)
        images = [targets[row, 0]] + [pred[row, 0] for pred in predictions]
        for column, tensor in enumerate(images):
            array = np.rint(tensor.numpy().clip(0, 1) * 255).astype(np.uint8)
            image = Image.fromarray(array).resize((tile, tile), Image.Resampling.NEAREST)
            canvas.paste(image, (label_width + column * (tile + gap), y))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args()
    root = Path(args.artifact_root)

    paths = {
        "old_strict": root / "reference" / "old_strict_calibrated_predictions.pt",
        "expanded_data": root / "results_cv" / "final" / "calibrated" / "predictions.pt",
        "validation_blend": root / "results_blend" / "final" / "predictions.pt",
    }
    packs = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    reference = packs["old_strict"]
    for name, pack in packs.items():
        if pack["objects"] != reference["objects"] or pack["audit_objects"] != reference["audit_objects"]:
            raise RuntimeError(f"object order differs for {name}")
        if not torch.equal(pack["target"], reference["target"]):
            raise RuntimeError(f"primary targets differ for {name}")
        if not torch.equal(pack["audit_target"], reference["audit_target"]):
            raise RuntimeError(f"audit targets differ for {name}")

    expected = {
        "old_strict": json.loads((root / "reference" / "old_strict_calibration.json").read_text()),
        "expanded_data": json.loads((root / "results_cv" / "summary.json").read_text()),
        "validation_blend": json.loads((root / "results_blend" / "summary.json").read_text()),
    }
    calculated = {name: metric_pack(pack) for name, pack in packs.items()}
    for name in packs:
        assert_metrics(calculated[name], expected[name])

    best_name = max(calculated, key=lambda name: calculated[name]["primary_test"]["overall"]["ssim"])
    preview = root / "ma_psdun_new_data_comparison_20260903.png"
    render_comparison(preview, [
        ("old strict", packs["old_strict"]),
        ("expanded", packs["expanded_data"]),
        ("blend", packs["validation_blend"]),
    ])
    report = {
        "status": "verified",
        "metric_tolerance": 1e-7,
        "targets_exactly_equal": True,
        "objects": reference["objects"],
        "audit_objects": reference["audit_objects"],
        "calculated": calculated,
        "best_strict_result": best_name,
        "old_best_retained": best_name == "old_strict",
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
        "preview": preview.name,
        "preview_sha256": sha256(preview),
    }
    (root / "independent_verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
