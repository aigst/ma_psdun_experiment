"""Train a supervised output refiner for the full-fit deployment snapshot.

This is deliberately separate from the ring-blind protocol. It consumes the
current full-fit predictions and all available reference labels to improve the
images delivered for this capture set; its score is a training-fit/deployment
quality measure, not an unknown-object generalization estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from ma_psdun.eval import image_metrics
from ma_psdun.model import _ssim_loss


class ResidualRefiner(nn.Module):
    def __init__(self, channels: int = 64, depth: int = 4, scale: float = 0.3):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.scale = float(scale)
        layers: list[nn.Module] = [
            nn.Conv2d(1, channels, 3, padding=1, padding_mode="reflect"),
            nn.GELU(),
        ]
        for _ in range(depth - 1):
            layers.extend([
                nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect"),
                nn.GELU(),
            ])
        layers.append(nn.Conv2d(channels, 1, 3, padding=1, padding_mode="reflect"))
        self.body = nn.Sequential(*layers)
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.body(x)
        return (x + self.scale * torch.tanh(delta)).clamp(0.0, 1.0)


def dihedral(x: torch.Tensor, transform: int) -> torch.Tensor:
    if transform >= 4:
        x = x.flip(-1)
        transform -= 4
    return torch.rot90(x, transform, (-2, -1))


def loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    edge = F.l1_loss(
        output[:, :, 1:] - output[:, :, :-1],
        target[:, :, 1:] - target[:, :, :-1],
    )
    edge = edge + F.l1_loss(
        output[:, :, :, 1:] - output[:, :, :, :-1],
        target[:, :, :, 1:] - target[:, :, :, :-1],
    )
    return (
        F.l1_loss(output, target)
        + 0.5 * F.mse_loss(output, target)
        + 0.2 * _ssim_loss(output, target)
        + 0.04 * edge
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_pack(pred: torch.Tensor, target: torch.Tensor, objects: list[str]) -> dict:
    return {
        "overall": image_metrics(pred, target),
        "by_object": [
            {"object": name, "metrics": image_metrics(pred[i:i + 1], target[i:i + 1])}
            for i, name in enumerate(objects)
        ],
    }


def save_preview(path: Path, pred: torch.Tensor, labels: list, objects: list[str], has_label: list[bool]) -> None:
    size = int(pred.shape[-1])
    scale = 4
    tile = size * scale
    label_width = 168
    gap = 8
    header = 44
    canvas = Image.new("L", (label_width + 2 * tile + gap, header + len(objects) * (tile + gap) - gap), 0)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    draw.text((label_width + 4, 12), "target/blank", fill=255, font=font)
    draw.text((label_width + tile + gap + 4, 12), "prediction", fill=255, font=font)
    for i, name in enumerate(objects):
        y = header + i * (tile + gap)
        draw.text((4, y + tile // 2 - 8), name, fill=255, font=font)
        if has_label[i] and labels[i] is not None:
            target = Image.fromarray(
                np.rint(np.clip(np.asarray(labels[i]), 0, 1) * 255).astype(np.uint8)
            ).resize((tile, tile), Image.Resampling.NEAREST)
        else:
            target = Image.new("L", (tile, tile), 0)
        output = Image.fromarray(
            np.rint(np.clip(pred[i, 0].numpy(), 0, 1) * 255).astype(np.uint8)
        ).resize((tile, tile), Image.Resampling.NEAREST)
        canvas.paste(target, (label_width, y))
        canvas.paste(output, (label_width + tile + gap, y))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pack", required=True)
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--scale", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260944)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--no-dihedral-augmentation", action="store_true")
    parser.add_argument("--init-checkpoint")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pack_path = Path(args.input_pack)
    pack = torch.load(pack_path, map_location="cpu", weights_only=False)
    pred = pack["pred"].float()
    objects = list(pack["objects"])
    has_label = list(pack["has_label"])
    labelled_indices = [i for i, value in enumerate(has_label) if value and pack["labels"][i] is not None]
    labelled_objects = [objects[i] for i in labelled_indices]
    target = torch.stack([
        torch.as_tensor(pack["labels"][i], dtype=torch.float32) for i in labelled_indices
    ]).unsqueeze(1)
    train_pred = pred[labelled_indices]

    model = ResidualRefiner(args.channels, args.depth, args.scale)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(initial["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    history = []
    out = Path(args.exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        transform = 0 if args.no_dihedral_augmentation else epoch % 8
        output = model(dihedral(train_pred, transform))
        transformed_target = dihedral(target, transform)
        loss = loss_fn(output, transformed_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach())
        history.append({"epoch": epoch, "loss": value})
        if value < best_loss:
            best_loss = value
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            print(json.dumps({"epoch": epoch, "loss": value}, ensure_ascii=False), flush=True)
    if best_state is None:
        raise RuntimeError("refiner did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        refined = model(pred)
    refined_labelled = refined[labelled_indices]
    baseline = pred[labelled_indices]
    metrics = {
        "status": "completed",
        "protocol": "supervised full-fit output refiner; all 22 labelled objects used",
        "input_prediction_sha256": sha256(pack_path),
        "input_prediction": str(pack_path),
        "labelled_objects": labelled_objects,
        "unlabelled_objects": [name for name, value in zip(objects, has_label) if not value],
        "configuration": vars(args),
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "baseline_labelled": metric_pack(baseline, target, labelled_objects),
        "refined_labelled": metric_pack(refined_labelled, target, labelled_objects),
        "test_labels_used_for_training": True,
        "interpretation": "Fit/deployment quality only; not an unknown-object generalization score.",
    }
    torch.save({
        "pred": refined,
        "objects": objects,
        "has_label": has_label,
        "labels": pack["labels"],
        "member_dirs": pack.get("member_dirs", []),
        "input_pack": str(pack_path),
        "refiner": vars(args),
    }, out / "predictions_all.pt")
    torch.save({"model": model.state_dict(), "configuration": vars(args)}, out / "checkpoint.pt")
    (out / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    save_preview(out / "preview_all_objects.png", refined, pack["labels"], objects, has_label)
    (out / "DONE").write_text("completed\n")
    print(json.dumps({
        "best_epoch": best_epoch,
        "baseline": metrics["baseline_labelled"]["overall"],
        "refined": metrics["refined_labelled"]["overall"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
