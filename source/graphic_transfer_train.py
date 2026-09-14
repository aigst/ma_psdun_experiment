"""Graphic-target synthetic pretraining followed by real-object fine-tuning.

The real sample export is dominated by binary graphic targets (bars, text,
rings and logos).  This trainer creates a diverse, antialiased graphic bank,
uses the measured 0/1 pattern for its forward projection, and bootstraps the
residual sequence from *training objects only*.  It then fine-tunes the same
MA-PSDUN on the strict object-level real split.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from ma_psdun.core import MeasurementOperator
from ma_psdun.model import MAPSDUN
from new_sample_train import _batch, _weighted_loss, load_sample, metrics


WORDS = (
    "TECHNOLOGY", "ENGINEERING", "SCIENCE", "QUALITY", "BOLT", "OPTICS",
    "DATA", "VISION", "AI", "2026", "TEST", "BEIJING", "LAB", "MA PSDUN",
)


def font_paths() -> list[str]:
    paths = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    return paths or [""]


def graphic_bank(count: int, size: int, seed: int) -> torch.Tensor:
    """Render varied high-contrast graphics at 4x and downsample to size."""
    rng = random.Random(seed)
    fonts = font_paths()
    bank: list[np.ndarray] = []
    scale = 4
    hi = size * scale
    for _ in range(count):
        image = Image.new("L", (hi, hi), 0)
        draw = ImageDraw.Draw(image)
        primitive_count = rng.randint(1, 5)
        for _ in range(primitive_count):
            kind = rng.choice(("rect", "ellipse", "ring", "line", "poly"))
            x0, y0 = rng.randint(-hi // 2, hi), rng.randint(-hi // 2, hi)
            w, h = rng.randint(8, hi // 2), rng.randint(8, hi // 2)
            box = (x0, y0, x0 + w, y0 + h)
            if kind == "rect":
                draw.rectangle(box, fill=rng.randint(210, 255))
            elif kind == "ellipse":
                draw.ellipse(box, fill=rng.randint(210, 255))
            elif kind == "ring":
                width = rng.randint(3, max(4, min(w, h) // 5))
                draw.ellipse(box, outline=rng.randint(210, 255), width=width)
            elif kind == "line":
                draw.line((x0, y0, x0 + w, y0 + rng.randint(-hi // 2, hi // 2)),
                          fill=rng.randint(210, 255), width=rng.randint(2, max(3, hi // 12)))
            else:
                points = []
                for k in range(rng.randint(3, 7)):
                    points.append((x0 + rng.randint(-w // 2, max(1, w)),
                                   y0 + rng.randint(-h // 2, max(1, h))))
                draw.polygon(points, fill=rng.randint(210, 255))
        if rng.random() < 0.75:
            text = rng.choice(WORDS)
            fpath = rng.choice(fonts)
            fsize = rng.randint(max(10, hi // 6), max(12, hi // 2))
            try:
                font = ImageFont.truetype(fpath, fsize) if fpath else ImageFont.load_default()
            except OSError:
                font = ImageFont.load_default()
            layer = Image.new("L", (hi, hi), 0)
            ImageDraw.Draw(layer).text((rng.randint(-hi // 3, hi // 2), rng.randint(-hi // 3, hi // 2)),
                                       text, font=font, fill=rng.randint(210, 255), stroke_width=0)
            layer = layer.rotate(rng.uniform(-55.0, 55.0), resample=Image.Resampling.BICUBIC,
                                 expand=False, fillcolor=0)
            image = Image.fromarray(np.maximum(np.asarray(image), np.asarray(layer)).astype(np.uint8))
        if rng.random() < 0.25:
            image = image.rotate(90, expand=False)
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        bank.append(np.asarray(image, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(bank))[:, None]


def label_aug_bank(labels: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    """Create valid image-space transforms of train labels only."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(labels.shape[0], (count,), generator=rng)
    source = labels[ids].float()
    theta = torch.zeros(count, 2, 3)
    theta[:, 0, 0] = torch.empty(count).uniform_(0.55, 1.45, generator=rng)
    theta[:, 1, 1] = torch.empty(count).uniform_(0.55, 1.45, generator=rng)
    angle = torch.empty(count).uniform_(-3.14159, 3.14159, generator=rng)
    ca, sa = angle.cos(), angle.sin()
    sx, sy = theta[:, 0, 0].clone(), theta[:, 1, 1].clone()
    theta[:, 0, 0], theta[:, 0, 1] = ca * sx, -sa * sy
    theta[:, 1, 0], theta[:, 1, 1] = sa * sx, ca * sy
    theta[:, 0, 2] = torch.empty(count).uniform_(-0.45, 0.45, generator=rng)
    theta[:, 1, 2] = torch.empty(count).uniform_(-0.45, 0.45, generator=rng)
    grid = F.affine_grid(theta, source.shape, align_corners=False)
    output = F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    morph = torch.rand(count, generator=rng)
    for i in range(count):
        if morph[i] < 0.18:
            output[i : i + 1] = F.max_pool2d(output[i : i + 1], 3, 1, 1)
        elif morph[i] > 0.82:
            output[i : i + 1] = -F.max_pool2d(-output[i : i + 1], 3, 1, 1)
    return output.clamp(0.0, 1.0)


def normalized_projection(operator: MeasurementOperator, images: torch.Tensor) -> torch.Tensor:
    p = operator.forward(images.flatten(1))
    return (p - p.mean(dim=1, keepdim=True)) / p.std(dim=1, keepdim=True).clamp_min(1e-6)


def make_synthetic_batch(
    bank: torch.Tensor,
    operator: MeasurementOperator,
    coeff: torch.Tensor,
    residual: torch.Tensor,
    batch_size: int,
    residual_scale: float,
    rng: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = torch.randint(bank.shape[0], (batch_size,), generator=rng, device=device)
    target = bank[ids].to(device)
    projection = normalized_projection(operator, target)
    rid = torch.randint(residual.shape[0], (batch_size,), generator=rng, device=device)
    noise = residual[rid]
    signal = coeff.view(1, 4, 1) * projection[:, None, :]
    y = signal + residual_scale * noise
    y = (y - y.mean(dim=-1, keepdim=True)) / y.std(dim=-1, keepdim=True).clamp_min(1e-6)
    raw = y
    dark = torch.zeros_like(raw)
    cond = torch.stack([
        torch.ones(batch_size, device=device),
        torch.ones(batch_size, device=device),
        torch.full((batch_size,), 550.0, device=device),
    ], dim=-1)
    return raw, dark, target, cond


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrain-steps", type=int, default=500)
    parser.add_argument("--finetune-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bank-size", type=int, default=2048)
    parser.add_argument("--bank-mode", choices=["graphic", "hybrid", "labels"], default="hybrid")
    parser.add_argument("--freeze-physics", action="store_true")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--pretrain-lr", type=float, default=2e-4)
    parser.add_argument("--finetune-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260970)
    parser.add_argument("--test-objects", default="obj18-20260901,obj19-20260901,obj20-20260901")
    parser.add_argument("--val-objects", default="obj15-20260901,obj16-20260901,obj17-20260901")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    patterns, samples = load_sample(args.data_root, args.patterns, 64, "detrend", "multi", "bilinear", 31, 40.0)
    objects = sorted({sample["object"] for sample in samples})
    tests = [value for value in args.test_objects.split(",") if value]
    vals = [value for value in args.val_objects.split(",") if value]
    train_objects = [value for value in objects if value not in set(tests + vals)]
    groups = {
        "train": [i for i, sample in enumerate(samples) if sample["object"] in train_objects],
        "val": [i for i, sample in enumerate(samples) if sample["object"] in vals],
        "test": [i for i, sample in enumerate(samples) if sample["object"] in tests],
    }
    if not all(groups.values()):
        raise ValueError(f"empty split: {groups}")
    operator = MeasurementOperator(patterns.shape[1], patterns.shape[0], patterns=torch.from_numpy(patterns), device=device)
    train_batch = _batch(samples, groups["train"], device, 0, "constant")
    val_batch = _batch(samples, groups["val"], device, 0, "constant")
    test_batch = _batch(samples, groups["test"], device, 0, "constant")

    # Build a residual bootstrap from train objects only.  Coefficients are
    # correlation-like signal gains in the normalized measurement space.
    train_y = train_batch[0] - train_batch[1]
    train_y = (train_y - train_y.mean(dim=-1, keepdim=True)) / train_y.std(dim=-1, keepdim=True).clamp_min(1e-6)
    train_projection = normalized_projection(operator, train_batch[2])
    coeff_rows = (train_y * train_projection[:, None, :]).mean(dim=-1)
    coeff = coeff_rows.mean(dim=0).detach()
    residual = train_y - coeff.view(1, 4, 1) * train_projection[:, None, :]
    if args.bank_mode == "graphic":
        bank = graphic_bank(args.bank_size, 64, args.seed + 77)
    elif args.bank_mode == "labels":
        bank = label_aug_bank(train_batch[2].cpu(), args.bank_size, args.seed + 77)
    else:
        n_label = args.bank_size // 2
        bank = torch.cat([
            label_aug_bank(train_batch[2].cpu(), n_label, args.seed + 77),
            graphic_bank(args.bank_size - n_label, 64, args.seed + 177),
        ], dim=0)
    bank = bank.to(device)

    model = MAPSDUN(operator, stages=12, backprojection_gain_init=4.0, lowpass_kernel=3,
                    od_channels=4, prior_residual_scale=0.50).to(device)
    out = Path(args.exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    cfg.update({"objects": objects, "train_objects": train_objects, "val_objects": vals,
                "test_objects": tests, "split_sizes": {k: len(v) for k, v in groups.items()},
                "coeff": coeff.cpu().tolist(), "pattern_sha256": hashlib.sha256(Path(args.patterns).read_bytes()).hexdigest()})
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    if args.freeze_physics:
        for name, parameter in model.named_parameters():
            if name.startswith("tcm.") or name in {"backprojection_gain", "backprojection_bias", "rho_logits"}:
                parameter.requires_grad_(False)
    pretrain_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    opt = torch.optim.AdamW(pretrain_parameters, lr=args.pretrain_lr, weight_decay=args.weight_decay)
    rng = torch.Generator(device=device).manual_seed(args.seed + 991)
    started = time.time()
    pretrain_logs = []
    for step in range(args.pretrain_steps):
        model.train()
        raw, dark, target, cond = make_synthetic_batch(bank, operator, coeff, residual,
                                                        args.batch_size, args.residual_scale, rng, device)
        pred, y = model(raw, dark, cond, (64, 64))
        loss = _weighted_loss(pred, target, y, operator, 0.01, 0.0, 0.2, 0.0, 0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == args.pretrain_steps - 1:
            pretrain_logs.append({"step": step, "loss": float(loss.detach())})
            print(json.dumps(pretrain_logs[-1]), flush=True)

    # Small-step adaptation on real objects.  Validation selects the checkpoint.
    if args.freeze_physics:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        opt = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
    else:
        for group in opt.param_groups:
            group["lr"] = args.finetune_lr
    best_ssim = -float("inf")
    best_epoch = -1
    logs = []
    for epoch in range(args.finetune_epochs):
        model.train()
        pred, y = model(train_batch[0], train_batch[1], train_batch[3], (64, 64))
        loss = _weighted_loss(pred, train_batch[2], y, operator, 0.01, 0.0, 0.2, 0.0, 0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            vp, vy = model(val_batch[0], val_batch[1], val_batch[3], (64, 64))
            vm = metrics(vp, val_batch[2], vy, operator)
        rec = {**vm, "epoch": epoch, "loss": float(loss.detach())}
        logs.append(rec)
        if epoch % 50 == 0 or epoch == args.finetune_epochs - 1:
            print(json.dumps(rec), flush=True)
        if vm["ssim"] > best_ssim:
            best_ssim = vm["ssim"]
            best_epoch = epoch
            torch.save({"epoch": epoch, "model": model.state_dict(), "validation": vm}, out / "checkpoint_best.pt")

    ck = torch.load(out / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        tp, ty = model(test_batch[0], test_batch[1], test_batch[3], (64, 64))
        tm = metrics(tp, test_batch[2], ty, operator)
    result = {"status": "completed", "checkpoint_epoch": int(ck["epoch"]), "test": tm,
              "validation": ck["validation"], "pretrain_logs": pretrain_logs,
              "elapsed_s": time.time() - started, "train_objects": train_objects,
              "test_objects": tests}
    rows = []
    for obj in tests:
        ids = [j for j, i in enumerate(groups["test"]) if samples[i]["object"] == obj]
        if ids:
            rows.append({"object": obj, "metrics": metrics(tp[ids], test_batch[2][ids], ty[ids], operator)})
    result["test_by_object"] = rows
    (out / "validation.jsonl").write_text("\n".join(json.dumps(row) for row in logs) + "\n")
    (out / "test.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save({"pred": tp.cpu(), "target": test_batch[2].cpu()}, out / "test_samples.pt")
    canvas = []
    for j in range(tp.shape[0]):
        canvas.append(np.concatenate([test_batch[2][j, 0].cpu().numpy(), np.zeros((64, 2), np.float32), tp[j, 0].detach().cpu().numpy()], axis=1))
    Image.fromarray(np.rint(np.clip(np.concatenate(canvas, axis=0), 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(out / "preview_target_pred.png")
    (out / "DONE").write_text("completed\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
