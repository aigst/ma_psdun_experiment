"""Direct multi-OD reconstructor for the real 64x64 sample export.

Each OD is backprojected independently with the measured binary operator and
fed as a separate image channel.  The small U-Net learns denoising/fusion
without imposing an additional data-consistency update whose simulator may be
miscalibrated.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ma_psdun.core import MeasurementOperator
from ma_psdun.eval import image_metrics
from new_sample_train import load_sample


class MultiInputUNet(nn.Module):
    def __init__(self, in_channels: int = 4, base: int = 24):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(in_channels, base, 3, padding=1), nn.GELU(),
                                nn.Conv2d(base, base, 3, padding=1), nn.GELU())
        self.e2 = nn.Sequential(nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.GELU(),
                                nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.GELU())
        self.mid = nn.Sequential(nn.Conv2d(base * 2, base * 3, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(base * 3, base * 3, 3, padding=1), nn.GELU())
        self.d2 = nn.Sequential(nn.ConvTranspose2d(base * 3, base * 2, 4, stride=2, padding=1), nn.GELU(),
                                nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.GELU())
        self.head = nn.Sequential(nn.Conv2d(base * 2 + base, base, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(base, 1, 3, padding=1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        h1 = self.e1(x)
        h2 = self.e2(h1)
        h = self.mid(h2)
        h = self.d2(h)
        # Average backprojection is the stable identity; learned features
        # can use all OD channels to restore detail and suppress noise.
        skip = x.mean(dim=1, keepdim=True)
        return (skip + self.head(torch.cat([h, h1], dim=1))).clamp(0.0, 1.0)


def ssim_loss(pred, target):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx = F.avg_pool2d(pred, 7, 1, 3)
    my = F.avg_pool2d(target, 7, 1, 3)
    vx = F.avg_pool2d(pred * pred, 7, 1, 3) - mx * mx
    vy = F.avg_pool2d(target * target, 7, 1, 3) - my * my
    cov = F.avg_pool2d(pred * target, 7, 1, 3) - mx * my
    score = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return 1.0 - score.mean()


def loss_fn(pred, target, edge_weight, ssim_weight):
    l1 = (pred - target).abs().mean()
    mse = (pred - target).square().mean()
    dxp = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dxt = target[:, :, :, 1:] - target[:, :, :, :-1]
    dyp = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dyt = target[:, :, 1:, :] - target[:, :, :-1, :]
    edge = (dxp - dxt).abs().mean() + (dyp - dyt).abs().mean()
    return l1 + 0.5 * mse + ssim_weight * ssim_loss(pred, target) + edge_weight * edge


def corr(a, b):
    return float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--patterns", required=True)
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--label-resample", default="bilinear")
    p.add_argument("--detrend-width", type=int, default=31)
    p.add_argument("--base", type=int, default=24)
    p.add_argument("--gain", type=float, default=4.0)
    p.add_argument("--lowpass", type=int, default=3)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--edge-weight", type=float, default=0.1)
    p.add_argument("--ssim-weight", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--test-objects", default="obj18-20260901,obj19-20260901,obj20-20260901")
    p.add_argument("--val-objects", default="obj15-20260901,obj16-20260901,obj17-20260901")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    patterns, samples = load_sample(a.data_root, a.patterns, 64, "detrend", "multi",
                                     a.label_resample, a.detrend_width, 40.0)
    objects = sorted({s["object"] for s in samples})
    test_objects = [x for x in a.test_objects.split(",") if x]
    val_objects = [x for x in a.val_objects.split(",") if x]
    train_objects = [x for x in objects if x not in set(test_objects + val_objects)]
    groups = {k: [i for i, s in enumerate(samples) if s["object"] in names]
              for k, names in [("train", train_objects), ("val", val_objects), ("test", test_objects)]}
    if not all(groups.values()):
        raise ValueError(f"empty split: {groups}")
    op = MeasurementOperator(patterns.shape[1], patterns.shape[0],
                             patterns=torch.from_numpy(patterns), device=device)
    inputs, targets = [], []
    for s in samples:
        y = torch.from_numpy(s["raw"] - s["dark"])[None].to(device)
        x = op.adjoint(y).reshape(4, 64, 64) * a.gain
        if a.lowpass > 1:
            x = F.avg_pool2d(x[:, None], a.lowpass, 1, a.lowpass // 2)[:, 0]
        inputs.append(x)
        targets.append(torch.from_numpy(s["label"])[None].to(device))
    inputs, targets = torch.stack(inputs), torch.stack(targets)
    model = MultiInputUNet(4, a.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    out = Path(a.exp_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2) + "\n")
    best, best_epoch = -float("inf"), 0
    logs, t0 = [], time.time()
    ti = torch.tensor(groups["train"], device=device)
    vi = torch.tensor(groups["val"], device=device)
    for epoch in range(a.epochs):
        model.train()
        pred = model(inputs[ti])
        loss = loss_fn(pred, targets[ti], a.edge_weight, a.ssim_weight)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(inputs[vi]); vm = image_metrics(vp, targets[vi]); vm["corr"] = corr(vp, targets[vi])
        rec = {**vm, "epoch": epoch, "train_loss": float(loss.detach()), "elapsed_s": time.time() - t0}
        logs.append(rec)
        if epoch % 100 == 0 or epoch == a.epochs - 1: print(json.dumps(rec), flush=True)
        if vm["ssim"] > best:
            best, best_epoch = vm["ssim"], epoch
            torch.save({"epoch": epoch, "model": model.state_dict(), "validation": vm}, out / "checkpoint_best.pt")
    (out / "validation.jsonl").write_text("\n".join(json.dumps(x) for x in logs) + "\n")
    model.load_state_dict(torch.load(out / "checkpoint_best.pt", map_location=device, weights_only=False)["model"])
    model.eval(); test_idx = torch.tensor(groups["test"], device=device)
    with torch.no_grad():
        tp = model(inputs[test_idx]); tm = image_metrics(tp, targets[test_idx]); tm["corr"] = corr(tp, targets[test_idx])
    tm.update({"checkpoint_epoch": best_epoch, "status": "completed", "test_objects": test_objects,
               "train_objects": train_objects})
    (out / "test.json").write_text(json.dumps(tm, indent=2) + "\n")
    torch.save({"pred": tp.cpu(), "target": targets[test_idx].cpu()}, out / "test_samples.pt")
    rows = []
    for obj in test_objects:
        idx = [j for j, i in enumerate(groups["test"]) if samples[i]["object"] == obj]
        if idx:
            rows.append({"object": obj, "metrics": {**image_metrics(tp[idx], targets[test_idx][idx]), "corr": corr(tp[idx], targets[test_idx][idx])}})
    (out / "test_by_object.json").write_text(json.dumps(rows, indent=2) + "\n")
    canvas = []
    for j in range(tp.shape[0]):
        aa = targets[test_idx][j, 0].detach().cpu().numpy(); bb = tp[j, 0].detach().cpu().numpy()
        canvas.append(np.concatenate([aa, np.zeros((64, 2), np.float32), bb], 1))
    Image.fromarray(np.rint(np.clip(np.concatenate(canvas, 0), 0, 1) * 255).astype(np.uint8), mode="L").save(out / "preview_target_pred.png")
    (out / "DONE").write_text("completed\n")
    print(json.dumps(tm), flush=True)


if __name__ == "__main__":
    main()
