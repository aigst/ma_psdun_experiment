"""Small direct 2-D reconstructor for the low-sample real-object setting."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ma_psdun.core import MeasurementOperator
from ma_psdun.eval import image_metrics
from new_sample_train import load_sample


class SmallUNet(nn.Module):
    def __init__(self, base: int = 16):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(1, base, 3, padding=1), nn.GELU(),
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
        # Residual around the low-pass physical reconstruction; this keeps
        # the network from inventing unsupported high-frequency structure.
        return (x + self.head(torch.cat([h, h1], dim=1))).clamp(0.0, 1.0)


def ssim_loss(pred, target):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx = F.avg_pool2d(pred, 7, 1, 3)
    my = F.avg_pool2d(target, 7, 1, 3)
    vx = F.avg_pool2d(pred * pred, 7, 1, 3) - mx * mx
    vy = F.avg_pool2d(target * target, 7, 1, 3) - my * my
    cov = F.avg_pool2d(pred * target, 7, 1, 3) - mx * my
    score = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return 1 - score.mean()


def loss_fn(pred, target, edge_weight: float):
    l1 = (pred - target).abs().mean()
    mse = (pred - target).square().mean()
    dxp = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dxt = target[:, :, :, 1:] - target[:, :, :, :-1]
    dyp = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dyt = target[:, :, 1:, :] - target[:, :, :-1, :]
    edge = (dxp - dxt).abs().mean() + (dyp - dyt).abs().mean()
    return l1 + 0.5 * mse + 0.2 * ssim_loss(pred, target) + edge_weight * edge


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--patterns", required=True)
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--label-resample", default="bilinear")
    p.add_argument("--detrend-width", type=int, default=31)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--gain", type=float, default=4.0)
    p.add_argument("--lowpass", type=int, default=3)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--edge-weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--test-objects", default="obj18-20260901,obj19-20260901,obj20-20260901")
    p.add_argument("--val-objects", default="obj15-20260901,obj16-20260901,obj17-20260901")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    patterns, samples = load_sample(a.data_root, a.patterns, 64, "detrend", "mean",
                                     a.label_resample, a.detrend_width, 40.0)
    objs = sorted({s["object"] for s in samples})
    test_objects = [x for x in a.test_objects.split(",") if x]
    val_objects = [x for x in a.val_objects.split(",") if x]
    train_objects = [x for x in objs if x not in set(test_objects + val_objects)]
    groups = {k: [i for i, s in enumerate(samples) if s["object"] in names]
              for k, names in [("train", train_objects), ("val", val_objects), ("test", test_objects)]}
    op = MeasurementOperator(patterns.shape[1], patterns.shape[0],
                             patterns=torch.from_numpy(patterns), device=device)
    ys = []
    labels = []
    for s in samples:
        y = torch.from_numpy(s["raw"] - s["dark"])[None].to(device)
        x = op.adjoint(y).reshape(1, 1, 64, 64) * a.gain
        if a.lowpass > 1:
            x = F.avg_pool2d(x, a.lowpass, 1, a.lowpass // 2)
        ys.append(x[0])
        labels.append(torch.from_numpy(s["label"])[None].to(device))
    inputs = torch.stack(ys)
    targets = torch.stack(labels)
    model = SmallUNet(a.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    out = Path(a.exp_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2) + "\n")
    best = -float("inf"); best_epoch = 0; logs = []; t0 = time.time()
    ti = torch.tensor(groups["train"], device=device); vi = torch.tensor(groups["val"], device=device)
    for epoch in range(a.epochs):
        model.train(); pred = model(inputs[ti]); loss = loss_fn(pred, targets[ti], a.edge_weight)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(inputs[vi]); vm = image_metrics(vp, targets[vi]); vm["corr"] = float(torch.corrcoef(torch.stack([vp.flatten(), targets[vi].flatten()]))[0, 1])
        rec = {**vm, "epoch": epoch, "train_loss": float(loss.detach()), "elapsed_s": time.time() - t0}; logs.append(rec)
        if epoch % 100 == 0 or epoch == a.epochs - 1: print(json.dumps(rec), flush=True)
        if vm["ssim"] > best:
            best = vm["ssim"]; best_epoch = epoch
            torch.save({"epoch": epoch, "model": model.state_dict(), "validation": vm}, out / "checkpoint_best.pt")
    (out / "validation.jsonl").write_text("\n".join(json.dumps(x) for x in logs) + "\n")
    model.load_state_dict(torch.load(out / "checkpoint_best.pt", map_location=device, weights_only=False)["model"]); model.eval()
    with torch.no_grad():
        tp = model(inputs[torch.tensor(groups["test"], device=device)])
        tm = image_metrics(tp, targets[torch.tensor(groups["test"], device=device)])
        tm["corr"] = float(torch.corrcoef(torch.stack([tp.flatten(), targets[torch.tensor(groups["test"], device=device)].flatten()]))[0, 1])
        tm.update({"checkpoint_epoch": best_epoch, "status": "completed"})
    (out / "test.json").write_text(json.dumps(tm, indent=2) + "\n")
    print(json.dumps(tm), flush=True)


if __name__ == "__main__":
    main()
