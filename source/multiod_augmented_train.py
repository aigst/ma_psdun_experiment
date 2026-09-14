"""MultiOD real training with a low-weight, physics-generated shape prior."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ma_psdun.core import MeasurementOperator
from ma_psdun.eval import image_metrics
from ma_psdun.model import MAPSDUN
from new_sample_train import _batch, _weighted_loss, load_sample, metrics


def dihedral_batch(x, transforms):
    out = []
    for i, t in enumerate(transforms.tolist()):
        z = x[i]
        if t >= 4:
            z = z.flip(-1); t -= 4
        out.append(torch.rot90(z, int(t), (-2, -1)))
    return torch.stack(out)


def pseudo_batch(labels, operator, batch_size, noise_stds, rng, device):
    ids = torch.randint(labels.shape[0], (batch_size,), generator=rng, device=device)
    transforms = torch.randint(0, 8, (batch_size,), generator=rng, device=device)
    target = dihedral_batch(labels[ids], transforms)
    ideal = operator.forward(target.flatten(1))
    stds = torch.tensor(noise_stds, device=device, dtype=ideal.dtype).view(1, 4, 1)
    y = ideal[:, None, :] + torch.randn((batch_size, 4, ideal.shape[-1]), generator=rng, device=device) * stds
    y = (y - y.mean(dim=-1, keepdim=True)) / y.std(dim=-1, keepdim=True).clamp_min(1e-5)
    raw, dark = y, torch.zeros_like(y)
    cond = torch.stack([torch.ones(batch_size, device=device), torch.ones(batch_size, device=device),
                        torch.full((batch_size,), 550.0, device=device)], dim=-1)
    return raw, dark, target, cond


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True); p.add_argument("--patterns", required=True); p.add_argument("--exp-dir", required=True)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--stages", type=int, default=12); p.add_argument("--prior-residual-scale", type=float, default=.5)
    p.add_argument("--epochs", type=int, default=700); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--pseudo-weight", type=float, default=.05); p.add_argument("--pseudo-noise", type=float, default=.01)
    p.add_argument("--consistency-weight", type=float, default=.01); p.add_argument("--ssim-weight", type=float, default=.2)
    p.add_argument("--edge-weight", type=float, default=.05); p.add_argument("--tv-weight", type=float, default=0.0); p.add_argument("--seed", type=int, default=20260960)
    p.add_argument("--test-objects", default="obj18-20260901,obj19-20260901"); p.add_argument("--val-objects", default="obj15-20260901,obj16-20260901,obj17-20260901")
    a = p.parse_args(); torch.manual_seed(a.seed); device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    patterns, samples = load_sample(a.data_root, a.patterns, 64, "detrend", "multi", "bilinear", 31, 40.0)
    objects = sorted({s["object"] for s in samples}); test_objects = [x for x in a.test_objects.split(",") if x]; val_objects = [x for x in a.val_objects.split(",") if x]
    train_objects = [x for x in objects if x not in set(test_objects + val_objects)]
    groups = {k: [i for i, s in enumerate(samples) if s["object"] in names] for k, names in [("train", train_objects), ("val", val_objects), ("test", test_objects)]}
    if not all(groups.values()): raise ValueError(f"empty split: {groups}")
    op = MeasurementOperator(patterns.shape[1], patterns.shape[0], patterns=torch.from_numpy(patterns), device=device)
    model = MAPSDUN(op, stages=a.stages, backprojection_gain_init=4.0, lowpass_kernel=3, od_channels=4, prior_residual_scale=a.prior_residual_scale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    out = Path(a.exp_dir); out.mkdir(parents=True, exist_ok=True); (out / "config.json").write_text(json.dumps(vars(a), indent=2) + "\n")
    train_batch = _batch(samples, groups["train"], device, 0, "constant"); val_batch = _batch(samples, groups["val"], device, 0, "constant")
    rng = torch.Generator(device=device).manual_seed(a.seed + 991); best_ssim = -float("inf"); best_mse = float("inf"); logs = []; t0 = time.time()
    # OD0 is cleaner; pseudo channels only add a small, controlled amount of
    # independent perturbation so the prior does not learn simulator noise.
    noise_stds = [a.pseudo_noise * x for x in (0.0, 0.7, 1.0, 1.4)]
    for epoch in range(a.epochs):
        model.train(); rp, ry = model(train_batch[0], train_batch[1], train_batch[3], (64, 64))
        real_loss = _weighted_loss(rp, train_batch[2], ry, op, a.consistency_weight, 0.0, a.ssim_weight, a.tv_weight, a.edge_weight)
        pp, pd, pt, pc = pseudo_batch(train_batch[2], op, len(groups["train"]), noise_stds, rng, device)
        sp, sy = model(pp, pd, pc, (64, 64))
        pseudo_loss = _weighted_loss(sp, pt, sy, op, a.consistency_weight, 0.0, a.ssim_weight, a.tv_weight, a.edge_weight)
        loss = real_loss + a.pseudo_weight * pseudo_loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad(): vp, vy = model(val_batch[0], val_batch[1], val_batch[3], (64, 64)); vm = metrics(vp, val_batch[2], vy, op)
        rec = {**vm, "epoch": epoch, "real_loss": float(real_loss.detach()), "pseudo_loss": float(pseudo_loss.detach()), "train_loss": float(loss.detach()), "elapsed_s": time.time() - t0}; logs.append(rec)
        if epoch % 50 == 0 or epoch == a.epochs - 1: print(json.dumps(rec), flush=True)
        if vm["ssim"] > best_ssim: best_ssim = vm["ssim"]; torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "validation": vm}, out / "checkpoint_best.pt")
        if vm["mse"] < best_mse: best_mse = vm["mse"]; torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "validation": vm}, out / "checkpoint_best_mse.pt")
    (out / "validation.jsonl").write_text("\n".join(json.dumps(x) for x in logs) + "\n")
    ck = torch.load(out / "checkpoint_best.pt", map_location=device, weights_only=False); model.load_state_dict(ck["model"]); model.eval(); test_batch = _batch(samples, groups["test"], device, 0, "constant")
    with torch.no_grad(): tp, ty = model(test_batch[0], test_batch[1], test_batch[3], (64, 64)); tm = metrics(tp, test_batch[2], ty, op)
    tm.update({"checkpoint_epoch": int(ck["epoch"]), "status": "completed", "test_objects": test_objects, "train_objects": train_objects})
    (out / "test.json").write_text(json.dumps(tm, indent=2) + "\n"); torch.save({"pred": tp.cpu(), "target": test_batch[2].cpu()}, out / "test_samples.pt")
    rows = []
    for obj in test_objects:
        ids = [j for j, i in enumerate(groups["test"]) if samples[i]["object"] == obj]
        if ids: rows.append({"object": obj, "metrics": metrics(tp[ids], test_batch[2][ids], ty[ids], op)})
    (out / "test_by_object.json").write_text(json.dumps(rows, indent=2) + "\n")
    canvas = []
    for j in range(tp.shape[0]):
        aa = test_batch[2][j, 0].detach().cpu().numpy(); bb = tp[j, 0].detach().cpu().numpy(); canvas.append(np.concatenate([aa, np.zeros((64, 2), np.float32), bb], 1))
    Image.fromarray(np.rint(np.clip(np.concatenate(canvas, 0), 0, 1) * 255).astype(np.uint8), mode="L").save(out / "preview_target_pred.png"); (out / "DONE").write_text("completed\n"); print(json.dumps(tm), flush=True)


if __name__ == "__main__": main()
