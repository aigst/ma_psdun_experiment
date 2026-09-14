"""Real-object training with physics-consistent target augmentation.

The labelled real set has only a handful of independent objects.  This
trainer augments those labels with dihedral transforms and synthesizes their
measurements through the same centered 0/1 operator.  Real measurements stay
in every update; pseudo pairs only regularize the image prior and are never
made from validation/test labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ma_psdun.core import MeasurementOperator
from ma_psdun.eval import image_metrics
from ma_psdun.model import MAPSDUN
from new_sample_train import _batch, _weighted_loss, load_sample, metrics


def dihedral(x: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    out = x
    for k in range(x.shape[0]):
        t = int(transform[k])
        z = out[k]
        if t >= 4:
            z = z.flip(-1)
            t -= 4
        out[k] = torch.rot90(z, t, (-2, -1))
    return out


def pseudo_batch(labels: torch.Tensor, operator: MeasurementOperator, batch_size: int,
                 noise_std: float, generator: torch.Generator, device: torch.device):
    ids = torch.randint(labels.shape[0], (batch_size,), generator=generator, device=device)
    target = labels[ids].clone()
    transform = torch.randint(0, 8, (batch_size,), generator=generator, device=device)
    target = dihedral(target, transform)
    ideal = operator.forward(target.flatten(1))
    noise = torch.randn(ideal.shape, generator=generator, device=device) * noise_std
    y = ideal + noise
    y = (y - y.mean(dim=-1, keepdim=True)) / y.std(dim=-1, keepdim=True).clamp_min(1e-5)
    raw, dark = y[:, None], torch.zeros_like(y[:, None])
    cond = torch.stack([
        torch.ones(batch_size, device=device),
        torch.ones(batch_size, device=device),
        torch.full((batch_size,), 550.0, device=device),
    ], dim=-1)
    return raw, dark, target, cond


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--patterns", required=True)
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--preprocess", default="detrend")
    p.add_argument("--fusion", default="mean")
    p.add_argument("--label-resample", default="bilinear")
    p.add_argument("--stages", type=int, default=12)
    p.add_argument("--gain-init", type=float, default=4.0)
    p.add_argument("--lowpass-kernel", type=int, default=3)
    p.add_argument("--prior-residual-scale", type=float, default=0.5)
    p.add_argument("--shared-prior", action="store_true")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--consistency-weight", type=float, default=0.05)
    p.add_argument("--ssim-weight", type=float, default=0.2)
    p.add_argument("--pseudo-weight", type=float, default=0.25)
    p.add_argument("--pseudo-noise-std", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--test-objects", default="obj18-20260901,obj19-20260901,obj20-20260901")
    p.add_argument("--val-objects", default="obj15-20260901,obj16-20260901,obj17-20260901")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    patterns, samples = load_sample(a.data_root, a.patterns, a.size, a.preprocess,
                                     a.fusion, a.label_resample)
    all_objects = sorted({s["object"] for s in samples})
    test_objects = [x for x in a.test_objects.split(",") if x]
    val_objects = [x for x in a.val_objects.split(",") if x]
    train_objects = [x for x in all_objects if x not in set(test_objects + val_objects)]
    groups = {
        "train": [i for i, s in enumerate(samples) if s["object"] in train_objects],
        "val": [i for i, s in enumerate(samples) if s["object"] in val_objects],
        "test": [i for i, s in enumerate(samples) if s["object"] in test_objects],
    }
    if not all(groups.values()):
        raise ValueError(f"empty split: {groups}")
    op = MeasurementOperator(patterns.shape[1], patterns.shape[0],
                             patterns=torch.from_numpy(patterns), device=device)
    model = MAPSDUN(op, stages=a.stages, backprojection_gain_init=a.gain_init,
                    lowpass_kernel=a.lowpass_kernel,
                    od_channels=1, prior_residual_scale=a.prior_residual_scale,
                    shared_prior=a.shared_prior).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    out = Path(a.exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = vars(a).copy()
    cfg.update({"objects": all_objects, "train_objects": train_objects,
                "val_objects": val_objects, "test_objects": test_objects,
                "split_sizes": {k: len(v) for k, v in groups.items()},
                "pattern_sha256": hashlib.sha256(Path(a.patterns).read_bytes()).hexdigest()})
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out / "split.json").write_text(json.dumps({k: [samples[i]["object"] + "/" + samples[i]["od"] for i in v]
                                                  for k, v in groups.items()}, indent=2) + "\n")
    train_batch = _batch(samples, groups["train"], device, 0, "constant")
    val_batch = _batch(samples, groups["val"], device, 0, "constant")
    label_bank = train_batch[2]
    rng = torch.Generator(device=device).manual_seed(a.seed + 991)
    best_ssim = -float("inf")
    best_mse = float("inf")
    log = []
    t0 = time.time()
    for epoch in range(a.epochs):
        model.train()
        rp, ry = model(train_batch[0], train_batch[1], train_batch[3], (a.size, a.size))
        real_loss = _weighted_loss(rp, train_batch[2], ry, op, a.consistency_weight,
                                    0.0, a.ssim_weight)
        pp, pd, pt, pc = pseudo_batch(label_bank, op, len(groups["train"]),
                                      a.pseudo_noise_std, rng, device)
        sp, sy = model(pp, pd, pc, (a.size, a.size))
        pseudo_loss = _weighted_loss(sp, pt, sy, op, a.consistency_weight,
                                     0.0, a.ssim_weight)
        loss = real_loss + a.pseudo_weight * pseudo_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            vp, vy = model(val_batch[0], val_batch[1], val_batch[3], (a.size, a.size))
            vm = metrics(vp, val_batch[2], vy, op)
        rec = {**vm, "epoch": epoch, "real_loss": float(real_loss.detach()),
               "pseudo_loss": float(pseudo_loss.detach()), "train_loss": float(loss.detach()),
               "elapsed_s": time.time() - t0}
        log.append(rec)
        if epoch % 50 == 0 or epoch == a.epochs - 1:
            print(json.dumps(rec), flush=True)
        if vm["ssim"] > best_ssim:
            best_ssim = vm["ssim"]
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": opt.state_dict(), "validation": vm}, out / "checkpoint_best.pt")
        if vm["mse"] < best_mse:
            best_mse = vm["mse"]
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": opt.state_dict(), "validation": vm}, out / "checkpoint_best_mse.pt")
        if epoch % 100 == 0 or epoch == a.epochs - 1:
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": opt.state_dict()}, out / "checkpoint_latest.pt")
    (out / "validation.jsonl").write_text("\n".join(json.dumps(x) for x in log) + "\n")
    ck = torch.load(out / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    test_batch = _batch(samples, groups["test"], device, 0, "constant")
    with torch.no_grad():
        tp, ty = model(test_batch[0], test_batch[1], test_batch[3], (a.size, a.size))
        tm = metrics(tp, test_batch[2], ty, op)
    tm.update({"checkpoint_epoch": int(ck["epoch"]), "status": "completed",
               "test_objects": test_objects, "train_objects": train_objects})
    (out / "test.json").write_text(json.dumps(tm, indent=2) + "\n")
    torch.save({"pred": tp.cpu(), "target": test_batch[2].cpu()}, out / "test_samples.pt")
    rows = []
    for obj in test_objects:
        idx = [j for j, i in enumerate(groups["test"]) if samples[i]["object"] == obj]
        if idx:
            rows.append({"object": obj, "metrics": metrics(tp[idx], test_batch[2][idx], ty[idx], op),
                         "od": [samples[groups["test"][j]]["od"] for j in idx]})
    (out / "test_by_object.json").write_text(json.dumps(rows, indent=2) + "\n")
    canvas = []
    for j in range(tp.shape[0]):
        aa = test_batch[2][j, 0].detach().cpu().numpy()
        bb = tp[j, 0].detach().cpu().numpy()
        gap = np.zeros((a.size, 2), np.float32)
        canvas.append(np.concatenate([aa, gap, bb], 1))
    from PIL import Image
    Image.fromarray(np.rint(np.clip(np.concatenate(canvas, 0), 0, 1) * 255).astype(np.uint8), mode="L").save(out / "preview_target_pred.png")
    (out / "DONE").write_text("completed\n")
    print(json.dumps(tm), flush=True)


if __name__ == "__main__":
    main()
