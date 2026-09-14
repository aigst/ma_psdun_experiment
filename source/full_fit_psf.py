"""Full-fit PSF model for deployment snapshots.

This path intentionally uses every labelled object to improve the images shown
for the current capture set.  It is separate from the ring-blind strict
evaluation path and must not be reported as an unknown-object score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ma_psdun.core import MeasurementOperator
from new_sample_train import _batch, dihedral, load_sample, read_patterns
from ring_blind_train import build_operator_views, make_model, train_loss


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--patterns", required=True)
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--preprocess", default="detrend")
    p.add_argument("--detrend-width", type=int, default=127)
    p.add_argument("--gauss-sigma", type=float, default=40.0)
    p.add_argument("--fusion", default="mean")
    p.add_argument("--label-resample", default="bilinear")
    p.add_argument("--condition-mode", default="constant")
    p.add_argument("--psf-sigma", type=float, default=1.0)
    p.add_argument("--psf-sigma-y", type=float, default=None)
    p.add_argument("--psf-angle", type=float, default=0.0)
    p.add_argument("--stages", type=int, default=12)
    p.add_argument("--gain-init", type=float, default=4.0)
    p.add_argument("--lowpass-kernel", type=int, default=3)
    p.add_argument("--prior-residual-scale", type=float, default=0.5)
    p.add_argument("--shared-prior", action="store_true")
    p.add_argument("--adaptive-fusion-scale", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--consistency-weight", type=float, default=0.05)
    p.add_argument("--ssim-weight", type=float, default=0.2)
    p.add_argument("--tv-weight", type=float, default=0.0)
    p.add_argument("--edge-weight", type=float, default=0.0)
    p.add_argument("--dice-weight", type=float, default=0.0)
    p.add_argument("--pyramid-weight", type=float, default=0.0)
    p.add_argument("--binary-weight", type=float, default=0.0)
    p.add_argument("--dihedral-augmentation", action="store_true")
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--log-every", type=int, default=100)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("full-fit PSF training requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    patterns, samples = load_sample(
        args.data_root,
        args.patterns,
        args.size,
        args.preprocess,
        args.fusion,
        args.label_resample,
        args.detrend_width,
        args.gauss_sigma,
    )
    objects = sorted({sample["object"] for sample in samples})
    indices = [index for index, sample in enumerate(samples) if sample["object"] in set(objects)]
    op = MeasurementOperator(
        patterns.shape[1],
        patterns.shape[0],
        patterns=torch.from_numpy(patterns),
        device=device,
        psf_sigma=args.psf_sigma,
        psf_sigma_y=args.psf_sigma_y,
        psf_angle=args.psf_angle,
        image_hw=(args.size, args.size),
    )
    od_channels = int(np.asarray(samples[0]["raw"]).shape[0]) if np.asarray(samples[0]["raw"]).ndim == 2 else 1
    model = make_model(op, args, od_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batch = _batch(samples, indices, device, 0, args.condition_mode)
    views = build_operator_views(op, args.size, args.dihedral_augmentation)
    base_view = views[0]
    out = Path(args.exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_files = [
        Path(__file__),
        Path(__file__).with_name("new_sample_train.py"),
        Path(__file__).with_name("ring_blind_train.py"),
        Path(__file__).parent / "ma_psdun" / "model.py",
        Path(__file__).parent / "ma_psdun" / "core.py",
    ]
    config = vars(args).copy()
    config.update({
        "device_resolved": str(device),
        "objects": objects,
        "train_objects": objects,
        "labelled_object_count": len(objects),
        "od_channels": od_channels,
        "deployment_full_fit": True,
        "pattern_sha256": sha256_file(args.patterns),
        "source_sha256": {path.name: sha256_file(path) for path in source_files},
    })
    save_json(out / "config.json", config)
    history = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        transform = epoch % 8 if args.dihedral_augmentation else 0
        op.centered_patterns = views[transform]
        target = dihedral(batch[2], transform)
        pred, y = model(batch[0], batch[1], batch[3], (args.size, args.size))
        loss = train_loss(pred, target, y, op, args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        op.centered_patterns = base_view
        record = {"epoch": epoch, "train_loss": float(loss.detach()), "elapsed_s": time.time() - started}
        history.append(record)
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            print(json.dumps(record), flush=True)
    op.centered_patterns = base_view
    save_json(out / "training_history.json", history)
    torch.save({"epoch": args.epochs - 1, "model": model.state_dict()}, out / "checkpoint_final.pt")
    save_json(out / "fit_manifest.json", {
        "status": "completed",
        "deployment_full_fit": True,
        "labelled_objects": objects,
        "checkpoint_epoch": args.epochs - 1,
        "checkpoint_sha256": sha256_file(out / "checkpoint_final.pt"),
        "test_labels_used_for_training": True,
    })
    (out / "DONE").write_text("completed\n")


if __name__ == "__main__":
    main()
