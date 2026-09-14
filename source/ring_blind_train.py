"""Ring-blind MA-PSDUN training on the 2026-09-01 real dataset.

The selection stage never uses circular-text targets for supervision. Spatial
augmentation transforms the reconstruction target and the physical pattern
operator together, preserving the measurement equation exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ma_psdun.core import MeasurementOperator
from ma_psdun.model import MAPSDUN
from new_sample_train import _batch, _weighted_loss, dihedral, load_sample, metrics


DEFAULT_TEST = "obj18-20260901,obj19-20260901,obj20-20260901"
DEFAULT_AUDIT = "obj15-20260901,obj16-20260901"
DEFAULT_FORBIDDEN = "obj15-20260901,obj16-20260901,obj19-20260901"


def parse_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def validate_split(
    available: list[str],
    train: list[str],
    validation: list[str],
    test: list[str],
    audit: list[str],
    forbidden_supervised: list[str],
    require_validation: bool,
) -> dict:
    groups = {
        "train": train,
        "validation": validation,
        "test": test,
        "audit": audit,
    }
    unknown = {name: sorted(set(values) - set(available)) for name, values in groups.items()}
    unknown = {name: values for name, values in unknown.items() if values}
    if unknown:
        raise ValueError(f"split contains unknown objects: {unknown}; available={available}")
    if not train or not test or (require_validation and not validation):
        raise ValueError(f"required split is empty: {groups}")
    for name, values in groups.items():
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate objects in {name}: {values}")
    overlaps = {}
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = sorted(set(groups[left]) & set(groups[right]))
            if overlap:
                overlaps[f"{left}:{right}"] = overlap
    if overlaps:
        raise ValueError(f"object-level split overlap: {overlaps}")
    forbidden = set(forbidden_supervised)
    supervised_overlap = {
        "train": sorted(forbidden & set(train)),
        "validation": sorted(forbidden & set(validation)),
    }
    if any(supervised_overlap.values()):
        raise ValueError(f"forbidden circular-text targets used for supervision: {supervised_overlap}")
    return {
        "available_objects": available,
        **groups,
        "forbidden_supervised_objects": forbidden_supervised,
        "forbidden_intersection": supervised_overlap,
        "object_level_disjoint": True,
        "ring_blind_supervision": True,
    }


def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    numerator = 2.0 * (pred_flat * target_flat).sum(dim=1) + 1e-5
    denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1) + 1e-5
    return (1.0 - numerator / denominator).mean()


def pyramid_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = []
    for scale in (2, 4):
        losses.append(F.l1_loss(F.avg_pool2d(pred, scale), F.avg_pool2d(target, scale)))
    return torch.stack(losses).mean()


def binary_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """A bounded BCE term for the mostly binary optical reference targets."""
    return F.binary_cross_entropy(pred.clamp(1e-4, 1.0 - 1e-4), target)


def build_operator_views(op: MeasurementOperator, size: int, enabled: bool) -> list[torch.Tensor]:
    if not enabled:
        return [op.centered_patterns]
    pattern_images = op.centered_patterns.reshape(op.centered_patterns.shape[0], 1, size, size)
    return [dihedral(pattern_images, transform).flatten(1).contiguous() for transform in range(8)]


def train_loss(pred, target, y, op, args):
    base = _weighted_loss(
        pred,
        target,
        y,
        op,
        args.consistency_weight,
        ssim_weight=args.ssim_weight,
        tv_weight=args.tv_weight,
        edge_weight=args.edge_weight,
    )
    return (base + args.dice_weight * soft_dice_loss(pred, target)
            + args.pyramid_weight * pyramid_l1_loss(pred, target)
            + args.binary_weight * binary_loss(pred, target))


def make_model(op: MeasurementOperator, args, od_channels: int) -> MAPSDUN:
    return MAPSDUN(
        op,
        stages=args.stages,
        backprojection_gain_init=args.gain_init,
        lowpass_kernel=args.lowpass_kernel,
        od_channels=od_channels,
        prior_residual_scale=args.prior_residual_scale,
        shared_prior=args.shared_prior,
        adaptive_fusion_scale=args.adaptive_fusion_scale,
    )


def sample_indices(samples, objects: list[str]) -> list[int]:
    selected = set(objects)
    return [index for index, sample in enumerate(samples) if sample["object"] in selected]


def evaluate_objects(model, samples, names, device, op, args):
    indices = sample_indices(samples, names)
    if not indices:
        return None
    batch = _batch(samples, indices, device, 0, args.condition_mode)
    model.eval()
    with torch.no_grad():
        pred, y = model(batch[0], batch[1], batch[3], (args.size, args.size))
        overall = metrics(pred, batch[2], y, op)
    rows = []
    for name in names:
        positions = [j for j, index in enumerate(indices) if samples[index]["object"] == name]
        if positions:
            rows.append({
                "object": name,
                "metrics": metrics(pred[positions], batch[2][positions], y[positions], op),
                "od": [samples[indices[j]]["od"] for j in positions],
            })
    return {
        "overall": overall,
        "by_object": rows,
        "pred": pred.detach().cpu(),
        "target": batch[2].detach().cpu(),
        "objects": names,
    }


def save_preview(path: Path, packs: list[dict], size: int) -> None:
    rows = []
    for pack in packs:
        for target, pred in zip(pack["target"][:, 0].numpy(), pack["pred"][:, 0].numpy()):
            gap = np.zeros((size, 2), dtype=np.float32)
            rows.append(np.concatenate([target, gap, pred], axis=1))
    canvas = np.concatenate(rows, axis=0)
    Image.fromarray(np.rint(np.clip(canvas, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


def prepare(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError(f"CUDA was requested but is unavailable: {args.device}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
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
        getattr(args, "label_crop", "center"),
        getattr(args, "label_contrast", "percentile"),
        getattr(args, "label_percentile_low", 1.0),
        getattr(args, "label_percentile_high", 99.0),
    )
    available = sorted({sample["object"] for sample in samples})
    train = parse_names(args.train_objects)
    validation = parse_names(args.val_objects)
    test = parse_names(args.test_objects)
    audit = parse_names(args.audit_objects)
    forbidden = parse_names(args.forbidden_supervised_objects)
    split = validate_split(available, train, validation, test, audit, forbidden, args.mode == "select")
    op = MeasurementOperator(
        patterns.shape[1],
        patterns.shape[0],
        patterns=torch.from_numpy(patterns),
        device=device,
        psf_sigma=args.psf_sigma,
        psf_sigma_y=getattr(args, "psf_sigma_y", None),
        psf_angle=getattr(args, "psf_angle", 0.0),
        image_hw=(args.size, args.size),
    )
    first_raw = np.asarray(samples[0]["raw"])
    od_channels = int(first_raw.shape[0]) if first_raw.ndim == 2 else 1
    return device, samples, split, op, od_channels


def run_train(args) -> None:
    device, samples, split, op, od_channels = prepare(args)
    out = Path(args.exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_files = [
        Path(__file__), Path(__file__).with_name("new_sample_train.py"),
        Path(__file__).parent / "ma_psdun" / "model.py",
        Path(__file__).parent / "ma_psdun" / "core.py",
        Path(__file__).parent / "ma_psdun" / "conditions.py",
        Path(__file__).parent / "ma_psdun" / "labels.py",
        Path(__file__).parent / "ma_psdun" / "noise.py",
    ]
    config = vars(args).copy()
    config.update({
        "device_resolved": str(device),
        "od_channels": od_channels,
        "pattern_sha256": sha256_file(args.patterns),
        "source_sha256": {path.name: sha256_file(path) for path in source_files},
    })
    save_json(out / "config.json", config)
    save_json(out / "split_audit.json", split)
    save_json(out / "label_preprocessing_audit.json", {
        sample["object"]: sample.get("label_audit") for sample in samples
    })

    model = make_model(op, args, od_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_batch = _batch(samples, sample_indices(samples, split["train"]), device, 0, args.condition_mode)
    val_batch = None
    if split["validation"]:
        val_batch = _batch(samples, sample_indices(samples, split["validation"]), device, 0, args.condition_mode)
    views = build_operator_views(op, args.size, args.dihedral_augmentation)
    base_view = views[0]
    best_ssim = -float("inf")
    best_epoch = None
    history = []
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        transform = epoch % 8 if args.dihedral_augmentation else 0
        op.centered_patterns = views[transform]
        target = dihedral(train_batch[2], transform)
        pred, y = model(train_batch[0], train_batch[1], train_batch[3], (args.size, args.size))
        loss = train_loss(pred, target, y, op, args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        op.centered_patterns = base_view

        record = {"epoch": epoch, "train_loss": float(loss.detach()), "transform": transform, "elapsed_s": time.time() - started}
        if val_batch is not None:
            model.eval()
            with torch.no_grad():
                val_pred, val_y = model(val_batch[0], val_batch[1], val_batch[3], (args.size, args.size))
                val_metrics = metrics(val_pred, val_batch[2], val_y, op)
            record.update({f"validation_{key}": value for key, value in val_metrics.items()})
            if val_metrics["ssim"] > best_ssim:
                best_ssim = val_metrics["ssim"]
                best_epoch = epoch
                torch.save({"epoch": epoch, "model": model.state_dict(), "validation": val_metrics}, out / "checkpoint_best.pt")
        history.append(record)
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            print(json.dumps(record), flush=True)

    op.centered_patterns = base_view
    save_json(out / "training_history.json", history)
    if args.mode == "select":
        summary = {
            "status": "selection_completed",
            "best_validation_ssim": best_ssim,
            "best_epoch": best_epoch,
            "test_labels_evaluated": False,
            "ring_blind_supervision": True,
        }
        save_json(out / "selection.json", summary)
        (out / "SELECTION_DONE").write_text("completed\n")
        print(json.dumps(summary), flush=True)
        return

    torch.save({"epoch": args.epochs - 1, "model": model.state_dict()}, out / "checkpoint_final.pt")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reloaded = make_model(op, args, od_channels).to(device)
    checkpoint = torch.load(out / "checkpoint_final.pt", map_location=device, weights_only=False)
    reloaded.load_state_dict(checkpoint["model"])
    primary = evaluate_objects(reloaded, samples, split["test"], device, op, args)
    audit = evaluate_objects(reloaded, samples, split["audit"], device, op, args)
    result = {
        "status": "completed",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_selection": "fixed_epoch_from_ring_blind_validation",
        "checkpoint_sha256": sha256_file(out / "checkpoint_final.pt"),
        "independent_checkpoint_reload": True,
        "ring_blind_supervision": True,
        "primary_test": {"overall": primary["overall"], "by_object": primary["by_object"]},
        "audit_ring_holdout": {"overall": audit["overall"], "by_object": audit["by_object"]} if audit else None,
    }
    save_json(out / "test.json", result)
    torch.save({
        "pred": primary["pred"],
        "target": primary["target"],
        "objects": primary["objects"],
        "audit_pred": audit["pred"] if audit else None,
        "audit_target": audit["target"] if audit else None,
        "audit_objects": audit["objects"] if audit else [],
    }, out / "predictions.pt")
    save_preview(out / "preview_target_pred.png", [pack for pack in (primary, audit) if pack], args.size)
    (out / "DONE").write_text("completed\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


def ensemble(args) -> None:
    members = [Path(value) for value in parse_names(args.members)]
    if len(members) < 2:
        raise ValueError("--members requires at least two final experiment directories")
    packs = [torch.load(path / "predictions.pt", map_location="cpu", weights_only=False) for path in members]
    objects = packs[0]["objects"]
    audit_objects = packs[0]["audit_objects"]
    for pack in packs[1:]:
        if pack["objects"] != objects or pack["audit_objects"] != audit_objects:
            raise ValueError("ensemble member object order differs")
        if not torch.equal(pack["target"], packs[0]["target"]) or not torch.equal(pack["audit_target"], packs[0]["audit_target"]):
            raise ValueError("ensemble member targets differ")
    pred = torch.stack([pack["pred"] for pack in packs]).mean(0)
    target = packs[0]["target"]
    audit_pred = torch.stack([pack["audit_pred"] for pack in packs]).mean(0)
    audit_target = packs[0]["audit_target"]

    def image_only_metrics(a, b):
        from ma_psdun.eval import image_metrics
        return image_metrics(a, b)

    rows = [{"object": name, "metrics": image_only_metrics(pred[i:i + 1], target[i:i + 1])} for i, name in enumerate(objects)]
    audit_rows = [{"object": name, "metrics": image_only_metrics(audit_pred[i:i + 1], audit_target[i:i + 1])} for i, name in enumerate(audit_objects)]
    result = {
        "status": "completed",
        "members": [str(path) for path in members],
        "member_count": len(members),
        "selection": "all predeclared seeds; no test-based member selection",
        "ring_blind_supervision": True,
        "primary_test": {"overall": image_only_metrics(pred, target), "by_object": rows},
        "audit_ring_holdout": {"overall": image_only_metrics(audit_pred, audit_target), "by_object": audit_rows},
    }
    out = Path(args.exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "ensemble.json", result)
    torch.save({"pred": pred, "target": target, "objects": objects, "audit_pred": audit_pred, "audit_target": audit_target, "audit_objects": audit_objects}, out / "predictions.pt")
    save_preview(out / "preview_target_pred.png", [
        {"pred": pred, "target": target},
        {"pred": audit_pred, "target": audit_target},
    ], pred.shape[-1])
    (out / "DONE").write_text("completed\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["select", "final", "ensemble"], default="select")
    parser.add_argument("--data-root", default="/sci_persistent_storage/ma_psdun_v2_20260830/data/sample/sample")
    parser.add_argument("--patterns", default="/sci_persistent_storage/ma_psdun_v2_20260830/data/4096.tif")
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--preprocess", default="detrend")
    parser.add_argument("--detrend-width", type=int, default=31)
    parser.add_argument("--gauss-sigma", type=float, default=40.0)
    parser.add_argument("--psf-sigma", type=float, default=0.0)
    parser.add_argument("--psf-sigma-y", type=float, default=None)
    parser.add_argument("--psf-angle", type=float, default=0.0)
    parser.add_argument(
        "--fusion",
        choices=["mean", "weighted", "quality", "agreement", "median", "od0", "multi"],
        default="mean",
    )
    parser.add_argument("--label-resample", choices=["bilinear", "lanczos"], default="bilinear")
    parser.add_argument("--label-crop", choices=["none", "center", "foreground"], default="center")
    parser.add_argument("--label-contrast", choices=["none", "minmax", "percentile"], default="percentile")
    parser.add_argument("--label-percentile-low", type=float, default=1.0)
    parser.add_argument("--label-percentile-high", type=float, default=99.0)
    parser.add_argument("--condition-mode", choices=["constant", "od", "stats", "stats_weak"], default="constant")
    parser.add_argument("--stages", type=int, default=12)
    parser.add_argument("--gain-init", type=float, default=4.0)
    parser.add_argument("--lowpass-kernel", type=int, default=3)
    parser.add_argument("--prior-residual-scale", type=float, default=0.5)
    parser.add_argument("--shared-prior", action="store_true")
    parser.add_argument("--adaptive-fusion-scale", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--edge-weight", type=float, default=0.0)
    parser.add_argument("--dice-weight", type=float, default=0.0)
    parser.add_argument("--pyramid-weight", type=float, default=0.0)
    parser.add_argument("--binary-weight", type=float, default=0.0)
    parser.add_argument("--dihedral-augmentation", action="store_true")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--train-objects", default="")
    parser.add_argument("--val-objects", default="")
    parser.add_argument("--test-objects", default=DEFAULT_TEST)
    parser.add_argument("--audit-objects", default=DEFAULT_AUDIT)
    parser.add_argument("--forbidden-supervised-objects", default=DEFAULT_FORBIDDEN)
    parser.add_argument("--members", default="")
    args = parser.parse_args()
    if args.mode == "ensemble":
        ensemble(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
