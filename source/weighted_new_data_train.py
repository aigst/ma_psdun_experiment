"""Ring-blind MA-PSDUN training with an explicit weight for new objects.

The 2026-09-03 export adds three labelled objects.  This entry point keeps
the existing ring-blind split contract while allowing those objects to enter
the image loss at a declared fractional weight.  A zero weight is the old
data-only control; the weight is selected by object-disjoint validation only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ma_psdun.model import MAPSDUN
from new_sample_train import _batch, _weighted_loss, load_sample
from ring_blind_train import (
    DEFAULT_AUDIT,
    DEFAULT_FORBIDDEN,
    DEFAULT_TEST,
    build_operator_views,
    dihedral,
    ensemble as ring_ensemble,
    evaluate_objects,
    make_model,
    metrics,
    parse_names,
    prepare,
    save_json,
    save_preview,
    sample_indices,
    sha256_file,
    train_loss,
    validate_split,
)


def _weighted_group_loss(pred, target, y, op, args, old_mask, new_mask):
    """Combine old/new object losses without changing the model forward pass."""
    old_count = int(old_mask.sum())
    new_count = int(new_mask.sum())
    old_loss = None
    if old_count:
        old_loss = train_loss(pred[old_mask], target[old_mask], y[old_mask], op, args)
    if new_count and args.new_object_weight > 0.0:
        new_loss = train_loss(pred[new_mask], target[new_mask], y[new_mask], op, args)
        if old_loss is None:
            return new_loss
        weight = float(args.new_object_weight)
        return (old_loss * old_count + new_loss * new_count * weight) / (old_count + new_count * weight)
    if old_loss is None:
        # A fold must contain at least one old non-ring object, but keeping this
        # branch explicit makes malformed invocations fail loudly.
        raise ValueError("weighted training batch has no positively weighted objects")
    return old_loss


def _object_masks(samples, indices, new_objects, device):
    names = [samples[i]["object"] for i in indices]
    new_set = set(new_objects)
    new_mask = torch.tensor([name in new_set for name in names], device=device, dtype=torch.bool)
    return ~new_mask, new_mask


def run_train(args) -> None:
    device, samples, split, op, od_channels = prepare(args)
    out = Path(args.exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_files = [
        Path(__file__),
        Path(__file__).with_name("ring_blind_train.py"),
        Path(__file__).with_name("new_sample_train.py"),
        Path(__file__).parent / "ma_psdun" / "model.py",
        Path(__file__).parent / "ma_psdun" / "core.py",
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

    model = make_model(op, args, od_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_indices = sample_indices(samples, split["train"])
    val_indices = sample_indices(samples, split["validation"])
    train_batch = _batch(samples, train_indices, device, 0, args.condition_mode)
    val_batch = _batch(samples, val_indices, device, 0, args.condition_mode) if val_indices else None
    old_mask, new_mask = _object_masks(samples, train_indices, parse_names(args.new_objects), device)
    views = build_operator_views(op, args.size, args.dihedral_augmentation)
    base_view = views[0]
    best_ssim = -float("inf")
    best_epoch = None
    history = []

    for epoch in range(args.epochs):
        model.train()
        transform = epoch % 8 if args.dihedral_augmentation else 0
        op.centered_patterns = views[transform]
        target = dihedral(train_batch[2], transform)
        pred, y = model(train_batch[0], train_batch[1], train_batch[3], (args.size, args.size))
        loss = _weighted_group_loss(pred, target, y, op, args, old_mask, new_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        op.centered_patterns = base_view

        record = {"epoch": epoch, "train_loss": float(loss.detach()), "transform": transform}
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
        result = {
            "status": "selection_completed",
            "new_object_weight": float(args.new_object_weight),
            "best_validation_ssim": best_ssim,
            "best_epoch": best_epoch,
            "test_labels_evaluated": False,
            "ring_blind_supervision": True,
        }
        save_json(out / "selection.json", result)
        (out / "SELECTION_DONE").write_text("completed\n")
        print(json.dumps(result), flush=True)
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
        "new_object_weight": float(args.new_object_weight),
        "checkpoint_epoch": int(checkpoint["epoch"]),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["select", "final", "ensemble"], default="select")
    parser.add_argument("--data-root", default="/sci_persistent_storage/ma_psdun_v2_20260903/data/sample")
    parser.add_argument("--patterns", default="/sci_persistent_storage/ma_psdun_v2_20260903/data/4096.tif")
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--preprocess", default="detrend")
    parser.add_argument("--detrend-width", type=int, default=31)
    parser.add_argument("--gauss-sigma", type=float, default=40.0)
    parser.add_argument("--fusion", choices=["mean", "weighted", "quality", "agreement", "median", "od0", "multi"], default="mean")
    parser.add_argument("--label-resample", choices=["bilinear", "lanczos"], default="bilinear")
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
    parser.add_argument("--dihedral-augmentation", action="store_true")
    parser.add_argument("--seed", type=int, default=20261100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--train-objects", default="")
    parser.add_argument("--val-objects", default="")
    parser.add_argument("--test-objects", default=DEFAULT_TEST)
    parser.add_argument("--audit-objects", default=DEFAULT_AUDIT)
    parser.add_argument("--forbidden-supervised-objects", default=DEFAULT_FORBIDDEN)
    parser.add_argument("--new-objects", default="obj22-20260902,obj24-20260903,obj25-20260903")
    parser.add_argument("--new-object-weight", type=float, default=1.0)
    parser.add_argument("--members", default="")
    args = parser.parse_args()
    if not 0.0 <= args.new_object_weight <= 1.0:
        parser.error("--new-object-weight must be in [0, 1]")
    if args.mode == "ensemble":
        # Reuse the audited ring-blind ensemble implementation; prediction
        # packs have the same schema and are checked for target/order equality.
        ring_ensemble(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
