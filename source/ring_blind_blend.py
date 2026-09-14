"""Select an old/new model blend using object-disjoint validation folds only."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch
from PIL import Image

from ma_psdun.eval import image_metrics
from ring_blind_calibrate import apply_calibration, candidates, metric_pack
from ring_blind_train import evaluate_objects, make_model, prepare, save_json, save_preview


def parse_paths(value: str) -> list[Path]:
    return [Path(item) for item in value.split(",") if item]


def blend_predictions(old: torch.Tensor, new: torch.Tensor, old_weight: float) -> torch.Tensor:
    if old.shape != new.shape:
        raise ValueError(f"prediction shape mismatch: old={tuple(old.shape)} new={tuple(new.shape)}")
    if not 0.0 <= old_weight <= 1.0:
        raise ValueError(f"old_weight must be in [0, 1], got {old_weight}")
    return old * old_weight + new * (1.0 - old_weight)


def require_matching_pack(old: dict, new: dict) -> None:
    for key in ("objects", "audit_objects"):
        if old[key] != new[key]:
            raise ValueError(f"{key} differ between old and new prediction packs")
    for key in ("target", "audit_target"):
        if old[key] is None or new[key] is None or not torch.equal(old[key], new[key]):
            raise ValueError(f"{key} differ between old and new prediction packs")


def select_candidate(old_pred: torch.Tensor, new_pred: torch.Tensor, target: torch.Tensor):
    rows = []
    for old_weight_int in range(11):
        old_weight = old_weight_int / 10.0
        raw = blend_predictions(old_pred, new_pred, old_weight)
        for spec in candidates():
            score = image_metrics(apply_calibration(raw, spec), target)
            rows.append({
                "old_weight": old_weight,
                "new_weight": 1.0 - old_weight,
                "calibration": spec,
                "validation": score,
            })
    rows.sort(key=lambda row: row["validation"]["ssim"], reverse=True)
    return rows[0], rows


def evaluate_checkpoint(selection_dir: Path, names: list[str]):
    config = json.loads((selection_dir / "config.json").read_text())
    # Configs written before the optional optical PSF wrapper was introduced
    # have no field for it.  They are the frozen no-PSF canonical family, so
    # make that compatibility default explicit when replaying their OOF
    # predictions.
    config.setdefault("psf_sigma", 0.0)
    args = Namespace(**config)
    device, samples, split, op, od_channels = prepare(args)
    if not split["ring_blind_supervision"] or any(split["forbidden_intersection"].values()):
        raise RuntimeError(f"ring-blind split audit failed: {selection_dir}")
    model = make_model(op, args, od_channels).to(device)
    checkpoint = torch.load(selection_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    evaluation = evaluate_objects(model, samples, names, device, op, args)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return evaluation, split, int(checkpoint["epoch"])


def collect_cross_validation(old_dirs: list[Path], new_dirs: list[Path]):
    if len(old_dirs) != 4 or len(new_dirs) != 4:
        raise ValueError("exactly four old and four new selection directories are required")

    old_predictions = []
    new_predictions = []
    targets = []
    fold_records = []
    validation_seen = set()
    old_universe = None
    forbidden = None
    test_objects = None
    audit_objects = None

    for fold, (old_dir, new_dir) in enumerate(zip(old_dirs, new_dirs), start=1):
        old_config = json.loads((old_dir / "config.json").read_text())
        old_names = [item for item in old_config["val_objects"].split(",") if item]
        old_eval, old_split, old_epoch = evaluate_checkpoint(old_dir, old_names)
        new_eval, new_split, new_epoch = evaluate_checkpoint(new_dir, old_names)

        overlap = validation_seen & set(old_names)
        if overlap:
            raise RuntimeError(f"old validation objects repeated across folds: {sorted(overlap)}")
        validation_seen.update(old_names)
        if not set(old_names).issubset(new_split["validation"]):
            raise RuntimeError(f"old fold {fold} is not a subset of the paired new validation fold")
        if not torch.equal(old_eval["target"], new_eval["target"]):
            raise RuntimeError(f"old/new validation targets differ in fold {fold}")

        current_universe = set(old_split["train"]) | set(old_split["validation"])
        if old_universe is None:
            old_universe = current_universe
        elif current_universe != old_universe:
            raise RuntimeError("old cross-validation folds do not share one object universe")
        if set(old_split["train"]) != current_universe - set(old_names):
            raise RuntimeError(f"old fold {fold} train set is not the validation complement")
        current_forbidden = set(old_split["forbidden_supervised_objects"])
        if forbidden is None:
            forbidden = current_forbidden
        elif current_forbidden != forbidden:
            raise RuntimeError("forbidden-object sets differ across old folds")
        if test_objects is not None and set(old_split["test"]) != set(test_objects):
            raise RuntimeError("test objects differ across old folds")
        if audit_objects is not None and set(old_split["audit"]) != set(audit_objects):
            raise RuntimeError("audit objects differ across old folds")
        for current, expected, label in (
            (set(new_split["forbidden_supervised_objects"]), forbidden, "forbidden"),
            (new_split["test"], test_objects, "test"),
            (new_split["audit"], audit_objects, "audit"),
        ):
            if expected is not None and set(current) != set(expected):
                raise RuntimeError(f"{label} objects differ across model families or folds")
        test_objects = old_split["test"] if test_objects is None else test_objects
        audit_objects = old_split["audit"] if audit_objects is None else audit_objects
        if set(new_split["test"]) != set(test_objects) or set(new_split["audit"]) != set(audit_objects):
            raise RuntimeError("test or audit objects differ between model families")

        old_predictions.append(old_eval["pred"])
        new_predictions.append(new_eval["pred"])
        targets.append(old_eval["target"])
        fold_records.append({
            "fold": fold,
            "old_directory": str(old_dir),
            "new_directory": str(new_dir),
            "objects": old_names,
            "old_checkpoint_epoch": old_epoch,
            "new_checkpoint_epoch": new_epoch,
            "old_raw": old_eval["overall"],
            "new_raw": new_eval["overall"],
        })

    if validation_seen != old_universe:
        missing = sorted((old_universe or set()) - validation_seen)
        extra = sorted(validation_seen - (old_universe or set()))
        raise RuntimeError(f"four-fold validation does not cover the old universe once: missing={missing} extra={extra}")
    if validation_seen & set(test_objects or []) or validation_seen & set(audit_objects or []):
        raise RuntimeError("test or audit objects leaked into blend selection")

    return {
        "old_pred": torch.cat(old_predictions),
        "new_pred": torch.cat(new_predictions),
        "target": torch.cat(targets),
        "objects": [name for row in fold_records for name in row["objects"]],
        "folds": fold_records,
        "forbidden": sorted(forbidden or []),
        "test_objects": test_objects,
        "audit_objects": audit_objects,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-selection-dirs", required=True)
    parser.add_argument("--new-selection-dirs", required=True)
    parser.add_argument("--old-ensemble-predictions", required=True)
    parser.add_argument("--new-ensemble-predictions", required=True)
    parser.add_argument("--exp-dir", required=True)
    args = parser.parse_args()

    validation = collect_cross_validation(
        parse_paths(args.old_selection_dirs),
        parse_paths(args.new_selection_dirs),
    )
    selected, rows = select_candidate(
        validation["old_pred"], validation["new_pred"], validation["target"],
    )

    old_pack = torch.load(args.old_ensemble_predictions, map_location="cpu", weights_only=False)
    new_pack = torch.load(args.new_ensemble_predictions, map_location="cpu", weights_only=False)
    require_matching_pack(old_pack, new_pack)
    if old_pack["objects"] != validation["test_objects"] or old_pack["audit_objects"] != validation["audit_objects"]:
        raise RuntimeError("final prediction objects do not match the audited cross-validation splits")

    raw_pred = blend_predictions(old_pack["pred"], new_pack["pred"], selected["old_weight"])
    raw_audit_pred = blend_predictions(old_pack["audit_pred"], new_pack["audit_pred"], selected["old_weight"])
    pred = apply_calibration(raw_pred, selected["calibration"])
    audit_pred = apply_calibration(raw_audit_pred, selected["calibration"])
    result = {
        "status": "completed",
        "protocol": "object-disjoint four-fold validation on 14 old eligible objects",
        "selection_objects": validation["objects"],
        "folds": validation["folds"],
        "forbidden_supervised_objects": validation["forbidden"],
        "object_level_disjoint": True,
        "test_used_for_selection": False,
        "selected": selected,
        "validation_family_metrics": {
            "old_raw": image_metrics(validation["old_pred"], validation["target"]),
            "new_raw": image_metrics(validation["new_pred"], validation["target"]),
        },
        "top_validation_candidates": rows[:20],
        "primary_test": metric_pack(pred, old_pack["target"], old_pack["objects"]),
        "audit_ring_holdout": metric_pack(audit_pred, old_pack["audit_target"], old_pack["audit_objects"]),
    }

    out = Path(args.exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "blend.json", result)
    torch.save({
        "pred": pred,
        "target": old_pack["target"],
        "objects": old_pack["objects"],
        "audit_pred": audit_pred,
        "audit_target": old_pack["audit_target"],
        "audit_objects": old_pack["audit_objects"],
        "old_weight": selected["old_weight"],
        "new_weight": selected["new_weight"],
        "calibration": selected["calibration"],
    }, out / "predictions.pt")
    preview = out / "preview_target_pred.png"
    save_preview(preview, [
        {"pred": pred, "target": old_pack["target"]},
        {"pred": audit_pred, "target": old_pack["audit_target"]},
    ], pred.shape[-1])
    with Image.open(preview) as image:
        image.resize((image.width * 4, image.height * 4), Image.Resampling.NEAREST).save(
            out / "preview_target_pred_4x.png"
        )
    (out / "DONE").write_text("completed\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
