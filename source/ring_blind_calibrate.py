"""Select a fixed output calibration on non-ring validation objects only."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

from ma_psdun.eval import image_metrics
from ring_blind_train import evaluate_objects, make_model, prepare, save_json, save_preview


def candidates():
    yield {"kind": "identity"}
    for gamma in (0.70, 0.80, 0.90, 1.10, 1.20, 1.35, 1.50):
        yield {"kind": "gamma", "gamma": gamma}
    for low in (0.00, 0.05, 0.10, 0.15):
        for high in (0.85, 0.90, 0.95, 1.00):
            if high > low:
                yield {"kind": "contrast", "low": low, "high": high}
    for threshold in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        for temperature in (0.05, 0.08, 0.12, 0.18):
            yield {"kind": "sigmoid", "threshold": threshold, "temperature": temperature}
    for amount in (0.25, 0.50, 0.75, 1.00):
        yield {"kind": "unsharp", "amount": amount}


def apply_calibration(pred: torch.Tensor, spec: dict) -> torch.Tensor:
    kind = spec["kind"]
    if kind == "identity":
        return pred
    if kind == "gamma":
        return pred.clamp(0, 1).pow(spec["gamma"])
    if kind == "contrast":
        return ((pred - spec["low"]) / (spec["high"] - spec["low"])).clamp(0, 1)
    if kind == "sigmoid":
        return torch.sigmoid((pred - spec["threshold"]) / spec["temperature"])
    if kind == "unsharp":
        blur = F.avg_pool2d(pred, 3, stride=1, padding=1)
        return (pred + spec["amount"] * (pred - blur)).clamp(0, 1)
    raise ValueError(f"unknown calibration kind: {kind}")


def metric_pack(pred, target, objects):
    return {
        "overall": image_metrics(pred, target),
        "by_object": [
            {"object": name, "metrics": image_metrics(pred[index:index + 1], target[index:index + 1])}
            for index, name in enumerate(objects)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    selection_group = parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument("--selection-dir")
    selection_group.add_argument(
        "--selection-dirs",
        help="comma-separated selection directories with disjoint validation objects",
    )
    parser.add_argument("--ensemble-predictions", required=True)
    parser.add_argument("--exp-dir", required=True)
    args = parser.parse_args()

    selection_dirs = [Path(value) for value in (args.selection_dirs or args.selection_dir).split(",") if value]
    validation_packs = []
    selection_checkpoints = []
    seen_objects = set()
    forbidden_objects = None
    for selection_dir in selection_dirs:
        config = json.loads((selection_dir / "config.json").read_text())
        selection_args = Namespace(**config)
        device, samples, split, op, od_channels = prepare(selection_args)
        if not split["ring_blind_supervision"] or split["forbidden_intersection"] != {"train": [], "validation": []}:
            raise RuntimeError("selection split did not pass ring-blind audit")
        overlap = seen_objects & set(split["validation"])
        if overlap:
            raise RuntimeError(f"validation objects repeated across selection folds: {sorted(overlap)}")
        seen_objects.update(split["validation"])
        current_forbidden = set(split["forbidden_supervised_objects"])
        if forbidden_objects is None:
            forbidden_objects = current_forbidden
        elif current_forbidden != forbidden_objects:
            raise RuntimeError("selection folds use different forbidden-object sets")
        model = make_model(op, selection_args, od_channels).to(device)
        checkpoint = torch.load(selection_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        validation = evaluate_objects(model, samples, split["validation"], device, op, selection_args)
        validation_packs.append(validation)
        selection_checkpoints.append({
            "directory": str(selection_dir),
            "epoch": int(checkpoint["epoch"]),
            "objects": split["validation"],
        })
    val_pred = torch.cat([pack["pred"] for pack in validation_packs])
    val_target = torch.cat([pack["target"] for pack in validation_packs])
    selection_objects = [name for pack in validation_packs for name in pack["objects"]]

    rows = []
    for spec in candidates():
        score = image_metrics(apply_calibration(val_pred, spec), val_target)
        rows.append({"spec": spec, "validation": score})
    rows.sort(key=lambda row: row["validation"]["ssim"], reverse=True)
    selected = rows[0]

    pack = torch.load(args.ensemble_predictions, map_location="cpu", weights_only=False)
    pred = apply_calibration(pack["pred"], selected["spec"])
    audit_pred = apply_calibration(pack["audit_pred"], selected["spec"])
    result = {
        "status": "completed",
        "selection_checkpoints": selection_checkpoints,
        "selection_objects": selection_objects,
        "forbidden_supervised_objects": sorted(forbidden_objects or []),
        "ring_blind_selection": True,
        "selected": selected,
        "top_validation_candidates": rows[:10],
        "primary_test": metric_pack(pred, pack["target"], pack["objects"]),
        "audit_ring_holdout": metric_pack(audit_pred, pack["audit_target"], pack["audit_objects"]),
    }
    out = Path(args.exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "calibration.json", result)
    torch.save({**pack, "pred": pred, "audit_pred": audit_pred, "calibration": selected["spec"]}, out / "predictions.pt")
    save_preview(out / "preview_target_pred.png", [
        {"pred": pred, "target": pack["target"]},
        {"pred": audit_pred, "target": pack["audit_target"]},
    ], pred.shape[-1])
    (out / "DONE").write_text("completed\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
