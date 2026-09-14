"""Select a ring-blind blend of old strict and stats-conditioned ensembles."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import torch

from ma_psdun.eval import image_metrics
from ring_blind_calibrate import apply_calibration, candidates
from ring_blind_train import evaluate_objects, make_model, prepare


def _metric_pack(pred, target, objects):
    return {
        "overall": image_metrics(pred, target),
        "by_object": [
            {"object": name, "metrics": image_metrics(pred[i:i + 1], target[i:i + 1])}
            for i, name in enumerate(objects)
        ],
    }


def _load_selection(directory: Path, object_names: list[str]):
    config = json.loads((directory / "config.json").read_text())
    args = Namespace(**config)
    device, samples, split, op, od_channels = prepare(args)
    model = make_model(op, args, od_channels).to(device)
    checkpoint = torch.load(directory / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    pack = evaluate_objects(model, samples, object_names, device, op, args)
    return pack, split


def main() -> None:
    root = Path("/sci_persistent_storage/ma_psdun_v2_20260903")
    old_root = root / "results_blend" / "selection_old"
    stats_root = root / "results_stats_condition_cv_20260904" / "selection"
    old_final = torch.load(root / "results" / "final" / "ensemble" / "predictions.pt", map_location="cpu", weights_only=False)
    stats_final = torch.load(stats_root.parent / "final" / "ensemble" / "predictions.pt", map_location="cpu", weights_only=False)
    if old_final["objects"] != stats_final["objects"] or old_final["audit_objects"] != stats_final["audit_objects"]:
        raise RuntimeError("old and stats final prediction order differs")

    val_old_parts, val_stats_parts, val_targets, selection_objects = [], [], [], []
    seen = set()
    for fold in range(1, 5):
        old_dir = old_root / f"fold{fold}"
        stats_dir = stats_root / f"fold{fold}" / "stats"
        old_config = json.loads((old_dir / "config.json").read_text())
        old_val = [x for x in old_config["val_objects"].split(",") if x]
        old_pack, old_split = _load_selection(old_dir, old_val)
        stats_pack, stats_split = _load_selection(stats_dir, old_val)
        if old_split["validation"] != old_val:
            raise RuntimeError(f"unexpected old validation split in fold {fold}")
        if set(old_val) - set(stats_split["validation"]):
            raise RuntimeError(f"old validation is not contained in stats split for fold {fold}")
        overlap = seen & set(old_val)
        if overlap:
            raise RuntimeError(f"validation object repeated: {sorted(overlap)}")
        seen.update(old_val)
        val_old_parts.append(old_pack["pred"])
        val_stats_parts.append(stats_pack["pred"])
        val_targets.append(old_pack["target"])
        selection_objects.extend(old_val)

    old_val = torch.cat(val_old_parts)
    stats_val = torch.cat(val_stats_parts)
    val_target = torch.cat(val_targets)
    rows = []
    for new_weight_i in range(0, 21):
        new_weight = new_weight_i / 20.0
        raw = (1.0 - new_weight) * old_val + new_weight * stats_val
        for spec in candidates():
            calibrated = apply_calibration(raw, spec)
            score = image_metrics(calibrated, val_target)
            rows.append({"old_weight": 1.0 - new_weight, "new_weight": new_weight, "calibration": spec, "validation": score})
    rows.sort(key=lambda row: row["validation"]["ssim"], reverse=True)
    selected = rows[0]
    old_only = max((row for row in rows if row["new_weight"] == 0.0), key=lambda row: row["validation"]["ssim"])
    stats_only = max((row for row in rows if row["new_weight"] == 1.0), key=lambda row: row["validation"]["ssim"])

    raw_test = (selected["old_weight"] * old_final["pred"] + selected["new_weight"] * stats_final["pred"])
    raw_audit = (selected["old_weight"] * old_final["audit_pred"] + selected["new_weight"] * stats_final["audit_pred"])
    pred = apply_calibration(raw_test, selected["calibration"])
    audit_pred = apply_calibration(raw_audit, selected["calibration"])
    result = {
        "status": "completed",
        "protocol": "four-fold object-disjoint validation on 14 old non-ring objects",
        "selection_objects": selection_objects,
        "test_used_for_selection": False,
        "ring_blind_selection": True,
        "selected": selected,
        "old_only_validation_selected": old_only,
        "old_only_test": _metric_pack(
            apply_calibration(old_final["pred"], old_only["calibration"]),
            old_final["target"], old_final["objects"],
        ),
        "stats_only_validation_selected": stats_only,
        "top_validation_candidates": rows[:20],
        "validation_old_raw": image_metrics(old_val, val_target),
        "validation_stats_raw": image_metrics(stats_val, val_target),
        "primary_test": _metric_pack(pred, old_final["target"], old_final["objects"]),
        "audit_ring_holdout": _metric_pack(audit_pred, old_final["audit_target"], old_final["audit_objects"]),
    }
    out = root / "results_stats_old_blend_20260904"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    torch.save({
        "pred": pred,
        "target": old_final["target"],
        "objects": old_final["objects"],
        "audit_pred": audit_pred,
        "audit_target": old_final["audit_target"],
        "audit_objects": old_final["audit_objects"],
        "old_weight": selected["old_weight"],
        "new_weight": selected["new_weight"],
        "calibration": selected["calibration"],
    }, out / "predictions.pt")
    (out / "DONE").write_text("completed\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
