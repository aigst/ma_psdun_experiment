"""Select a target-free agreement-gated old/weak-stat model policy."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from ma_psdun.eval import image_metrics
from ring_blind_calibrate import apply_calibration, candidates
from ring_blind_train import evaluate_objects, make_model, prepare


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def agreement_score(root: Path, object_name: str) -> float:
    captures = []
    for od in sorted((root / object_name).glob("OD*")):
        values = np.loadtxt(od / "traindata.txt", dtype=np.float32)
        signal = values[1::2] - values[0::2]
        signal = (signal - signal.mean()) / max(float(signal.std()), 1e-8)
        captures.append(signal)
    if len(captures) < 2:
        return 1.0
    corr = np.corrcoef(np.stack(captures))
    n = len(captures)
    return float((corr.sum() - n) / (n * (n - 1)))


def load_selection(directory: Path, object_names: list[str]):
    args = Namespace(**json.loads((directory / "config.json").read_text()))
    device, samples, split, op, od_channels = prepare(args)
    model = make_model(op, args, od_channels).to(device)
    checkpoint = torch.load(directory / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return evaluate_objects(model, samples, object_names, device, op, args), split


def metric_pack(pred, target, objects):
    return {
        "overall": image_metrics(pred, target),
        "by_object": [
            {"object": name, "metrics": image_metrics(pred[i:i + 1], target[i:i + 1])}
            for i, name in enumerate(objects)
        ],
    }


def main() -> None:
    root = Path("/sci_persistent_storage/ma_psdun_v2_20260903")
    data_root = root / "data" / "sample"
    old_selection = root / "results_blend" / "selection_old"
    weak_selection = root / "results_stats_weak_condition_cv_20260904" / "selection"
    old_final = torch.load(root / "results" / "final" / "ensemble" / "predictions.pt", map_location="cpu", weights_only=False)
    weak_final = torch.load(weak_selection.parent / "final" / "ensemble" / "predictions.pt", map_location="cpu", weights_only=False)
    if old_final["objects"] != weak_final["objects"] or old_final["audit_objects"] != weak_final["audit_objects"]:
        raise RuntimeError("final prediction order differs")

    old_parts, weak_parts, targets, objects, agreements = [], [], [], [], []
    seen = set()
    for fold in range(1, 5):
        old_dir = old_selection / f"fold{fold}"
        weak_dir = weak_selection / f"fold{fold}" / "stats"
        old_cfg = json.loads((old_dir / "config.json").read_text())
        val_objects = [x for x in old_cfg["val_objects"].split(",") if x]
        old_pack, old_split = load_selection(old_dir, val_objects)
        weak_pack, weak_split = load_selection(weak_dir, val_objects)
        if old_split["validation"] != val_objects or set(val_objects) - set(weak_split["validation"]):
            raise RuntimeError(f"unexpected fold split {fold}")
        if seen & set(val_objects):
            raise RuntimeError(f"validation overlap in fold {fold}")
        seen.update(val_objects)
        old_parts.append(old_pack["pred"])
        weak_parts.append(weak_pack["pred"])
        targets.append(old_pack["target"])
        objects.extend(val_objects)
        agreements.extend(agreement_score(data_root, name) for name in val_objects)

    old_val, weak_val, target = torch.cat(old_parts), torch.cat(weak_parts), torch.cat(targets)
    thresholds = [-1.0] + [round(x, 2) for x in np.arange(-0.2, 0.81, 0.05)] + [1.0]
    rows = []
    for direction in ("weak_if_high", "weak_if_low"):
        for threshold in thresholds:
            use_weak = torch.tensor(
                [(score >= threshold) if direction == "weak_if_high" else (score <= threshold) for score in agreements],
                dtype=torch.bool,
            )
            raw = torch.where(use_weak[:, None, None, None], weak_val, old_val)
            for spec in candidates():
                score = image_metrics(apply_calibration(raw, spec), target)
                rows.append({"direction": direction, "threshold": threshold, "weak_count": int(use_weak.sum()), "calibration": spec, "validation": score})
    rows.sort(key=lambda row: row["validation"]["ssim"], reverse=True)
    selected = rows[0]
    test_agreements = [agreement_score(data_root, name) for name in old_final["objects"]]
    audit_agreements = [agreement_score(data_root, name) for name in old_final["audit_objects"]]

    def apply_policy(old, weak, scores):
        if selected["direction"] == "weak_if_high":
            use = [x >= selected["threshold"] for x in scores]
        else:
            use = [x <= selected["threshold"] for x in scores]
        mask = torch.tensor(use, dtype=torch.bool)
        return torch.where(mask[:, None, None, None], weak, old), use

    raw_test, test_use = apply_policy(old_final["pred"], weak_final["pred"], test_agreements)
    raw_audit, audit_use = apply_policy(old_final["audit_pred"], weak_final["audit_pred"], audit_agreements)
    pred, audit_pred = apply_calibration(raw_test, selected["calibration"]), apply_calibration(raw_audit, selected["calibration"])
    result = {
        "status": "completed",
        "protocol": "four-fold object-disjoint validation on 14 old non-ring objects",
        "selection_objects": objects,
        "test_used_for_selection": False,
        "ring_blind_selection": True,
        "selected": selected,
        "validation_agreement_scores": dict(zip(objects, agreements)),
        "test_agreement_scores": dict(zip(old_final["objects"], test_agreements)),
        "audit_agreement_scores": dict(zip(old_final["audit_objects"], audit_agreements)),
        "test_weak_selected": dict(zip(old_final["objects"], test_use)),
        "audit_weak_selected": dict(zip(old_final["audit_objects"], audit_use)),
        "top_validation_candidates": rows[:20],
        "primary_test": metric_pack(pred, old_final["target"], old_final["objects"]),
        "audit_ring_holdout": metric_pack(audit_pred, old_final["audit_target"], old_final["audit_objects"]),
    }
    out = root / "results_agreement_policy_20260904"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=json_default) + "\n")
    torch.save({"pred": pred, "target": old_final["target"], "objects": old_final["objects"], "audit_pred": audit_pred, "audit_target": old_final["audit_target"], "audit_objects": old_final["audit_objects"], "policy": selected}, out / "predictions.pt")
    (out / "DONE").write_text("completed\n")
    print(json.dumps(result, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
