"""Validation-only selection and final evaluation for refined model blending."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ring_blind_blend import blend_predictions, require_matching_pack, select_candidate
from ring_blind_calibrate import apply_calibration, metric_pack
from ring_blind_train import save_json, save_preview


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select(args) -> None:
    old_path = Path(args.old_oof)
    new_path = Path(args.new_oof)
    refinement_path = Path(args.refinement_selection)
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    new = torch.load(new_path, map_location="cpu", weights_only=False)
    refinement = json.loads(refinement_path.read_text())
    refined = torch.load(
        refinement_path.parent / "selection_predictions.pt",
        map_location="cpu",
        weights_only=False,
    )
    if not old.get("object_level_disjoint") or not new.get("object_level_disjoint"):
        raise RuntimeError("input OOF packs are not object-disjoint")
    if old.get("test_labels_evaluated") or new.get("test_labels_evaluated"):
        raise RuntimeError("test labels were evaluated in an OOF input")
    selected_name = refinement["selected"]["name"]
    new_index = {name: index for index, name in enumerate(new["objects"])}
    positions = [new_index[name] for name in old["objects"]]
    if not torch.equal(old["target"], new["target"][positions]):
        raise RuntimeError("old/new OOF targets differ after object alignment")
    selected, rows = select_candidate(
        old["pred"], refined["predictions"][selected_name][positions], old["target"],
    )
    result = {
        "status": "selection_completed",
        "protocol": "common-object OOF blend selection; no test labels read",
        "selected_refiner": refinement["selected"],
        "selected_blend": selected,
        "top_candidates": rows[:20],
        "objects": old["objects"],
        "forbidden_supervised_objects": old["forbidden_supervised_objects"],
        "object_level_disjoint": True,
        "test_labels_evaluated": False,
        "artifact_sha256": {
            "old_oof": sha256(old_path),
            "new_oof": sha256(new_path),
            "refinement_selection": sha256(refinement_path),
        },
    }
    out = Path(args.exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "selection.json", result)
    print(json.dumps(result, indent=2))


def final(args) -> None:
    selection_path = Path(args.selection_json)
    selection = json.loads(selection_path.read_text())
    if not selection.get("object_level_disjoint") or selection.get("test_labels_evaluated"):
        raise RuntimeError("blend selection did not satisfy the strict contract")
    old_path = Path(args.old_final)
    new_path = Path(args.new_final)
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    new = torch.load(new_path, map_location="cpu", weights_only=False)
    require_matching_pack(old, new)
    spec = selection["selected_blend"]
    pred = apply_calibration(
        blend_predictions(old["pred"], new["pred"], spec["old_weight"]),
        spec["calibration"],
    )
    audit_pred = apply_calibration(
        blend_predictions(old["audit_pred"], new["audit_pred"], spec["old_weight"]),
        spec["calibration"],
    )
    result = {
        "status": "completed",
        "selected_refiner": selection["selected_refiner"],
        "selected_blend": spec,
        "selection_json_sha256": sha256(selection_path),
        "input_sha256": {"old": sha256(old_path), "new": sha256(new_path)},
        "object_level_disjoint": True,
        "test_used_for_selection": False,
        "primary_test": metric_pack(pred, old["target"], old["objects"]),
        "audit_ring_holdout": metric_pack(audit_pred, old["audit_target"], old["audit_objects"]),
    }
    out = Path(args.exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "test.json", result)
    torch.save(
        {
            "pred": pred,
            "target": old["target"],
            "objects": old["objects"],
            "audit_pred": audit_pred,
            "audit_target": old["audit_target"],
            "audit_objects": old["audit_objects"],
            "selected_blend": spec,
        },
        out / "predictions.pt",
    )
    save_preview(out / "preview_target_pred.png", [
        {"pred": pred, "target": old["target"]},
        {"pred": audit_pred, "target": old["audit_target"]},
    ], pred.shape[-1])
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["select", "final"], required=True)
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--old-oof")
    parser.add_argument("--new-oof")
    parser.add_argument("--refinement-selection")
    parser.add_argument("--selection-json")
    parser.add_argument("--old-final")
    parser.add_argument("--new-final")
    args = parser.parse_args()
    if args.mode == "select":
        if not args.old_oof or not args.new_oof or not args.refinement_selection:
            parser.error("select mode requires old/new OOF and refinement selection")
        select(args)
    else:
        if not args.selection_json or not args.old_final or not args.new_final:
            parser.error("final mode requires selection JSON and old/new final packs")
        final(args)


if __name__ == "__main__":
    main()
