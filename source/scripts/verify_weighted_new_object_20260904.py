"""Independently verify the weighted new-object MA-PSDUN result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import torch


EXPECTED_TEST = ["obj18-20260901", "obj19-20260901", "obj20-20260901"]
EXPECTED_AUDIT = ["obj15-20260901", "obj16-20260901"]
FORBIDDEN = set(EXPECTED_AUDIT + ["obj19-20260901"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(pack: dict, image_metrics) -> dict:
    return {
        "primary_test": {
            "overall": image_metrics(pack["pred"], pack["target"]),
            "by_object": [
                {
                    "object": name,
                    "metrics": image_metrics(pack["pred"][index:index + 1], pack["target"][index:index + 1]),
                }
                for index, name in enumerate(pack["objects"])
            ],
        },
        "audit_ring_holdout": {
            "overall": image_metrics(pack["audit_pred"], pack["audit_target"]),
            "by_object": [
                {
                    "object": name,
                    "metrics": image_metrics(
                        pack["audit_pred"][index:index + 1],
                        pack["audit_target"][index:index + 1],
                    ),
                }
                for index, name in enumerate(pack["audit_objects"])
            ],
        },
    }


def assert_metrics(actual: dict, expected: dict, tolerance: float) -> None:
    for group in ("primary_test", "audit_ring_holdout"):
        for key in ("mse", "psnr", "ssim"):
            delta = abs(actual[group]["overall"][key] - expected[group]["overall"][key])
            if delta > tolerance:
                raise RuntimeError(f"metric mismatch for {group}.{key}: {delta}")
        actual_rows = {row["object"]: row["metrics"] for row in actual[group]["by_object"]}
        expected_rows = {row["object"]: row["metrics"] for row in expected[group]["by_object"]}
        if actual_rows.keys() != expected_rows.keys():
            raise RuntimeError(f"object mismatch for {group}")
        for name in actual_rows:
            for key in ("mse", "psnr", "ssim"):
                delta = abs(actual_rows[name][key] - expected_rows[name][key])
                if delta > tolerance:
                    raise RuntimeError(f"metric mismatch for {group}.{name}.{key}: {delta}")


def verify_split(path: Path, *, selection: bool) -> None:
    split = json.loads(path.read_text())
    groups = {name: set(split[name]) for name in ("train", "validation", "test", "audit")}
    for left, right in combinations(groups, 2):
        overlap = groups[left] & groups[right]
        if overlap:
            raise RuntimeError(f"split overlap in {path}: {left}:{right}={sorted(overlap)}")
    if groups["test"] != set(EXPECTED_TEST) or groups["audit"] != set(EXPECTED_AUDIT):
        raise RuntimeError(f"unexpected test/audit objects in {path}")
    if FORBIDDEN & (groups["train"] | groups["validation"]):
        raise RuntimeError(f"forbidden supervision in {path}")
    if not split.get("object_level_disjoint") or not split.get("ring_blind_supervision"):
        raise RuntimeError(f"missing split audit flags in {path}")
    if selection:
        selection_result = json.loads((path.parent / "selection.json").read_text())
        if selection_result.get("test_labels_evaluated") is not False:
            raise RuntimeError(f"selection accessed test labels in {path.parent}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--reference-predictions", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--tolerance", type=float, default=1e-7)
    args = parser.parse_args()

    sys.path.insert(0, args.source_root)
    from ma_psdun.eval import image_metrics

    root = Path(args.artifact_root)
    summary = json.loads((root / "weighted_final_summary.json").read_text())
    seed_dirs = sorted(path for path in (root / "final").glob("seed*") if path.is_dir())
    if [path.name for path in seed_dirs] != [
        "seed20261171",
        "seed20261172",
        "seed20261173",
        "seed20261174",
    ]:
        raise RuntimeError(f"unexpected seed set: {[path.name for path in seed_dirs]}")

    seed_packs = [torch.load(path / "predictions.pt", map_location="cpu", weights_only=False) for path in seed_dirs]
    ensemble_path = root / "final" / "ensemble" / "predictions.pt"
    calibrated_path = root / "final" / "calibrated" / "predictions.pt"
    ensemble = torch.load(ensemble_path, map_location="cpu", weights_only=False)
    calibrated = torch.load(calibrated_path, map_location="cpu", weights_only=False)
    reference_path = Path(args.reference_predictions)
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)

    for name, pack in [(path.name, pack) for path, pack in zip(seed_dirs, seed_packs)] + [
        ("ensemble", ensemble),
        ("calibrated", calibrated),
        ("old_strict", reference),
    ]:
        if pack["objects"] != EXPECTED_TEST or pack["audit_objects"] != EXPECTED_AUDIT:
            raise RuntimeError(f"unexpected object order for {name}")
        if not torch.equal(pack["target"], reference["target"]):
            raise RuntimeError(f"primary target mismatch for {name}")
        if not torch.equal(pack["audit_target"], reference["audit_target"]):
            raise RuntimeError(f"audit target mismatch for {name}")

    expected_ensemble = torch.stack([pack["pred"] for pack in seed_packs]).mean(0)
    expected_audit_ensemble = torch.stack([pack["audit_pred"] for pack in seed_packs]).mean(0)
    if not torch.equal(ensemble["pred"], expected_ensemble):
        raise RuntimeError("saved primary ensemble differs from the four-seed mean")
    if not torch.equal(ensemble["audit_pred"], expected_audit_ensemble):
        raise RuntimeError("saved audit ensemble differs from the four-seed mean")
    if not torch.equal(calibrated["pred"], ensemble["pred"]):
        raise RuntimeError("identity-calibrated primary prediction differs from ensemble")
    if not torch.equal(calibrated["audit_pred"], ensemble["audit_pred"]):
        raise RuntimeError("identity-calibrated audit prediction differs from ensemble")

    calculated = metrics(calibrated, image_metrics)
    old_calculated = metrics(reference, image_metrics)
    assert_metrics(calculated, summary, args.tolerance)

    selection_splits = sorted((root / "selection").glob("fold*/w*/split_audit.json"))
    final_splits = sorted((root / "final").glob("seed*/split_audit.json"))
    if len(selection_splits) != 16 or len(final_splits) != 4:
        raise RuntimeError(f"expected 16 selection and 4 final split audits, got {len(selection_splits)} and {len(final_splits)}")
    for path in selection_splits:
        verify_split(path, selection=True)
    for path in final_splits:
        verify_split(path, selection=False)

    checkpoints = []
    for seed_dir in seed_dirs:
        checkpoint_path = seed_dir / "checkpoint_final.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        test = json.loads((seed_dir / "test.json").read_text())
        digest = sha256(checkpoint_path)
        if checkpoint.get("epoch") != 599 or not isinstance(checkpoint.get("model"), dict):
            raise RuntimeError(f"malformed checkpoint in {seed_dir}")
        if digest != test.get("checkpoint_sha256"):
            raise RuntimeError(f"checkpoint digest mismatch in {seed_dir}")
        if test.get("independent_checkpoint_reload") is not True:
            raise RuntimeError(f"checkpoint reload flag missing in {seed_dir}")
        if not all(torch.isfinite(value).all() for value in checkpoint["model"].values()):
            raise RuntimeError(f"non-finite checkpoint tensor in {seed_dir}")
        checkpoints.append({
            "seed": seed_dir.name,
            "epoch": checkpoint["epoch"],
            "sha256": digest,
            "tensor_count": len(checkpoint["model"]),
        })

    current_ssim = calculated["primary_test"]["overall"]["ssim"]
    old_ssim = old_calculated["primary_test"]["overall"]["ssim"]
    report = {
        "status": "verified",
        "metric_tolerance": args.tolerance,
        "primary_targets_exactly_equal": True,
        "audit_targets_exactly_equal": True,
        "objects": EXPECTED_TEST,
        "audit_objects": EXPECTED_AUDIT,
        "selection_split_audits": len(selection_splits),
        "final_split_audits": len(final_splits),
        "forbidden_supervision_intersections_empty": True,
        "test_labels_used_for_selection": False,
        "checkpoint_reload": checkpoints,
        "four_seed_mean_exact": True,
        "identity_calibration_exact": True,
        "calculated": calculated,
        "old_strict": old_calculated,
        "strict_test_ssim_delta": current_ssim - old_ssim,
        "old_best_retained": current_ssim <= old_ssim,
        "artifact_sha256": {
            "ensemble_predictions": sha256(ensemble_path),
            "calibrated_predictions": sha256(calibrated_path),
            "reference_predictions": sha256(reference_path),
        },
    }
    output = Path(args.output) if args.output else root / "independent_verification.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
