#!/usr/bin/env python3
"""Audit a completed four-fold exponential-attenuation bridge experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch

from domain_bridge_train import validate_synthetic_dataset


def load_json(path: Path):
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--synthetic-count", type=int, default=960)
    args = parser.parse_args()

    root = Path(args.exp_dir)
    fold_dirs = sorted(root.glob("fold*_baseline"))
    require(len(fold_dirs) == args.folds, f"expected {args.folds} folds, found {len(fold_dirs)}")
    train_counts: Counter[str] = Counter()
    validation_counts: Counter[str] = Counter()
    rows = []
    test_objects = None

    for fold_dir in fold_dirs:
        required = [
            "DONE", "split.json", "simulator_split_audit.json", "attenuation_fit.json",
            "noise_profile.json", "synthetic_dataset.pt", "synthetic_dataset_audit.json",
            "synthetic_manifest.json", "checkpoint_pretrain.pt", "checkpoint_best.pt",
            "checkpoint_final.pt", "test_predictions.pt", "summary.json", "run_config.json",
        ]
        missing = [name for name in required if not (fold_dir / name).is_file()]
        require(not missing, f"{fold_dir.name} missing artifacts: {missing}")

        split = load_json(fold_dir / "split.json")
        audit = load_json(fold_dir / "simulator_split_audit.json")
        attenuation = load_json(fold_dir / "attenuation_fit.json")
        noise = load_json(fold_dir / "noise_profile.json")
        dataset_audit = load_json(fold_dir / "synthetic_dataset_audit.json")
        manifest = load_json(fold_dir / "synthetic_manifest.json")
        config = load_json(fold_dir / "run_config.json")
        summary = load_json(fold_dir / "summary.json")
        train = set(split["train_objects"])
        validation = set(split["validation_objects"])
        tests = set(split["test_objects"])
        test_objects = tests if test_objects is None else test_objects
        require(tests == test_objects, "test objects differ across folds")
        require(not train & validation and not train & tests and not validation & tests,
                f"object split overlap in {fold_dir.name}")
        require(set(attenuation["source_objects"]) == train,
                f"attenuation sources differ from training split in {fold_dir.name}")
        require(set(noise["source_objects"]) == train,
                f"noise sources differ from training split in {fold_dir.name}")
        require(noise["file_count"] == 4 * len(train),
                f"noise profile does not contain four captures per training object in {fold_dir.name}")
        require(audit["status"] == "verified" and not audit["forbidden_overlap"],
                f"simulator split audit failed in {fold_dir.name}")
        require(audit["attenuation_exactly_train_only"] and audit["noise_exactly_train_only"],
                f"fold-local simulator source equality failed in {fold_dir.name}")
        require(config["fit_attenuation"] and config["fold_local_noise"]
                and config["preserve_od_amplitude"],
                f"new simulator flags are not all enabled in {fold_dir.name}")
        require(config["synthetic_dataset_size"] == args.synthetic_count,
                f"unexpected requested synthetic count in {fold_dir.name}")

        fits = attenuation["fits"]
        require(len(fits) == len(train), f"not every training object was fitted in {fold_dir.name}")
        for fit in fits:
            values = [fit["A"], fit["k"], fit["b"], fit["rmse_normalized"], fit["r2"]]
            require(all(math.isfinite(value) for value in values),
                    f"non-finite attenuation fit for {fit['object']}")
            require(fit["A"] > 0.0 and fit["k"] > 0.0 and fit["b"] >= 0.0,
                    f"invalid constrained fit for {fit['object']}")

        dataset_path = fold_dir / "synthetic_dataset.pt"
        require(dataset_audit["count"] == args.synthetic_count
                and dataset_audit["exact_tensor_reload"],
                f"synthetic save/reload audit failed in {fold_dir.name}")
        digest = sha256_file(dataset_path)
        require(digest == dataset_audit["sha256"] == manifest["dataset_sha256"],
                f"synthetic dataset hash mismatch in {fold_dir.name}")
        require(manifest["count"] == args.synthetic_count
                and len(manifest["samples"]) == args.synthetic_count,
                f"synthetic manifest count mismatch in {fold_dir.name}")
        dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
        validate_synthetic_dataset(dataset, attenuation)
        centered_std = (dataset["raw"] - dataset["dark"]).std(dim=-1)
        measured_ratio = centered_std / centered_std[:, :1].clamp_min(1e-8)
        expected_ratio = dataset["od_responses"] / dataset["od_responses"][:, :1].clamp_min(1e-8)
        amplitude_corr = float(torch.corrcoef(torch.stack([
            measured_ratio.flatten(), expected_ratio.flatten(),
        ]))[0, 1])
        require(math.isfinite(amplitude_corr) and amplitude_corr > 0.8,
                f"saved OD amplitudes do not track fitted responses in {fold_dir.name}: {amplitude_corr}")
        require(bool(torch.all(dataset["od_responses"][:, 1:] <= dataset["od_responses"][:, :-1])),
                f"synthetic response is not monotonically attenuated in {fold_dir.name}")
        sampled_sources = sorted(set(dataset["attenuation_source_object"]))
        del dataset
        gc.collect()

        train_counts.update(train)
        validation_counts.update(validation)
        rows.append({
            "fold": fold_dir.name,
            "train_count": len(train),
            "validation_count": len(validation),
            "test_count": len(tests),
            "fit_count": len(fits),
            "fit_r2_min": min(float(fit["r2"]) for fit in fits),
            "fit_r2_median": float(torch.tensor([fit["r2"] for fit in fits]).median()),
            "k_min": min(float(fit["k"]) for fit in fits),
            "k_max": max(float(fit["k"]) for fit in fits),
            "noise_file_count": noise["file_count"],
            "synthetic_count": dataset_audit["count"],
            "synthetic_sha256": digest,
            "sampled_attenuation_sources": sampled_sources,
            "od_amplitude_response_corr": amplitude_corr,
            "best_validation_ssim": summary["best_validation_ssim"],
            "test_before_tto_ssim": summary["test_before_tto"]["ssim"],
            "test_tto_ssim": summary["test_tto"]["ssim"],
        })

    eligible = set(train_counts) | set(validation_counts)
    require(all(train_counts[name] == args.folds - 1 for name in eligible),
            f"not every CV object appears in {args.folds - 1} training folds")
    require(all(validation_counts[name] == 1 for name in eligible),
            "not every CV object appears in exactly one validation fold")
    require(not eligible & test_objects, "locked tests entered cross-validation")

    ensemble = load_json(root / "ensemble" / "ensemble.json")
    independent = load_json(root / "independent_reload.json")
    require((root / "ensemble" / "DONE").is_file() and (root / "COMPLETE").is_file(),
            "experiment or ensemble completion marker is missing")
    require(ensemble["size"] == args.folds, "ensemble does not contain every fold")
    require(independent["status"] == "verified" and independent["preserve_od_amplitude"],
            "independent reload report is invalid")
    fold_reload_diffs = [
        row["max_abs_final_reload_vs_saved"]
        for row in independent["saved_fold_predictions"]
    ]
    ensemble_reload_diff = independent["saved_ensemble"]["max_abs_final_reload_vs_saved"]
    require(max(fold_reload_diffs + [ensemble_reload_diff]) <= 1e-6,
            "final checkpoint reload does not reproduce saved predictions")
    require(independent["saved_ensemble"]["max_abs_saved_target_vs_canonical"] == 0.0,
            "saved and canonical targets differ")

    result = {
        "status": "verified",
        "experiment_dir": str(root),
        "requirements": {
            "four_fold_training_complete": True,
            "each_fit_uses_only_15_fold_training_objects": True,
            "fold_local_noise_has_no_validation_or_test_objects": True,
            "joint_A_k_b_sampling_manifested": True,
            "cross_od_amplitude_preserved": True,
            "synthetic_datasets_saved_and_hash_verified": True,
            "final_checkpoints_independently_reload_exactly": True,
            "four_model_ensemble_complete": True,
        },
        "cross_validation": {
            "eligible_object_count": len(eligible),
            "test_objects": sorted(test_objects),
            "train_appearances": dict(sorted(train_counts.items())),
            "validation_appearances": dict(sorted(validation_counts.items())),
        },
        "folds": rows,
        "ensemble": ensemble,
        "independent_reload": {
            "reloaded_best_ssim": independent["reloaded_ensemble"]["overall"]["ssim"],
            "reloaded_final_ssim": independent["reloaded_final_ensemble"]["overall"]["ssim"],
            "saved_ensemble_ssim": independent["saved_ensemble"]["overall"]["ssim"],
            "max_abs_final_fold_reload_diff": max(fold_reload_diffs),
            "max_abs_final_ensemble_reload_diff": ensemble_reload_diff,
            "max_abs_saved_target_diff": independent["saved_ensemble"]["max_abs_saved_target_vs_canonical"],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": result["status"],
        "output": str(output),
        "ensemble_ssim": ensemble["full"]["ssim"],
        "reload_diff": ensemble_reload_diff,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
