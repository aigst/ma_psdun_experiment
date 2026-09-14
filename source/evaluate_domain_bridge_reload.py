"""Independently reload and evaluate a domain-bridge run.

The training process writes test predictions after its optional target-free
adaptation.  This checker deliberately ignores those predictions when it
reconstructs each ``checkpoint_best.pt`` and reruns the fixed test objects.
It also reports the saved ensemble separately so the two evaluation paths are
not conflated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from domain_bridge_train import make_model, preserve_multi_od_amplitude, real_batch
from ma_psdun.core import MeasurementOperator
from ma_psdun.eval import image_metrics
from new_sample_train import load_sample


def _metrics_by_object(pred, target, objects):
    rows = []
    for index, name in enumerate(objects):
        rows.append({"object": name, **image_metrics(pred[index:index + 1], target[index:index + 1])})
    return rows


def _mean_metrics(pred, target):
    return {key: float(value) for key, value in image_metrics(pred, target).items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--test-objects", nargs="+", default=["obj18-20260901", "obj20-20260901"])
    parser.add_argument("--label-resample", choices=["nearest", "bilinear", "bicubic", "lanczos"], default="lanczos")
    parser.add_argument("--psf-sigma", type=float, required=True)
    parser.add_argument("--psf-sigma-y", type=float, required=True)
    parser.add_argument("--psf-angle", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preserve-od-amplitude", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    patterns, samples = load_sample(
        args.data_root, args.patterns, 64, "detrend", "multi", args.label_resample, 127, 40.0,
    )
    if args.preserve_od_amplitude:
        samples = preserve_multi_od_amplitude(samples, "detrend", 127, 40.0)
    op = MeasurementOperator(
        patterns.shape[1], patterns.shape[0], patterns=torch.from_numpy(patterns), device=device,
        psf_sigma=args.psf_sigma, psf_sigma_y=args.psf_sigma_y, psf_angle=args.psf_angle,
        image_hw=(64, 64),
    )
    model_args = SimpleNamespace(
        stages=12, shared_prior=False, lowpass_kernel=3, prior_residual_scale=0.50,
        preserve_od_amplitude=args.preserve_od_amplitude,
    )
    test_batch = real_batch(samples, list(args.test_objects), device)
    reload_rows = []
    saved_rows = []
    reload_predictions = []
    final_reload_predictions = []
    targets = None
    root = Path(args.exp_dir)
    fold_dirs = sorted(root.glob("fold*_baseline"))
    if not fold_dirs:
        raise FileNotFoundError(f"no fold*_baseline directories under {root}")
    for fold_dir in fold_dirs:
        checkpoint_path = fold_dir / "checkpoint_best.pt"
        final_checkpoint_path = fold_dir / "checkpoint_final.pt"
        saved_prediction_path = fold_dir / "test_predictions.pt"
        if not checkpoint_path.exists() or not final_checkpoint_path.exists() or not saved_prediction_path.exists():
            raise FileNotFoundError(f"missing best/final checkpoint or saved predictions in {fold_dir}")
        model = make_model(op, model_args).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        with torch.no_grad():
            pred, _ = model(test_batch[0], test_batch[1], test_batch[3], (64, 64))
        target = test_batch[2].detach().cpu()
        pred_cpu = pred.detach().cpu()
        objects = list(test_batch[4])
        reload_predictions.append(pred_cpu)
        targets = target if targets is None else targets
        final_checkpoint = torch.load(final_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(final_checkpoint["model"])
        model.eval()
        with torch.no_grad():
            final_pred, _ = model(test_batch[0], test_batch[1], test_batch[3], (64, 64))
        final_pred_cpu = final_pred.detach().cpu()
        final_reload_predictions.append(final_pred_cpu)
        reload_rows.append({
            "fold": fold_dir.name,
            "checkpoint": str(checkpoint_path),
            "best_validation_ssim": float(checkpoint.get("best_validation_ssim", float("nan"))),
            "checkpoint_reload": {
                "overall": _mean_metrics(pred_cpu, target),
                "by_object": _metrics_by_object(pred_cpu, target, objects),
            },
            "final_checkpoint": str(final_checkpoint_path),
            "final_checkpoint_reload": {
                "stage": final_checkpoint.get("stage"),
                "uda_accepted": bool(final_checkpoint.get("uda_accepted", False)),
                "overall": _mean_metrics(final_pred_cpu, target),
                "by_object": _metrics_by_object(final_pred_cpu, target, objects),
            },
        })
        saved = torch.load(saved_prediction_path, map_location="cpu", weights_only=False)
        saved_pred = saved["pred"].float()
        saved_target = saved["target"].float()
        if list(saved["objects"]) != objects:
            raise ValueError(f"object order mismatch in {saved_prediction_path}")
        saved_rows.append({
            "fold": fold_dir.name,
            "saved_test_predictions": {
                "overall": _mean_metrics(saved_pred, target),
                "by_object": _metrics_by_object(saved_pred, target, objects),
            },
            "saved_target_label_resample": "lanczos",
            "max_abs_saved_target_vs_canonical": float((saved_target - target).abs().max()),
            "max_abs_best_reload_vs_saved": float((pred_cpu - saved_pred).abs().max()),
            "max_abs_final_reload_vs_saved": float((final_pred_cpu - saved_pred).abs().max()),
        })

    reloaded_ensemble = torch.stack(reload_predictions).mean(dim=0)
    reloaded_final_ensemble = torch.stack(final_reload_predictions).mean(dim=0)
    result = {
        "status": "verified",
        "protocol": (
            "independent reload of checkpoint_best before target-free obj12 adaptation "
            "and checkpoint_final after accepted/rolled-back adaptation"
        ),
        "data_root": str(args.data_root),
        "test_objects": list(args.test_objects),
        "label_resample": args.label_resample,
        "preserve_od_amplitude": args.preserve_od_amplitude,
        "operator": {
            "psf_sigma": args.psf_sigma,
            "psf_sigma_y": args.psf_sigma_y,
            "psf_angle": args.psf_angle,
        },
        "folds": reload_rows,
        "saved_fold_predictions": saved_rows,
        "reloaded_ensemble": {
            "overall": _mean_metrics(reloaded_ensemble, targets),
            "by_object": _metrics_by_object(reloaded_ensemble, targets, list(args.test_objects)),
        },
        "reloaded_final_ensemble": {
            "overall": _mean_metrics(reloaded_final_ensemble, targets),
            "by_object": _metrics_by_object(reloaded_final_ensemble, targets, list(args.test_objects)),
        },
    }
    saved_ensemble_path = root / "ensemble" / "predictions.pt"
    if saved_ensemble_path.exists():
        saved_ensemble = torch.load(saved_ensemble_path, map_location="cpu", weights_only=False)
        saved_target = saved_ensemble["target"].float()
        result["saved_ensemble"] = {
            "path": str(saved_ensemble_path),
            "overall": _mean_metrics(saved_ensemble["pred"].float(), targets),
            "by_object": _metrics_by_object(
                saved_ensemble["pred"].float(), targets, list(saved_ensemble["objects"]),
            ),
            "saved_target_label_resample": "lanczos",
            "max_abs_saved_target_vs_canonical": float((saved_target - targets).abs().max()),
            "max_abs_best_reload_vs_saved": float((reloaded_ensemble - saved_ensemble["pred"].float()).abs().max()),
            "max_abs_final_reload_vs_saved": float((reloaded_final_ensemble - saved_ensemble["pred"].float()).abs().max()),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "reloaded_best_ssim": result["reloaded_ensemble"]["overall"]["ssim"],
        "reloaded_final_ssim": result["reloaded_final_ensemble"]["overall"]["ssim"],
        "saved_ssim": result.get("saved_ensemble", {}).get("overall", {}).get("ssim"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
