"""Target-free measured-data adaptation with a label-blind selection protocol.

The base predictions are adapted per object by a tiny affine residual in logit
space.  Every optimization and candidate decision uses only measured bucket
sequences.  Test labels are loaded only in ``evaluate_after_adaptation`` after
the selected configuration has already been applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ma_psdun.core import MeasurementOperator
from ma_psdun.eval import image_metrics
from ma_psdun.labels import preprocess_label
from new_sample_train import _normalize, _truth_file, read_patterns


def load_measurements_target_free(data_root, pattern_path, objects, *, preprocess="detrend",
                                  detrend_width=127, gauss_sigma=40.0):
    """Load only measured dark/bright sequences; never opens a label file."""
    root = Path(data_root)
    pattern = read_patterns(pattern_path)
    requested = set(objects)
    rows = {}
    for obj in sorted(root.iterdir()):
        if not obj.is_dir() or obj.name not in requested:
            continue
        captures = []
        for od_dir in sorted(obj.glob("OD*")):
            path = od_dir / "traindata.txt"
            if not path.exists():
                continue
            values = np.loadtxt(path, dtype=np.float32).reshape(-1)
            if values.size != pattern.shape[0] * 2:
                raise ValueError(f"{path}: {values.size} values, expected {pattern.shape[0] * 2}")
            dark, raw = values[0::2], values[1::2]
            raw_n, dark_n = _normalize(raw, dark, preprocess, detrend_width, gauss_sigma)
            captures.append(raw_n - dark_n)
        if not captures:
            raise ValueError(f"no measured OD captures found for {obj.name}")
        signal = np.mean(np.stack(captures), axis=0)
        signal = (signal - signal.mean()) / max(float(signal.std()), 1e-8)
        rows[obj.name] = signal.astype(np.float32)
    missing = sorted(requested - set(rows))
    if missing:
        raise ValueError(f"missing measured objects: {missing}")
    return pattern, np.stack([rows[name] for name in objects]).astype(np.float32)


def _measurement_loss(pred, measured, op, obs_scale=0.35):
    expected = op.forward(pred.flatten(1))
    observed = measured * obs_scale
    return F.smooth_l1_loss(expected, observed)


def adapt_affine(base, measured, op, *, steps, lr, reg, tv_weight, obs_scale):
    """Adapt predictions using measurement consistency only."""
    base = base.detach().clamp(1e-4, 1.0 - 1e-4)
    logits = torch.logit(base)
    batch = base.shape[0]
    log_scale = torch.zeros(batch, device=base.device, requires_grad=True)
    bias = torch.zeros(batch, device=base.device, requires_grad=True)
    optimizer = torch.optim.Adam([log_scale, bias], lr=lr)
    for _ in range(steps):
        output = torch.sigmoid(log_scale[:, None, None, None].exp() * logits
                               + bias[:, None, None, None])
        tv = ((output[:, :, 1:] - output[:, :, :-1]).abs().mean()
              + (output[:, :, :, 1:] - output[:, :, :, :-1]).abs().mean())
        data = _measurement_loss(output, measured, op, obs_scale)
        prior = (log_scale.square() + bias.square()).mean()
        loss = data + reg * prior + tv_weight * tv
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_scale.clamp_(-0.7, 0.7)
            bias.clamp_(-0.7, 0.7)
    with torch.no_grad():
        output = torch.sigmoid(log_scale[:, None, None, None].exp() * logits
                               + bias[:, None, None, None])
        final_loss = _measurement_loss(output, measured, op, obs_scale)
    return output.detach(), {
        "measurement_smooth_l1": float(final_loss),
        "mean_log_scale": float(log_scale.detach().mean()),
        "mean_bias": float(bias.detach().mean()),
    }


def candidate_grid() -> list[dict]:
    return [
        {"steps": steps, "lr": lr, "reg": reg, "tv_weight": tv, "obs_scale": scale}
        for steps in (0, 20, 50, 100)
        for lr in (0.01, 0.03)
        for reg in (0.003, 0.01)
        for tv in (0.0, 0.005)
        for scale in (0.30, 0.35, 0.40)
    ]


def run_candidate(base, measured, op, cfg):
    if cfg["steps"] == 0:
        output = base.detach().clone()
        details = {"measurement_smooth_l1": float(_measurement_loss(output, measured, op, cfg["obs_scale"]))}
    else:
        output, details = adapt_affine(base, measured, op, **cfg)
    return output, details


def evaluate_after_adaptation(pred, objects, data_root, pattern_path, *, size, preprocess,
                              detrend_width, gauss_sigma, label_resample):
    """Read only test labels after target-free selection and adaptation."""
    root = Path(data_root)
    by_name = {}
    for name in objects:
        obj = root / name
        truth = _truth_file(obj)
        if truth is None:
            raise ValueError(f"missing final evaluation label for {name}")
        label, _ = preprocess_label(truth, size, resample=label_resample, crop="center", contrast="percentile")
        by_name[name] = label
    target = torch.from_numpy(np.stack([by_name[name] for name in objects]))[:, None]
    return image_metrics(pred.cpu(), target), target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--base-predictions", required=True,
                        help="torch file containing only pred and objects")
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-objects", required=True)
    parser.add_argument("--val-objects", required=True)
    parser.add_argument("--test-objects", required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--preprocess", default="detrend")
    parser.add_argument("--detrend-width", type=int, default=127)
    parser.add_argument("--gauss-sigma", type=float, default=40.0)
    parser.add_argument("--label-resample", default="bilinear")
    parser.add_argument("--psf-sigma", type=float, default=0.0)
    parser.add_argument("--psf-sigma-y", type=float, default=None)
    parser.add_argument("--psf-angle", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pack = torch.load(args.base_predictions, map_location="cpu", weights_only=False)
    if set(pack) - {"pred", "objects"}:
        raise ValueError("--base-predictions must contain only pred and objects; remove labels before adaptation")
    all_objects = list(pack["objects"])
    pred = torch.as_tensor(pack["pred"], dtype=torch.float32)
    groups = {"train": [x.strip() for x in args.train_objects.split(",") if x.strip()],
              "val": [x.strip() for x in args.val_objects.split(",") if x.strip()],
              "test": [x.strip() for x in args.test_objects.split(",") if x.strip()]}
    if set(sum(groups.values(), [])) - set(all_objects):
        raise ValueError("split contains an object absent from base predictions")
    patterns, measured_np = load_measurements_target_free(
        args.data_root, args.patterns, all_objects, preprocess=args.preprocess,
        detrend_width=args.detrend_width, gauss_sigma=args.gauss_sigma,
    )
    index = {name: i for i, name in enumerate(all_objects)}
    op = MeasurementOperator(patterns.shape[1], patterns.shape[0], patterns=torch.from_numpy(patterns),
                             device=device, psf_sigma=args.psf_sigma, psf_sigma_y=args.psf_sigma_y,
                             psf_angle=args.psf_angle, image_hw=(args.size, args.size))
    base = pred.to(device)
    measured = torch.from_numpy(measured_np).to(device)
    train_idx = [index[name] for name in groups["train"]]
    val_idx = [index[name] for name in groups["val"]]
    test_idx = [index[name] for name in groups["test"]]
    # Selection remains label-blind: train and validation measurement losses are
    # the only ranking signals, while the test split is untouched here.
    ranking = []
    for cfg in candidate_grid():
        train_out, train_details = run_candidate(base[train_idx], measured[train_idx], op, cfg)
        val_out, val_details = run_candidate(base[val_idx], measured[val_idx], op, cfg)
        selection_loss = 0.5 * train_details["measurement_smooth_l1"] + 0.5 * val_details["measurement_smooth_l1"]
        ranking.append({"candidate": cfg, "train": train_details, "validation": val_details,
                        "selection_measurement_loss": selection_loss})
    ranking.sort(key=lambda row: row["selection_measurement_loss"])
    selected = ranking[0]["candidate"]
    test_out, test_details = run_candidate(base[test_idx], measured[test_idx], op, selected)
    out = Path(args.exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "completed",
        "protocol": "target-free measurement-consistency adaptation",
        "candidate_count": len(ranking),
        "selection": "mean(train measurement loss, validation measurement loss)",
        "test_labels_read_for_selection": False,
        "selected": selected,
        "ranking": ranking,
        "train_objects": groups["train"],
        "validation_objects": groups["val"],
        "test_objects": groups["test"],
        "test_adaptation": test_details,
        "base_predictions_sha256": hashlib.sha256(Path(args.base_predictions).read_bytes()).hexdigest(),
        "config": vars(args),
    }
    # The target-free artifact deliberately contains no target tensor.
    torch.save({"pred": test_out.cpu(), "objects": groups["test"]}, out / "adapted_predictions_target_free.pt")
    result["test_labels_read_after_adaptation"] = True
    metrics, target = evaluate_after_adaptation(
        test_out.cpu(), groups["test"], args.data_root, args.patterns, size=args.size,
        preprocess=args.preprocess, detrend_width=args.detrend_width,
        gauss_sigma=args.gauss_sigma, label_resample=args.label_resample,
    )
    result["final_evaluation"] = metrics
    torch.save({"pred": test_out.cpu(), "target": target, "objects": groups["test"]}, out / "predictions.pt")
    (out / "self_supervised.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (out / "DONE").write_text("completed\n")
    print(json.dumps({"selected": selected, "final_evaluation": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
