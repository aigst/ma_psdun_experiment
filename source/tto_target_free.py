"""Low-capacity target-free test-time adaptation for the PSF reconstruction.

The adapter changes only per-object brightness/contrast around a frozen
prediction.  Its objective uses the measured, normalized bucket sequence and
the same Gaussian-PSF operator as the base model; labels are used only to
rank candidates on the already object-disjoint OOF pack.
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
from new_sample_train import load_sample, read_patterns


FORBIDDEN = {"obj15-20260901", "obj16-20260901", "obj19-20260901"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    return (x[:, :, 1:] - x[:, :, :-1]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()


def adapt_affine(
    base: torch.Tensor,
    obs: torch.Tensor,
    op: MeasurementOperator,
    *,
    steps: int,
    lr: float,
    reg: float,
    tv_weight: float,
    obs_scale: float,
) -> tuple[torch.Tensor, dict]:
    """Adapt a batch of predictions with one affine pair per object."""
    # Parameterize positive scale and bounded output through a sigmoid.  The
    # zero initialization is exactly identity up to numerical roundoff.
    p = base.detach().clamp(1e-4, 1.0 - 1e-4)
    logits = torch.logit(p)
    batch = p.shape[0]
    log_scale = torch.zeros(batch, device=p.device, requires_grad=True)
    bias = torch.zeros(batch, device=p.device, requires_grad=True)
    optimizer = torch.optim.Adam([log_scale, bias], lr=lr)
    curve = []
    for step in range(steps):
        scale = log_scale.exp().view(batch, 1, 1, 1)
        shift = bias.view(batch, 1, 1, 1)
        output = torch.sigmoid(scale * logits + shift)
        measured = op.forward(output.flatten(1))
        data = F.smooth_l1_loss(measured, obs * obs_scale)
        prior = (log_scale.square() + bias.square()).mean()
        loss = data + reg * prior + tv_weight * tv_loss(output)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_scale.clamp_(-0.7, 0.7)
            bias.clamp_(-0.7, 0.7)
        if step == 0 or step == steps - 1 or step % max(1, steps // 5) == 0:
            curve.append({
                "step": step,
                "loss": float(loss.detach()),
                "measurement_smooth_l1": float(data.detach()),
                "tv": float(tv_loss(output).detach()),
                "mean_log_scale": float(log_scale.detach().mean()),
                "mean_bias": float(bias.detach().mean()),
            })
    with torch.no_grad():
        output = torch.sigmoid(log_scale.view(batch, 1, 1, 1).exp() * logits + bias.view(batch, 1, 1, 1))
        measured = op.forward(output.flatten(1))
    return output.detach(), {
        "method": "affine",
        "steps": steps,
        "lr": lr,
        "reg": reg,
        "tv_weight": tv_weight,
        "obs_scale": obs_scale,
        "curve": curve,
        "measurement_smooth_l1": float(F.smooth_l1_loss(measured, obs * obs_scale)),
        "log_scale": log_scale.detach().cpu().tolist(),
        "bias": bias.detach().cpu().tolist(),
    }


def adapt_conv3(
    base: torch.Tensor,
    obs: torch.Tensor,
    op: MeasurementOperator,
    *,
    steps: int,
    lr: float,
    reg: float,
    tv_weight: float,
    obs_scale: float,
) -> tuple[torch.Tensor, dict]:
    """Adapt a per-object 3x3 residual filter around a frozen prediction."""
    p = base.detach().clamp(1e-4, 1.0 - 1e-4)
    logits = torch.logit(p)
    batch = p.shape[0]
    kernel = torch.zeros(batch, 1, 3, 3, device=p.device, requires_grad=True)
    bias = torch.zeros(batch, device=p.device, requires_grad=True)
    optimizer = torch.optim.Adam([kernel, bias], lr=lr)
    curve = []
    for step in range(steps):
        # Grouped convolution applies an independent 3x3 residual to each
        # object while keeping the parameter count at ten per object.
        grouped_input = F.pad(logits[:, 0].unsqueeze(0), (1, 1, 1, 1), mode="reflect")
        delta = F.conv2d(grouped_input, kernel, bias=bias, groups=batch).reshape_as(logits)
        output = torch.sigmoid(logits + 0.25 * delta)
        measured = op.forward(output.flatten(1))
        data = F.smooth_l1_loss(measured, obs * obs_scale)
        prior = (kernel.square().mean() + bias.square().mean())
        loss = data + reg * prior + tv_weight * tv_loss(output)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            kernel.clamp_(-1.0, 1.0)
            bias.clamp_(-1.0, 1.0)
        if step == 0 or step == steps - 1 or step % max(1, steps // 5) == 0:
            curve.append({
                "step": step,
                "loss": float(loss.detach()),
                "measurement_smooth_l1": float(data.detach()),
                "tv": float(tv_loss(output).detach()),
                "kernel_rms": float(kernel.detach().square().mean().sqrt()),
                "mean_bias": float(bias.detach().mean()),
            })
    with torch.no_grad():
        grouped_input = F.pad(logits[:, 0].unsqueeze(0), (1, 1, 1, 1), mode="reflect")
        delta = F.conv2d(grouped_input, kernel, bias=bias, groups=batch).reshape_as(logits)
        output = torch.sigmoid(logits + 0.25 * delta)
        measured = op.forward(output.flatten(1))
    return output.detach(), {
        "method": "conv3",
        "steps": steps,
        "lr": lr,
        "reg": reg,
        "tv_weight": tv_weight,
        "obs_scale": obs_scale,
        "curve": curve,
        "measurement_smooth_l1": float(F.smooth_l1_loss(measured, obs * obs_scale)),
        "kernel": kernel.detach().cpu().tolist(),
        "bias": bias.detach().cpu().tolist(),
    }


def metric_pack(pred: torch.Tensor, target: torch.Tensor, objects: list[str]) -> dict:
    return {
        "overall": image_metrics(pred, target),
        "by_object": [
            {"object": name, "metrics": image_metrics(pred[i:i + 1], target[i:i + 1])}
            for i, name in enumerate(objects)
        ],
    }


def object_measurements(samples, objects: list[str], device: torch.device) -> torch.Tensor:
    by_name = {sample["object"]: sample for sample in samples}
    missing = sorted(set(objects) - set(by_name))
    if missing:
        raise ValueError(f"missing measured objects: {missing}")
    # ``load_sample(..., fusion=mean)`` already performs the exact per-object
    # normalization used by the trained single-channel PSF model.
    rows = []
    for name in objects:
        sample = by_name[name]
        signal = np.asarray(sample["raw"], np.float32) - np.asarray(sample["dark"], np.float32)
        rows.append(0.35 * signal)
    return torch.from_numpy(np.stack(rows)).to(device)


def candidate_name(cfg: dict) -> str:
    return "{method}_s{steps}_lr{lr:g}_r{reg:g}_tv{tv_weight:g}_os{obs_scale:g}".format(**cfg)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--patterns", required=True)
    p.add_argument("--oof-pack", required=True)
    p.add_argument("--selection-pack", required=True)
    p.add_argument("--final-pack", required=True)
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--psf-sigma", type=float, default=1.0)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--preprocess", default="detrend")
    p.add_argument("--detrend-width", type=int, default=127)
    p.add_argument("--gauss-sigma", type=float, default=40.0)
    p.add_argument("--label-resample", default="bilinear")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("this experiment requires the A800 CUDA Pod")
    out = Path(args.exp_dir)
    out.mkdir(parents=True, exist_ok=True)

    patterns = read_patterns(args.patterns)
    samples_pattern, samples = load_sample(
        args.data_root, args.patterns, args.size, args.preprocess, "mean",
        args.label_resample, args.detrend_width, args.gauss_sigma,
    )
    op = MeasurementOperator(
        patterns.shape[1], patterns.shape[0], patterns=torch.from_numpy(patterns),
        device=device, psf_sigma=args.psf_sigma, image_hw=(args.size, args.size),
    )
    oof = torch.load(args.oof_pack, map_location="cpu", weights_only=False)
    selected = torch.load(args.selection_pack, map_location="cpu", weights_only=False)
    final = torch.load(args.final_pack, map_location="cpu", weights_only=False)
    oof_objects = list(oof["objects"])
    if not oof.get("object_level_disjoint") or oof.get("test_labels_evaluated"):
        raise RuntimeError("OOF pack violates the object-disjoint selection contract")
    if FORBIDDEN & set(oof_objects):
        raise RuntimeError("forbidden object present in OOF pack")
    base_oof = selected["predictions"]["conv3"].float().to(device)
    target_oof = oof["target"].float().to(device)
    oof_obs = object_measurements(samples, oof_objects, device)

    candidates = []
    for steps in (20, 50, 100, 200):
        for lr in (0.01, 0.03):
            for reg in (0.003, 0.01):
                candidates.append({"method": "affine", "steps": steps, "lr": lr, "reg": reg, "tv_weight": 0.0, "obs_scale": 1.0})
    # Include a mild smoothness prior and alternate forward calibration scales
    # as predeclared candidates; selection remains entirely on OOF objects.
    candidates.extend([
        {"method": "affine", "steps": 100, "lr": 0.03, "reg": 0.01, "tv_weight": 0.005, "obs_scale": 0.95},
        {"method": "affine", "steps": 100, "lr": 0.03, "reg": 0.01, "tv_weight": 0.005, "obs_scale": 1.05},
        {"method": "affine", "steps": 200, "lr": 0.03, "reg": 0.03, "tv_weight": 0.005, "obs_scale": 1.0},
    ])
    for steps in (20, 50, 100, 200):
        for lr in (0.003, 0.01):
            for reg in (0.01, 0.05):
                candidates.append({"method": "conv3", "steps": steps, "lr": lr, "reg": reg, "tv_weight": 0.0, "obs_scale": 1.0})

    ranking = []
    candidate_outputs = {}
    for index, cfg in enumerate(candidates):
        method = cfg["method"]
        kwargs = {key: value for key, value in cfg.items() if key != "method"}
        adapter = adapt_affine if method == "affine" else adapt_conv3
        output, details = adapter(base_oof, oof_obs, op, **kwargs)
        score = metric_pack(output, target_oof, oof_objects)
        name = candidate_name(cfg)
        ranking.append({"candidate": cfg, "name": name, "cross_validation": score})
        candidate_outputs[name] = output.cpu()
        print(json.dumps({"candidate": name, "ssim": score["overall"]["ssim"]}), flush=True)
    ranking.sort(key=lambda row: (row["cross_validation"]["overall"]["ssim"], -row["cross_validation"]["overall"]["mse"]), reverse=True)
    selected_cfg = ranking[0]["candidate"]
    selected_name = ranking[0]["name"]

    test_objects = list(final["objects"])
    audit_objects = list(final["audit_objects"])
    base_test = final["pred"].float().to(device)
    base_audit = final["audit_pred"].float().to(device)
    target_test = final["target"].float().to(device)
    target_audit = final["audit_target"].float().to(device)
    test_obs = object_measurements(samples, test_objects, device)
    audit_obs = object_measurements(samples, audit_objects, device)
    selected_method = selected_cfg["method"]
    selected_kwargs = {key: value for key, value in selected_cfg.items() if key != "method"}
    adapter = adapt_affine if selected_method == "affine" else adapt_conv3
    adapted_test, test_details = adapter(base_test, test_obs, op, **selected_kwargs)
    adapted_audit, audit_details = adapter(base_audit, audit_obs, op, **selected_kwargs)

    result = {
        "status": "completed",
        "protocol": "OOF-only selection of per-object two-parameter target-free TTO",
        "base_model": "Gaussian PSF sigma=1.0 + OOF-selected conv3",
        "selected": {"name": selected_name, **selected_cfg},
        "candidate_count": len(candidates),
        "oof_objects": oof_objects,
        "forbidden_supervised_objects": sorted(FORBIDDEN),
        "test_labels_evaluated_for_selection": False,
        "oof_ranking": ranking,
        "primary_test": metric_pack(adapted_test.cpu(), target_test.cpu(), test_objects),
        "primary_test_base": metric_pack(base_test.cpu(), target_test.cpu(), test_objects),
        "audit_ring_holdout": metric_pack(adapted_audit.cpu(), target_audit.cpu(), audit_objects),
        "audit_ring_holdout_base": metric_pack(base_audit.cpu(), target_audit.cpu(), audit_objects),
        "adaptation": {"test": test_details, "audit": audit_details},
        "sha256": {
            "oof_pack": sha256(Path(args.oof_pack)),
            "selection_pack": sha256(Path(args.selection_pack)),
            "final_pack": sha256(Path(args.final_pack)),
            "patterns": sha256(Path(args.patterns)),
        },
    }
    (out / "tto.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    torch.save({
        "pred": adapted_test.cpu(), "target": target_test.cpu(), "objects": test_objects,
        "audit_pred": adapted_audit.cpu(), "audit_target": target_audit.cpu(), "audit_objects": audit_objects,
        "base_pred": base_test.cpu(), "base_audit_pred": base_audit.cpu(), "selected": result["selected"],
    }, out / "predictions.pt")
    print(json.dumps({
        "selected": result["selected"],
        "oof_ssim": ranking[0]["cross_validation"]["overall"]["ssim"],
        "base_test_ssim": result["primary_test_base"]["overall"]["ssim"],
        "tto_test_ssim": result["primary_test"]["overall"]["ssim"],
        "base_audit_ssim": result["audit_ring_holdout_base"]["overall"]["ssim"],
        "tto_audit_ssim": result["audit_ring_holdout"]["overall"]["ssim"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
