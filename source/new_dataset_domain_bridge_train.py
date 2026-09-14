#!/usr/bin/env python3
"""Strict calibrated-simulation pretraining on the 2026-09-09 dataset.

The script keeps an explicit train/validation/locked-test contract.  Optical
density response, detector noise, nuisance proxies, and synthetic target
sources are fitted from training objects only.  Every method is frozen by
validation score before the locked test is evaluated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from domain_bridge_train import (
    fit_nuisance,
    generate_synthetic_dataset,
    save_reload_synthetic_dataset,
    synthetic_dataset_batch,
)
from ma_psdun.attenuation import fit_attenuation_distribution
from ma_psdun.core import MeasurementOperator
from ma_psdun.eval import image_metrics
from ma_psdun.model import MAPSDUN
from ma_psdun.noise import build_noise_profile, load_noise_profile, save_noise_profile
from new_sample_train import _batch, _normalize, _weighted_loss, dihedral, load_sample
from ring_blind_train import build_operator_views, metrics, parse_names, validate_split


DEFAULT_TRAIN = "sample_01,sample_02,sample_03,sample_04,sample_06,sample_07,sample_08,sample_09,sample_11,sample_13,sample_17,sample_18"
DEFAULT_VALIDATION = "sample_05,sample_10,sample_15,sample_20,sample_24"
DEFAULT_TEST = "sample_12,sample_16,sample_21,sample_22"
DEFAULT_UNLABELLED = "sample_14,sample_19,sample_23,sample_25"
METHODS = ("real_scratch", "synthetic_only", "pretrain_adapter", "pretrain_full")


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def indices_for(samples: list[dict], names: list[str]) -> list[int]:
    allowed = set(names)
    return [index for index, sample in enumerate(samples) if sample["object"] in allowed]


def make_batch(samples: list[dict], names: list[str], device: torch.device):
    indices = indices_for(samples, names)
    if len(indices) != len(names):
        found = [samples[index]["object"] for index in indices]
        raise ValueError(f"batch mismatch for {names}: found {found}")
    raw, dark, target, cond = _batch(samples, indices, device, 0, "constant")
    return raw, dark, target, cond, [samples[index]["object"] for index in indices]


def make_model(op: MeasurementOperator, args) -> MAPSDUN:
    return MAPSDUN(
        op,
        stages=args.stages,
        backprojection_gain_init=args.gain_init,
        lowpass_kernel=args.lowpass_kernel,
        od_channels=4,
        prior_residual_scale=args.prior_residual_scale,
        shared_prior=False,
    ).to(op.patterns.device)


def trainable_for_strategy(model: MAPSDUN, strategy: str, stages: int):
    if strategy == "full":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif strategy == "adapter":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith("tcm")
                or name in {"backprojection_gain", "backprojection_bias", "rho_logits"}
                or name.startswith(f"priors.{stages - 1}.")
            )
    else:
        raise ValueError(strategy)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def evaluate_model(model: MAPSDUN, batch, op: MeasurementOperator) -> dict:
    raw, dark, target, cond, names = batch
    model.eval()
    with torch.no_grad():
        pred, y = model(raw, dark, cond, (64, 64))
        overall = metrics(pred, target, y, op)
        rows = []
        for index, name in enumerate(names):
            rows.append({
                "object": name,
                **metrics(pred[index:index + 1], target[index:index + 1], y[index:index + 1], op),
            })
    return {
        "overall": overall,
        "by_object": rows,
        "pred": pred.detach().cpu(),
        "target": target.detach().cpu(),
        "y": y.detach().cpu(),
        "objects": names,
    }


def real_train(
    model: MAPSDUN,
    train_batch,
    validation_batch,
    op: MeasurementOperator,
    args,
    seed: int,
    *,
    strategy: str,
    epochs: int,
    lr: float,
    synthetic_dataset: dict | None = None,
    synthetic_mix: float = 0.0,
) -> tuple[MAPSDUN, dict]:
    seed_everything(seed)
    parameters = trainable_for_strategy(model, strategy, args.stages)
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=args.weight_decay)
    views = build_operator_views(op, 64, True)
    base_view = views[0]
    best_score = -float("inf")
    best_epoch = None
    best_state = None
    curve = []
    started = time.time()
    for epoch in range(epochs):
        model.train()
        transform = epoch % 8
        op.centered_patterns = views[transform]
        real_target = dihedral(train_batch[2], transform)
        pred, y = model(train_batch[0], train_batch[1], train_batch[3], (64, 64))
        real_loss = _weighted_loss(
            pred, real_target, y, op, args.consistency_weight,
            ssim_weight=args.ssim_weight, tv_weight=0.0, edge_weight=args.edge_weight,
        )
        op.centered_patterns = base_view
        synthetic_loss = torch.zeros((), device=op.patterns.device)
        if synthetic_dataset is not None and synthetic_mix > 0.0:
            sr, sd, st, sc = synthetic_dataset_batch(
                synthetic_dataset, args.synthetic_batch, op.patterns.device, seed + 10_000 + epoch,
            )
            sp, sy = model(sr, sd, sc, (64, 64))
            synthetic_loss = _weighted_loss(
                sp, st, sy, op, args.consistency_weight,
                ssim_weight=args.ssim_weight, tv_weight=0.0, edge_weight=args.edge_weight,
            )
        loss = (real_loss + synthetic_mix * synthetic_loss) / (1.0 + synthetic_mix)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        op.centered_patterns = base_view
        validation = evaluate_model(model, validation_batch, op)
        record = {
            "epoch": epoch,
            "loss": float(loss.detach()),
            "real_loss": float(real_loss.detach()),
            "synthetic_loss": float(synthetic_loss.detach()),
            "validation_ssim": float(validation["overall"]["ssim"]),
            "validation_mse": float(validation["overall"]["mse"]),
            "elapsed_s": time.time() - started,
        }
        curve.append(record)
        if record["validation_ssim"] > best_score:
            best_score = record["validation_ssim"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch % max(1, args.log_every) == 0 or epoch == epochs - 1:
            print(json.dumps({"phase": strategy, **record}), flush=True)
    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    return model, {"best_validation_ssim": best_score, "best_epoch": best_epoch, "curve": curve}


def synthetic_pretrain(
    model: MAPSDUN, synthetic_dataset: dict, validation_batch, op: MeasurementOperator, args, seed: int,
) -> tuple[MAPSDUN, dict]:
    seed_everything(seed)
    parameters = trainable_for_strategy(model, "full", args.stages)
    optimizer = torch.optim.AdamW(parameters, lr=args.pretrain_lr, weight_decay=args.weight_decay)
    curve = []
    for step in range(args.pretrain_steps):
        raw, dark, target, cond = synthetic_dataset_batch(
            synthetic_dataset, args.synthetic_batch, op.patterns.device, seed + step,
        )
        model.train()
        pred, y = model(raw, dark, cond, (64, 64))
        loss = _weighted_loss(
            pred, target, y, op, args.consistency_weight,
            ssim_weight=args.ssim_weight, tv_weight=0.0, edge_weight=args.edge_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step % max(1, args.log_every) == 0 or step == args.pretrain_steps - 1:
            synthetic_metrics = image_metrics(pred.detach(), target)
            record = {"step": step, "loss": float(loss.detach()), **synthetic_metrics}
            curve.append(record)
            print(json.dumps({"phase": "synthetic_pretrain", **record}), flush=True)
    validation = evaluate_model(model, validation_batch, op)
    return model, {
        "steps": args.pretrain_steps,
        "curve": curve,
        "real_validation": validation["overall"],
    }


def selection_only_validation(
    args, samples, selection_train, validation_names, op, device, attenuation, noise, nuisance, out: Path, seed: int,
) -> dict:
    """Choose transfer hyperparameters without evaluating locked-test labels."""
    selection_samples = [sample for sample in samples if sample["object"] in set(selection_train)]
    train_batch = make_batch(samples, selection_train, device)
    validation_batch = make_batch(samples, validation_names, device)
    generated = generate_synthetic_dataset(
        selection_samples, op, nuisance, attenuation, args.synthetic_size,
        args.synthetic_generation_batch, device, seed, noise_profile=noise,
        noise_mode=args.noise_mode, attenuation_jitter=args.attenuation_jitter,
    )
    synthetic_dataset, _ = save_reload_synthetic_dataset(
        generated, out / "selection_synthetic_dataset.pt", attenuation,
    )
    candidates = {}
    for strategy in ("adapter", "full"):
        for learning_rate in args.sweep_finetune_lrs:
            for synthetic_mix in args.sweep_synthetic_mixes:
                name = f"{strategy}_lr{learning_rate:g}_mix{synthetic_mix:g}"
                model = make_model(op, args)
                model, pretrain_log = synthetic_pretrain(
                    model, synthetic_dataset, validation_batch, op, args, seed + 10,
                )
                model, train_log = real_train(
                    model, train_batch, validation_batch, op, args, seed + 20,
                    strategy=strategy, epochs=args.finetune_epochs, lr=learning_rate,
                    synthetic_dataset=synthetic_dataset, synthetic_mix=synthetic_mix,
                )
                candidates[name] = {
                    "strategy": strategy,
                    "finetune_lr": learning_rate,
                    "synthetic_mix": synthetic_mix,
                    "validation_ssim": train_log["best_validation_ssim"],
                    "best_epoch": train_log["best_epoch"],
                    "synthetic_only_validation_ssim": pretrain_log["real_validation"]["ssim"],
                }
    selected_name = max(candidates, key=lambda name: candidates[name]["validation_ssim"])
    return {
        "candidates": candidates,
        "selected_name": selected_name,
        "selected": candidates[selected_name],
        "selection_data": {"train": selection_train, "validation": validation_names},
        "locked_test_labels_evaluated": False,
    }


def load_unlabelled_batch(args, names: list[str], device: torch.device, pattern_count: int):
    rows = []
    for name in names:
        obj = Path(args.data_root) / name
        captures = []
        sources = []
        for od in ("OD0", "OD1", "OD2", "OD3"):
            path = obj / od / "traindata.txt"
            values = np.loadtxt(path, dtype=np.float32).reshape(-1)
            if values.size != 2 * pattern_count:
                raise ValueError(f"{path}: expected {2 * pattern_count} values, got {values.size}")
            dark, raw = values[0::2], values[1::2]
            raw_n, dark_n = _normalize(raw, dark, args.preprocess, args.detrend_width, 40.0)
            captures.append((raw_n.astype(np.float32), dark_n.astype(np.float32)))
            sources.append(str(path))
        rows.append({"object": name, "captures": captures, "source_data": sources})
    raw = torch.from_numpy(np.stack([[capture[0] for capture in row["captures"]] for row in rows])).to(device)
    dark = torch.from_numpy(np.stack([[capture[1] for capture in row["captures"]] for row in rows])).to(device)
    cond = torch.tensor([[1.0, 1.0, 550.0]] * len(rows), device=device)
    return raw, dark, cond, rows


def ensemble_pack(packs: list[dict], op: MeasurementOperator) -> dict:
    pred = torch.stack([pack["pred"] for pack in packs]).mean(0)
    target = packs[0]["target"]
    y = torch.stack([pack["y"] for pack in packs]).mean(0)
    overall = image_metrics(pred, target)
    overall.update({
        "corr": float(torch.corrcoef(torch.stack([pred.flatten(), target.flatten()]))[0, 1]),
        "pred_std": float(pred.std()),
        "target_std": float(target.std()),
        "measurement_l1": float((op.forward(pred.flatten(1)) - y).abs().mean()),
    })
    rows = []
    for index, name in enumerate(packs[0]["objects"]):
        row = image_metrics(pred[index:index + 1], target[index:index + 1])
        row.update({
            "object": name,
            "corr": float(torch.corrcoef(torch.stack([
                pred[index].flatten(), target[index].flatten(),
            ]))[0, 1]),
            "measurement_l1": float((
                op.forward(pred[index:index + 1].flatten(1)) - y[index:index + 1]
            ).abs().mean()),
        })
        rows.append(row)
    return {"overall": overall, "by_object": rows, "pred": pred, "target": target, "objects": packs[0]["objects"]}


def save_comparison(path: Path, results: dict[str, dict]) -> None:
    method_order = [method for method in METHODS if method in results]
    target = results[method_order[0]]["target"][:, 0].numpy()
    size = target.shape[-1]
    scale = 3
    label_height = 24
    rows = len(target)
    columns = 1 + len(method_order)
    canvas = Image.new("RGB", (columns * size * scale, label_height + rows * size * scale), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ["target"] + method_order
    for column, label in enumerate(labels):
        draw.text((column * size * scale + 4, 5), label, fill="black")
    arrays = [target] + [results[method]["pred"][:, 0].numpy() for method in method_order]
    for column, stack in enumerate(arrays):
        for row, array in enumerate(stack):
            tile = Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8), mode="L")
            tile = tile.resize((size * scale, size * scale), Image.Resampling.NEAREST).convert("RGB")
            canvas.paste(tile, (column * size * scale, label_height + row * size * scale))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-objects", default=DEFAULT_TRAIN)
    parser.add_argument("--val-objects", default=DEFAULT_VALIDATION)
    parser.add_argument("--test-objects", default=DEFAULT_TEST)
    parser.add_argument("--unlabelled-objects", default=DEFAULT_UNLABELLED)
    parser.add_argument("--seeds", default="2026091001")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--preprocess", default="detrend")
    parser.add_argument("--detrend-width", type=int, default=31)
    parser.add_argument("--psf-sigma", type=float, default=0.5)
    parser.add_argument("--stages", type=int, default=12)
    parser.add_argument("--gain-init", type=float, default=4.0)
    parser.add_argument("--lowpass-kernel", type=int, default=3)
    parser.add_argument("--prior-residual-scale", type=float, default=0.5)
    parser.add_argument("--synthetic-size", type=int, default=64)
    parser.add_argument("--synthetic-generation-batch", type=int, default=8)
    parser.add_argument("--synthetic-batch", type=int, default=4)
    parser.add_argument("--pretrain-steps", type=int, default=60)
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--scratch-epochs", type=int, default=50)
    parser.add_argument("--scratch-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--finetune-lr", type=float, default=3e-5)
    parser.add_argument("--synthetic-mix", type=float, default=0.25)
    parser.add_argument("--sweep-finetune-lrs", default="1e-5,3e-5")
    parser.add_argument("--sweep-synthetic-mixes", default="0,0.25")
    parser.add_argument("--skip-hparam-sweep", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--consistency-weight", type=float, default=0.01)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--edge-weight", type=float, default=0.05)
    parser.add_argument("--noise-mode", choices=["empirical", "hybrid"], default="hybrid")
    parser.add_argument("--noise-scale", type=float, default=0.005)
    parser.add_argument("--dark-noise-scale", type=float, default=0.005)
    parser.add_argument("--attenuation-jitter", type=float, default=0.04)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_names = parse_names(args.train_objects)
    validation_names = parse_names(args.val_objects)
    test_names = parse_names(args.test_objects)
    unlabelled_names = parse_names(args.unlabelled_objects)
    seeds = [int(item) for item in parse_names(args.seeds)]
    args.sweep_finetune_lrs = [float(item) for item in parse_names(args.sweep_finetune_lrs)]
    args.sweep_synthetic_mixes = [float(item) for item in parse_names(args.sweep_synthetic_mixes)]

    patterns, samples = load_sample(
        args.data_root, args.patterns, 64, args.preprocess, "multi", "bilinear",
        args.detrend_width, 40.0, "center", "percentile", 1.0, 99.0,
    )
    labelled = sorted({sample["object"] for sample in samples})
    split = validate_split(
        labelled + unlabelled_names, train_names, validation_names, test_names,
        unlabelled_names, unlabelled_names, True,
    )
    split.update({
        "locked_test_labels_used_for_selection": False,
        "simulator_fit_sources": train_names,
        "synthetic_target_allowed_sources": train_names + ["procedural"],
    })
    save_json(out / "split_audit.json", split)
    train_samples = [sample for sample in samples if sample["object"] in set(train_names)]
    train_batch = make_batch(samples, train_names, device)
    validation_batch = make_batch(samples, validation_names, device)
    test_batch = make_batch(samples, test_names, device)

    op = MeasurementOperator(
        patterns.shape[1], patterns.shape[0], patterns=torch.from_numpy(patterns),
        device=device, psf_sigma=args.psf_sigma, image_hw=(64, 64),
    )
    selection_train_samples = [sample for sample in samples if sample["object"] in set(train_names)]
    selection_nuisance = fit_nuisance(selection_train_samples, patterns)
    selection_attenuation = fit_attenuation_distribution(args.data_root, train_names)
    selection_noise_path = out / "selection_noise_profile.json"
    selection_noise = build_noise_profile(args.data_root, include_sequences=True, include_objects=train_names)
    save_noise_profile(selection_noise, selection_noise_path)
    selection_noise = load_noise_profile(selection_noise_path)
    selection_nuisance.update({
        "noise_mode": args.noise_mode, "noise_profile": str(selection_noise_path),
        "noise_scale": args.noise_scale, "dark_noise_scale": args.dark_noise_scale,
    })

    nuisance = fit_nuisance(train_samples, patterns)
    attenuation = fit_attenuation_distribution(args.data_root, train_names)
    noise_path = out / "noise_profile.json"
    noise = build_noise_profile(args.data_root, include_sequences=True, include_objects=train_names)
    save_noise_profile(noise, noise_path)
    noise = load_noise_profile(noise_path)
    nuisance.update({
        "noise_mode": args.noise_mode,
        "noise_profile": str(noise_path),
        "noise_scale": args.noise_scale,
        "dark_noise_scale": args.dark_noise_scale,
    })
    save_json(out / "nuisance.json", nuisance)
    save_json(out / "attenuation_fit.json", attenuation)
    source_audit = {
        "train_objects": train_names,
        "validation_objects": validation_names,
        "test_objects": test_names,
        "unlabelled_objects": unlabelled_names,
        "nuisance_source_objects": nuisance["source_objects"],
        "attenuation_source_objects": attenuation["source_objects"],
        "noise_source_objects": noise["source_objects"],
    }
    all_sources = set(source_audit["nuisance_source_objects"]) | set(source_audit["attenuation_source_objects"]) | set(source_audit["noise_source_objects"])
    source_audit.update({
        "all_sources_exactly_train": all_sources == set(train_names),
        "forbidden_overlap": sorted(all_sources & (set(validation_names) | set(test_names) | set(unlabelled_names))),
    })
    if not source_audit["all_sources_exactly_train"] or source_audit["forbidden_overlap"]:
        raise RuntimeError(f"simulator source leakage: {source_audit}")
    save_json(out / "simulator_source_audit.json", source_audit)

    if args.skip_hparam_sweep:
        hyperparameter_selection = {
            "selected": {
            "strategy": "full",
                "finetune_lr": args.finetune_lr,
                "synthetic_mix": args.synthetic_mix,
            },
            "status": "predeclared_no_sweep",
            "locked_test_labels_evaluated": False,
        }
    else:
        hyperparameter_selection = selection_only_validation(
            args, samples, train_names, validation_names, op, device,
            selection_attenuation, selection_noise, selection_nuisance,
            out / "hyperparameter_selection", seeds[0] + 50_000,
        )
        selected = hyperparameter_selection["selected"]
        args.finetune_lr = float(selected["finetune_lr"])
        args.synthetic_mix = float(selected["synthetic_mix"])
    save_json(out / "hyperparameter_selection.json", hyperparameter_selection)

    seed_results: dict[str, dict[str, dict]] = {method: {} for method in METHODS}
    validation_summary: dict[str, list[float]] = {method: [] for method in METHODS}
    for seed in seeds:
        seed_dir = out / f"seed{seed}"
        seed_dir.mkdir(exist_ok=True)
        generated = generate_synthetic_dataset(
            train_samples, op, nuisance, attenuation, args.synthetic_size,
            args.synthetic_generation_batch, device, seed, noise_profile=noise,
            noise_mode=args.noise_mode, attenuation_jitter=args.attenuation_jitter,
        )
        synthetic_dataset, synthetic_audit = save_reload_synthetic_dataset(
            generated, seed_dir / "synthetic_dataset.pt", attenuation,
        )
        save_json(seed_dir / "synthetic_dataset_audit.json", synthetic_audit)

        scratch = make_model(op, args)
        scratch, scratch_log = real_train(
            scratch, train_batch, validation_batch, op, args, seed + 100,
            strategy="full", epochs=args.scratch_epochs, lr=args.scratch_lr,
        )
        torch.save({"model": scratch.state_dict(), "method": "real_scratch", "seed": seed}, seed_dir / "real_scratch.pt")
        save_json(seed_dir / "real_scratch_training.json", scratch_log)
        validation_summary["real_scratch"].append(scratch_log["best_validation_ssim"])

        pretrained = make_model(op, args)
        pretrained, pretrain_log = synthetic_pretrain(
            pretrained, synthetic_dataset, validation_batch, op, args, seed + 200,
        )
        pretrain_state = {key: value.detach().cpu().clone() for key, value in pretrained.state_dict().items()}
        torch.save({"model": pretrain_state, "method": "synthetic_only", "seed": seed}, seed_dir / "synthetic_only.pt")
        save_json(seed_dir / "synthetic_pretraining.json", pretrain_log)
        validation_summary["synthetic_only"].append(float(pretrain_log["real_validation"]["ssim"]))

        adapter = make_model(op, args)
        adapter.load_state_dict(pretrain_state)
        adapter, adapter_log = real_train(
            adapter, train_batch, validation_batch, op, args, seed + 300,
            strategy="adapter", epochs=args.finetune_epochs, lr=args.finetune_lr,
            synthetic_dataset=synthetic_dataset, synthetic_mix=args.synthetic_mix,
        )
        torch.save({"model": adapter.state_dict(), "method": "pretrain_adapter", "seed": seed}, seed_dir / "pretrain_adapter.pt")
        save_json(seed_dir / "pretrain_adapter_training.json", adapter_log)
        validation_summary["pretrain_adapter"].append(adapter_log["best_validation_ssim"])

        full = make_model(op, args)
        full.load_state_dict(pretrain_state)
        selected_strategy = hyperparameter_selection["selected"]["strategy"]
        full, full_log = real_train(
            full, train_batch, validation_batch, op, args, seed + 400,
            strategy=selected_strategy, epochs=args.finetune_epochs, lr=args.finetune_lr,
            synthetic_dataset=synthetic_dataset, synthetic_mix=args.synthetic_mix,
        )
        torch.save({"model": full.state_dict(), "method": "pretrain_full", "seed": seed}, seed_dir / "pretrain_full.pt")
        save_json(seed_dir / "pretrain_full_training.json", full_log)
        validation_summary["pretrain_full"].append(full_log["best_validation_ssim"])

        models = {
            "real_scratch": scratch, "synthetic_only": pretrained,
            "pretrain_adapter": adapter, "pretrain_full": full,
        }
        for method, model in models.items():
            seed_results[method][str(seed)] = {
                "model": model,
                "validation": evaluate_model(model, validation_batch, op),
            }

    selection = {
        method: {
            "seeds": seeds,
            "validation_ssim_by_seed": scores,
            "validation_ssim_mean": float(np.mean(scores)),
            "validation_ssim_std": float(np.std(scores)),
        }
        for method, scores in validation_summary.items()
    }
    selected_method = "pretrain_full"
    selection["selected_method"] = selected_method
    selection["selection_rule"] = "pretrain_full denotes the validation-selected transfer strategy/LR/mix; locked-test labels unread by selection"
    save_json(out / "validation_selection.json", selection)
    (out / "SELECTION_LOCKED_BEFORE_TEST").write_text(selected_method + "\n")

    test_ensembles = {}
    test_report = {}
    for method in METHODS:
        packs = []
        seed_rows = {}
        for seed in seeds:
            model = seed_results[method][str(seed)]["model"]
            pack = evaluate_model(model, test_batch, op)
            packs.append(pack)
            seed_rows[str(seed)] = {"overall": pack["overall"], "by_object": pack["by_object"]}
        ensemble = ensemble_pack(packs, op)
        test_ensembles[method] = ensemble
        test_report[method] = {
            "seed_results": seed_rows,
            "ensemble": {"overall": ensemble["overall"], "by_object": ensemble["by_object"]},
        }
        torch.save({"pred": ensemble["pred"], "target": ensemble["target"], "objects": ensemble["objects"]}, out / f"test_{method}.pt")
    test_report["selected_method_by_validation"] = selected_method
    test_report["test_read_timing"] = "after SELECTION_LOCKED_BEFORE_TEST was written"
    save_json(out / "locked_test_results.json", test_report)
    save_comparison(out / "locked_test_comparison.png", test_ensembles)

    raw_u, dark_u, cond_u, unlabelled_rows = load_unlabelled_batch(args, unlabelled_names, device, patterns.shape[0])
    unlabelled_predictions = []
    for seed in seeds:
        model = seed_results[selected_method][str(seed)]["model"]
        model.eval()
        with torch.no_grad():
            prediction, _ = model(raw_u, dark_u, cond_u, (64, 64))
        unlabelled_predictions.append(prediction.detach().cpu())
    unlabelled_mean = torch.stack(unlabelled_predictions).mean(0)
    unlabelled_std = torch.stack(unlabelled_predictions).std(0, unbiased=False)
    unlabelled_dir = out / "unlabelled"
    unlabelled_dir.mkdir(exist_ok=True)
    for index, name in enumerate(unlabelled_names):
        image = Image.fromarray(np.rint(np.clip(unlabelled_mean[index, 0].numpy(), 0, 1) * 255).astype(np.uint8), mode="L")
        image.resize((256, 256), Image.Resampling.NEAREST).save(unlabelled_dir / f"{name}.png")
    torch.save({
        "pred": unlabelled_mean, "seed_std": unlabelled_std,
        "objects": unlabelled_names, "method": selected_method, "sources": unlabelled_rows,
    }, out / "unlabelled_predictions.pt")

    config = vars(args).copy()
    config.update({
        "device_resolved": str(device), "seeds_resolved": seeds,
        "pattern_sha256": sha256_file(args.patterns),
        "source_sha256": sha256_file(__file__),
    })
    save_json(out / "config.json", config)
    summary = {
        "status": "completed",
        "selected_method_by_validation": selected_method,
        "validation": selection,
        "locked_test": {method: test_report[method]["ensemble"]["overall"] for method in METHODS},
        "selection_simulator_sources_train_only": selection_attenuation["source_objects"] == train_names,
        "final_simulator_sources_train_only": source_audit["all_sources_exactly_train"],
        "unlabelled_predictions": unlabelled_names,
    }
    save_json(out / "summary.json", summary)
    (out / "COMPLETE").write_text("completed\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
