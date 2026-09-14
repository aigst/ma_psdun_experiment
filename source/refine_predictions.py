"""Object-disjoint cross-validation for a low-capacity image refiner."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ma_psdun.eval import image_metrics
from ma_psdun.model import _ssim_loss


@dataclass(frozen=True)
class Candidate:
    name: str
    epochs: int
    lr: float
    weight_decay: float


CANDIDATES = (
    Candidate("identity", 0, 0.0, 0.0),
    Candidate("affine", 300, 2e-2, 0.0),
    Candidate("conv3", 300, 5e-3, 1e-4),
    Candidate("multiscale", 300, 1e-2, 1e-4),
    Candidate("residual8", 400, 3e-3, 1e-4),
)


class AffineRefiner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.log_scale.exp() + self.bias).clamp(0.0, 1.0)


class Conv3Refiner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=0)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        self.conv.weight.data[0, 0, 1, 1] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (1, 1, 1, 1), mode="reflect")).clamp(0.0, 1.0)


class MultiscaleRefiner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.head.weight.data[0, 0, 0, 0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [x, F.avg_pool2d(x, 3, 1, 1), F.avg_pool2d(x, 5, 1, 2)],
            dim=1,
        )
        return self.head(features).clamp(0.0, 1.0)


class ResidualRefiner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(8, 8, 3, padding=1, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(8, 1, 3, padding=1, padding_mode="reflect"),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 0.25 * torch.tanh(self.body(x))).clamp(0.0, 1.0)


def make_refiner(name: str) -> nn.Module:
    if name == "affine":
        return AffineRefiner()
    if name == "conv3":
        return Conv3Refiner()
    if name == "multiscale":
        return MultiscaleRefiner()
    if name == "residual8":
        return ResidualRefiner()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"unknown refiner: {name}")


def dihedral(x: torch.Tensor, transform: int) -> torch.Tensor:
    if transform >= 4:
        x = x.flip(-1)
        transform -= 4
    return torch.rot90(x, transform, (-2, -1))


def train_refiner(candidate: Candidate, pred: torch.Tensor, target: torch.Tensor, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    model = make_refiner(candidate.name)
    if candidate.epochs == 0:
        return model.eval()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=candidate.lr, weight_decay=candidate.weight_decay,
    )
    for epoch in range(candidate.epochs):
        transformed_pred = dihedral(pred, epoch % 8)
        transformed_target = dihedral(target, epoch % 8)
        output = model(transformed_pred)
        edge = F.l1_loss(output[:, :, 1:] - output[:, :, :-1], transformed_target[:, :, 1:] - transformed_target[:, :, :-1])
        edge = edge + F.l1_loss(output[:, :, :, 1:] - output[:, :, :, :-1], transformed_target[:, :, :, 1:] - transformed_target[:, :, :, :-1])
        loss = (F.l1_loss(output, transformed_target)
                + 0.5 * F.mse_loss(output, transformed_target)
                + 0.2 * _ssim_loss(output, transformed_target)
                + 0.02 * edge)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval()


def metric_pack(pred: torch.Tensor, target: torch.Tensor, objects: list[str]) -> dict:
    return {
        "overall": image_metrics(pred, target),
        "by_object": [
            {"object": name, "metrics": image_metrics(pred[index:index + 1], target[index:index + 1])}
            for index, name in enumerate(objects)
        ],
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_preview(path: Path, pred: torch.Tensor, target: torch.Tensor) -> None:
    rows = []
    size = int(pred.shape[-1])
    for truth, output in zip(target[:, 0].numpy(), pred[:, 0].numpy()):
        rows.append(np.concatenate([truth, np.zeros((size, 2), np.float32), output], axis=1))
    canvas = np.concatenate(rows, axis=0)
    Image.fromarray(np.rint(np.clip(canvas, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


def select(args) -> None:
    pack_path = Path(args.oof_pack)
    pack = torch.load(pack_path, map_location="cpu", weights_only=False)
    if not pack.get("object_level_disjoint") or pack.get("test_labels_evaluated"):
        raise RuntimeError("OOF pack does not satisfy the strict selection contract")
    forbidden = set(pack["forbidden_supervised_objects"])
    if forbidden & set(pack["objects"]):
        raise RuntimeError("forbidden target present in OOF refinement data")
    pred = pack["pred"].float()
    target = pack["target"].float()
    folds = list(pack["folds"])
    unique_folds = sorted(set(folds))
    if unique_folds != [1, 2, 3, 4] or len(folds) != len(pack["objects"]):
        raise RuntimeError(f"invalid OOF fold assignment: {folds}")

    ranking = []
    saved_predictions = {}
    for candidate_index, candidate in enumerate(CANDIDATES):
        held_out = []
        fold_rows = []
        for fold in unique_folds:
            train_indices = [index for index, value in enumerate(folds) if value != fold]
            val_indices = [index for index, value in enumerate(folds) if value == fold]
            model = train_refiner(
                candidate,
                pred[train_indices],
                target[train_indices],
                seed=2026090400 + 10 * candidate_index + fold,
            )
            with torch.no_grad():
                output = model(pred[val_indices])
            held_out.append((val_indices, output))
            fold_rows.append({
                "fold": fold,
                "objects": [pack["objects"][index] for index in val_indices],
                "metrics": image_metrics(output, target[val_indices]),
            })
        ordered = torch.empty_like(pred)
        for indices, output in held_out:
            ordered[indices] = output
        result = {
            "candidate": asdict(candidate),
            "cross_validation": image_metrics(ordered, target),
            "folds": fold_rows,
        }
        ranking.append(result)
        saved_predictions[candidate.name] = ordered
    ranking.sort(key=lambda row: (row["cross_validation"]["ssim"], -row["cross_validation"]["mse"]), reverse=True)

    out = Path(args.exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "selection_completed",
        "protocol": "outer object-disjoint four-fold CV on base-model OOF predictions",
        "selected": ranking[0]["candidate"],
        "ranking": ranking,
        "objects": pack["objects"],
        "folds": folds,
        "forbidden_supervised_objects": sorted(forbidden),
        "object_level_disjoint": True,
        "test_labels_evaluated": False,
        "oof_pack_sha256": sha256(pack_path),
    }
    (out / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save({"predictions": saved_predictions, "target": target, "objects": pack["objects"]}, out / "selection_predictions.pt")
    print(json.dumps(result, indent=2))


def final(args) -> None:
    selection_path = Path(args.selection_json)
    selection = json.loads(selection_path.read_text())
    if not selection.get("object_level_disjoint") or selection.get("test_labels_evaluated"):
        raise RuntimeError("refiner selection did not satisfy the strict protocol")
    selected = Candidate(**selection["selected"])
    if selected not in CANDIDATES:
        raise RuntimeError(f"selected candidate is not predeclared: {selected}")

    oof_path = Path(args.oof_pack)
    oof = torch.load(oof_path, map_location="cpu", weights_only=False)
    model = train_refiner(selected, oof["pred"].float(), oof["target"].float(), seed=2026090499)
    final_path = Path(args.final_pack)
    pack = torch.load(final_path, map_location="cpu", weights_only=False)
    with torch.no_grad():
        pred = model(pack["pred"].float())
        audit_pred = model(pack["audit_pred"].float())
    result = {
        "status": "completed",
        "selected": asdict(selected),
        "selection_json_sha256": sha256(selection_path),
        "oof_pack_sha256": sha256(oof_path),
        "input_pack_sha256": sha256(final_path),
        "checkpoint_selection": "fixed architecture and epochs from object-disjoint OOF CV",
        "test_used_for_selection": False,
        "primary_test": metric_pack(pred, pack["target"], pack["objects"]),
        "audit_ring_holdout": metric_pack(audit_pred, pack["audit_target"], pack["audit_objects"]),
    }
    out = Path(args.exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "test.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(
        {
            "model": model.state_dict(),
            "pred": pred,
            "target": pack["target"],
            "objects": pack["objects"],
            "audit_pred": audit_pred,
            "audit_target": pack["audit_target"],
            "audit_objects": pack["audit_objects"],
            "selected": asdict(selected),
        },
        out / "predictions.pt",
    )
    save_preview(out / "preview_target_pred.png", pred, pack["target"])
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["select", "final"], required=True)
    parser.add_argument("--oof-pack", required=True)
    parser.add_argument("--selection-json")
    parser.add_argument("--final-pack")
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.mode == "select":
        select(args)
    else:
        if not args.selection_json or not args.final_pack:
            parser.error("final mode requires --selection-json and --final-pack")
        final(args)


if __name__ == "__main__":
    main()
