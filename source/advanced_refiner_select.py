"""OOF selection for small supervised refiners on frozen MA-PSDUN outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ma_psdun.eval import image_metrics
from ma_psdun.model import _ssim_loss


@dataclass(frozen=True)
class Candidate:
    name: str
    channels: int
    epochs: int
    lr: float
    residual_scale: float
    bce_weight: float
    edge_weight: float
    logit_mode: bool


CANDIDATES = (
    Candidate("residual16", 16, 300, 0.002, 0.25, 0.0, 0.02, False),
    Candidate("residual16_bce", 16, 300, 0.002, 0.25, 0.10, 0.02, False),
    Candidate("residual32_bce", 32, 300, 0.001, 0.20, 0.10, 0.02, False),
    Candidate("logit8", 8, 300, 0.002, 0.30, 0.0, 0.02, True),
    Candidate("logit16_bce", 16, 350, 0.001, 0.25, 0.10, 0.02, True),
    Candidate("logit16_edge", 16, 350, 0.001, 0.25, 0.05, 0.08, True),
    Candidate("multiscale16_bce", 16, 350, 0.001, 0.25, 0.10, 0.02, True),
)


class ResidualRefiner(nn.Module):
    def __init__(self, channels: int, scale: float, logit_mode: bool, multiscale: bool = False):
        super().__init__()
        self.scale = scale
        self.logit_mode = logit_mode
        in_channels = 3 if multiscale else 1
        self.multiscale = multiscale
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(channels, 1, 3, padding=1, padding_mode="reflect"),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multiscale:
            features = torch.cat([x, F.avg_pool2d(x, 3, 1, 1), F.avg_pool2d(x, 5, 1, 2)], dim=1)
        else:
            features = x
        delta = self.body(features)
        if self.logit_mode:
            base = torch.logit(x.clamp(1e-4, 1.0 - 1e-4))
            return torch.sigmoid(base + self.scale * torch.tanh(delta))
        return (x + self.scale * torch.tanh(delta)).clamp(0.0, 1.0)


def make_model(candidate: Candidate) -> nn.Module:
    return ResidualRefiner(
        candidate.channels,
        candidate.residual_scale,
        candidate.logit_mode,
        multiscale=candidate.name.startswith("multiscale"),
    )


def dihedral(x: torch.Tensor, transform: int) -> torch.Tensor:
    if transform >= 4:
        x = x.flip(-1)
        transform -= 4
    return torch.rot90(x, transform, (-2, -1))


def loss_fn(output: torch.Tensor, target: torch.Tensor, candidate: Candidate) -> torch.Tensor:
    edge = F.l1_loss(output[:, :, 1:] - output[:, :, :-1], target[:, :, 1:] - target[:, :, :-1])
    edge = edge + F.l1_loss(output[:, :, :, 1:] - output[:, :, :, :-1], target[:, :, :, 1:] - target[:, :, :, :-1])
    bce = F.binary_cross_entropy(output.clamp(1e-4, 1.0 - 1e-4), target)
    return (F.l1_loss(output, target) + 0.5 * F.mse_loss(output, target)
            + 0.2 * _ssim_loss(output, target) + candidate.edge_weight * edge
            + candidate.bce_weight * bce)


def train(candidate: Candidate, pred: torch.Tensor, target: torch.Tensor, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    model = make_model(candidate)
    if candidate.epochs == 0:
        return model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=candidate.lr, weight_decay=1e-4)
    best = float("inf")
    best_state = None
    for epoch in range(candidate.epochs):
        model.train()
        transform = epoch % 8
        output = model(dihedral(pred, transform))
        loss = loss_fn(output, dihedral(target, transform), candidate)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if float(loss.detach()) < best:
            best = float(loss.detach())
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-pack", required=True)
    parser.add_argument("--selection-pack", required=True)
    parser.add_argument("--final-pack", required=True)
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.exp_dir)
    out.mkdir(parents=True, exist_ok=True)
    oof = torch.load(args.oof_pack, map_location="cpu", weights_only=False)
    selection_pack = torch.load(args.selection_pack, map_location="cpu", weights_only=False)
    final = torch.load(args.final_pack, map_location="cpu", weights_only=False)
    pred = selection_pack["predictions"]["conv3"].float()
    target = selection_pack["target"].float()
    objects = list(selection_pack["objects"])
    folds = list(oof["folds"])
    if oof.get("test_labels_evaluated") or not oof.get("object_level_disjoint"):
        raise RuntimeError("OOF contract failed")
    ranking = []
    for index, candidate in enumerate(CANDIDATES):
        held_out = torch.empty_like(pred)
        fold_rows = []
        for fold in sorted(set(folds)):
            train_indices = [i for i, f in enumerate(folds) if f != fold]
            val_indices = [i for i, f in enumerate(folds) if f == fold]
            model = train(candidate, pred[train_indices], target[train_indices], 2026090800 + index * 17 + fold)
            with torch.no_grad():
                held_out[val_indices] = model(pred[val_indices])
            fold_rows.append({
                "fold": fold,
                "objects": [objects[i] for i in val_indices],
                "metrics": image_metrics(held_out[val_indices], target[val_indices]),
            })
        overall = image_metrics(held_out, target)
        ranking.append({"candidate": asdict(candidate), "cross_validation": overall, "folds": fold_rows})
        print(json.dumps({"candidate": candidate.name, "oof_ssim": overall["ssim"]}), flush=True)
    ranking.sort(key=lambda row: (row["cross_validation"]["ssim"], -row["cross_validation"]["mse"]), reverse=True)
    selected = Candidate(**ranking[0]["candidate"])
    final_model = train(selected, pred, target, 2026090899)
    with torch.no_grad():
        test_pred = final_model(final["pred"].float())
        audit_pred = final_model(final["audit_pred"].float())
    result = {
        "status": "completed",
        "protocol": "object-disjoint four-fold OOF selection on frozen conv3 predictions",
        "selected": asdict(selected),
        "ranking": ranking,
        "objects": objects,
        "folds": folds,
        "forbidden_supervised_objects": ["obj15-20260901", "obj16-20260901", "obj19-20260901"],
        "test_used_for_selection": False,
        "primary_test_base": {"overall": image_metrics(final["pred"], final["target"])},
        "primary_test": {"overall": image_metrics(test_pred, final["target"]), "by_object": [
            {"object": n, "metrics": image_metrics(test_pred[i:i + 1], final["target"][i:i + 1])}
            for i, n in enumerate(final["objects"])
        ]},
        "audit_ring_holdout_base": {"overall": image_metrics(final["audit_pred"], final["audit_target"])},
        "audit_ring_holdout": {"overall": image_metrics(audit_pred, final["audit_target"]), "by_object": [
            {"object": n, "metrics": image_metrics(audit_pred[i:i + 1], final["audit_target"][i:i + 1])}
            for i, n in enumerate(final["audit_objects"])
        ]},
        "sha256": {"oof_pack": sha256(Path(args.oof_pack)), "final_pack": sha256(Path(args.final_pack))},
    }
    (out / "advanced_refiner.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    torch.save({"pred": test_pred, "target": final["target"], "objects": final["objects"],
                "audit_pred": audit_pred, "audit_target": final["audit_target"],
                "audit_objects": final["audit_objects"], "selected": asdict(selected)}, out / "predictions.pt")
    print(json.dumps({"selected": asdict(selected), "oof_ssim": ranking[0]["cross_validation"]["ssim"],
                      "test_ssim": result["primary_test"]["overall"]["ssim"],
                      "audit_ssim": result["audit_ring_holdout"]["overall"]["ssim"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
