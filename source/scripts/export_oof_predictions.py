"""Export object-disjoint validation predictions from trained CV folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import torch

from ring_blind_train import evaluate_objects, make_model, prepare


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-dirs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    directories = [Path(value) for value in args.selection_dirs.split(",") if value]
    if len(directories) != 4:
        raise ValueError(f"expected four selection directories, got {len(directories)}")

    predictions = []
    targets = []
    objects = []
    folds = []
    seen = set()
    forbidden = None
    for fold, directory in enumerate(directories, start=1):
        config = json.loads((directory / "config.json").read_text())
        config["device"] = args.device
        fold_args = Namespace(**config)
        device, samples, split, operator, od_channels = prepare(fold_args)
        if any(split["forbidden_intersection"].values()):
            raise RuntimeError(f"forbidden supervision in fold {fold}: {split['forbidden_intersection']}")
        overlap = seen & set(split["validation"])
        if overlap:
            raise RuntimeError(f"validation objects repeated across folds: {sorted(overlap)}")
        seen.update(split["validation"])
        current_forbidden = set(split["forbidden_supervised_objects"])
        if forbidden is None:
            forbidden = current_forbidden
        elif forbidden != current_forbidden:
            raise RuntimeError("forbidden-object sets differ across folds")

        model = make_model(operator, fold_args, od_channels).to(device)
        checkpoint_path = directory / "checkpoint_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        pack = evaluate_objects(model, samples, split["validation"], device, operator, fold_args)
        predictions.append(pack["pred"])
        targets.append(pack["target"])
        objects.extend(pack["objects"])
        folds.extend([fold] * len(pack["objects"]))
        del model

    if seen & (forbidden or set()):
        raise RuntimeError(f"forbidden objects present in OOF set: {sorted(seen & (forbidden or set()))}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pred": torch.cat(predictions),
            "target": torch.cat(targets),
            "objects": objects,
            "folds": folds,
            "selection_dirs": [str(path) for path in directories],
            "checkpoint_sha256": {str(path): sha256(path / "checkpoint_best.pt") for path in directories},
            "forbidden_supervised_objects": sorted(forbidden or []),
            "object_level_disjoint": True,
            "test_labels_evaluated": False,
        },
        output,
    )
    print(json.dumps({"output": str(output), "objects": objects, "folds": folds}, indent=2))


if __name__ == "__main__":
    main()
