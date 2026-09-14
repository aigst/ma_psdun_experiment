#!/usr/bin/env python3
"""Prepare the 2026-09-09 sample export for the MA-PSDUN loaders.

The source export is kept immutable.  Each output object receives an ASCII
identifier, ``DAQrawdata.txt`` is copied to the legacy ``traindata.txt``
name, and ``gt.png`` is copied to ``ground_truth.png`` when present.  The
manifest records the original names and validates the dark/bright and
pre-differenced image contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


OD_NAMES = ("OD0", "OD1", "OD2", "OD3")
OD_VALUES = {"OD0": 0.0, "OD1": 0.3, "OD2": 0.5, "OD3": 1.0}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_array(path: Path, values: np.ndarray) -> None:
    np.savetxt(path, values.astype(np.float32), fmt="%.9g")


def prepare(source: Path, output: Path) -> dict:
    source = source.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_pattern = source / "pattern.tif"
    if not source_pattern.exists():
        raise FileNotFoundError(source_pattern)
    with Image.open(source_pattern) as tif:
        frames = getattr(tif, "n_frames", 1)
        shape = tuple(np.asarray(tif).shape)
    if frames != 4096 or shape != (64, 64):
        raise ValueError(f"unexpected pattern: frames={frames}, shape={shape}")
    pattern_out = output / "pattern.tif"
    shutil.copy2(source_pattern, pattern_out)

    objects = [p for p in source.iterdir() if p.is_dir() and any((p / od).is_dir() for od in OD_NAMES)]
    objects.sort(key=lambda p: p.name)
    records = []
    for index, obj in enumerate(objects, 1):
        object_id = f"sample_{index:02d}"
        dest = output / object_id
        dest.mkdir(exist_ok=True)
        truth = obj / "gt.png"
        if truth.exists():
            shutil.copy2(truth, dest / "ground_truth.png")
        od_records = []
        for od in OD_NAMES:
            od_src = obj / od
            if not od_src.is_dir():
                raise FileNotFoundError(od_src)
            raw_path = od_src / "DAQrawdata.txt"
            image_path = od_src / "imagedata.txt"
            raw = np.loadtxt(raw_path, dtype=np.float64).reshape(-1)
            image = np.loadtxt(image_path, dtype=np.float64).reshape(-1)
            if raw.size != 8192 or image.size != 4096:
                raise ValueError(f"{obj.name}/{od}: raw={raw.size}, image={image.size}")
            dark, bright = raw[0::2], raw[1::2]
            delta = bright - dark
            max_abs = float(np.max(np.abs(delta - image)))
            if max_abs > 2e-4:
                raise ValueError(f"{obj.name}/{od}: imagedata mismatch max_abs={max_abs}")
            od_dest = dest / od
            od_dest.mkdir(exist_ok=True)
            shutil.copy2(raw_path, od_dest / "traindata.txt")
            # Keep the supplied pre-differenced sequence for later diagnostics.
            shutil.copy2(image_path, od_dest / "imagedata.txt")
            od_records.append({
                "od": od,
                "optical_density": OD_VALUES[od],
                "raw_values": int(raw.size),
                "pairs": int(dark.size),
                "imagedata_values": int(image.size),
                "imagedata_max_abs_error": max_abs,
                "raw_sha256": sha256(raw_path),
                "imagedata_sha256": sha256(image_path),
            })
        records.append({
            "object_id": object_id,
            "source_name": obj.name,
            "labelled": truth.exists(),
            "source_truth": str(truth) if truth.exists() else None,
            "od": od_records,
        })
    manifest = {
        "source_root": str(source),
        "output_root": str(output),
        "pattern": {"path": str(pattern_out), "frames": frames, "shape": shape, "sha256": sha256(pattern_out)},
        "od_values": OD_VALUES,
        "object_count": len(records),
        "labelled_count": sum(r["labelled"] for r in records),
        "unlabelled_count": sum(not r["labelled"] for r in records),
        "objects": records,
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    manifest = prepare(Path(args.source), Path(args.output))
    print(json.dumps({
        "output": manifest["output_root"],
        "pattern_sha256": manifest["pattern"]["sha256"],
        "object_count": manifest["object_count"],
        "labelled_count": manifest["labelled_count"],
        "unlabelled_count": manifest["unlabelled_count"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
