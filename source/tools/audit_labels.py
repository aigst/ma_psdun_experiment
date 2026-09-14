#!/usr/bin/env python3
"""Write per-object GT geometry/intensity audits without training a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from new_sample_train import load_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--preprocess", default="detrend")
    parser.add_argument("--label-resample", default="bilinear")
    parser.add_argument("--label-crop", choices=["none", "center", "foreground"], default="center")
    parser.add_argument("--label-contrast", choices=["none", "minmax", "percentile"], default="percentile")
    parser.add_argument("--label-percentile-low", type=float, default=1.0)
    parser.add_argument("--label-percentile-high", type=float, default=99.0)
    args = parser.parse_args()
    _, samples = load_sample(
        args.data_root, args.patterns, args.size, args.preprocess, "none",
        args.label_resample, 127, 40.0, args.label_crop, args.label_contrast,
        args.label_percentile_low, args.label_percentile_high,
    )
    audits = {sample["object"]: sample["label_audit"] for sample in samples}
    Path(args.output).write_text(json.dumps({"config": vars(args), "objects": audits}, indent=2, ensure_ascii=False) + "\n")
    print({"object_count": len(audits), "output": args.output})


if __name__ == "__main__":
    main()
