#!/usr/bin/env python3
"""Create a label-free adjoint baseline pack for protocol smoke tests."""

from __future__ import annotations

import argparse

import torch

from ma_psdun.core import MeasurementOperator
from self_supervised_adapt import load_measurements_target_free


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    objects = [x.strip() for x in args.objects.split(",") if x.strip()]
    patterns, measured = load_measurements_target_free(args.data_root, args.patterns, objects)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    op = MeasurementOperator(patterns.shape[1], patterns.shape[0], patterns=torch.from_numpy(patterns), device=device)
    with torch.no_grad():
        pred = op.adjoint(torch.from_numpy(measured).to(device) * 0.35).reshape(-1, 1, args.size, args.size)
        pred = (pred - pred.amin(dim=(-2, -1), keepdim=True)) / (
            pred.amax(dim=(-2, -1), keepdim=True) - pred.amin(dim=(-2, -1), keepdim=True)
        ).clamp_min(1e-6)
    torch.save({"pred": pred.cpu(), "objects": objects}, args.output)
    print({"objects": len(objects), "output": args.output})


if __name__ == "__main__":
    main()
