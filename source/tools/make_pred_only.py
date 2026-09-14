#!/usr/bin/env python3
"""Strip evaluation targets from a prediction pack before target-free adaptation."""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pack = torch.load(args.input, map_location="cpu", weights_only=False)
    if "pred" not in pack or "objects" not in pack:
        raise ValueError("input pack must contain pred and objects")
    torch.save({"pred": pack["pred"].cpu(), "objects": list(pack["objects"])}, args.output)
    print({"objects": list(pack["objects"]), "output": args.output})


if __name__ == "__main__":
    main()
