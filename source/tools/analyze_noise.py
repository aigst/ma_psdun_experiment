#!/usr/bin/env python3
"""Build a JSON empirical-noise profile from an MA-PSDUN data export."""

from __future__ import annotations

import argparse

from ma_psdun.noise import build_noise_profile, save_noise_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--max-sequences-per-od", type=int, default=0)
    args = parser.parse_args()
    profile = build_noise_profile(
        args.data_root,
        sample_rate=args.sample_rate,
        include_sequences=True,
        max_sequences_per_od=(args.max_sequences_per_od or None),
    )
    save_noise_profile(profile, args.output)
    print({"output": args.output, "file_count": profile["file_count"], "ods": sorted(profile["by_od"])})


if __name__ == "__main__":
    main()
