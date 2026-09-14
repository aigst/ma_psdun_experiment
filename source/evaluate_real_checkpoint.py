"""Export predictions for every labelled real object from a checkpoint.

This is used for validation-only ensemble weighting.  Labels are always loaded
with the common BILINEAR 64x64 convention, independent of the training label
resampler, so all checkpoints share one metric target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ma_psdun.core import MeasurementOperator
from ma_psdun.model import MAPSDUN
from new_sample_train import _batch, load_sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--patterns", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    exp = Path(a.exp_dir)
    cfg = json.loads((exp / "config.json").read_text())
    size = int(cfg.get("size", 64))
    patterns, samples = load_sample(
        a.data_root, a.patterns, size,
        cfg.get("preprocess", "detrend"), cfg.get("fusion", "mean"),
        "bilinear", int(cfg.get("detrend_width", 31)), float(cfg.get("gauss_sigma", 40.0)),
    )
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    op = MeasurementOperator(patterns.shape[1], patterns.shape[0],
                             patterns=torch.from_numpy(patterns), device=device)
    od_channels = int(samples[0]["raw"].shape[0]) if samples[0]["raw"].ndim == 2 else 1
    model = MAPSDUN(
        op, stages=int(cfg.get("stages", 8)),
        backprojection_gain_init=float(cfg.get("gain_init", 4.0)),
        lowpass_kernel=int(cfg.get("lowpass_kernel", 3)),
        od_channels=od_channels,
        prior_residual_scale=float(cfg.get("prior_residual_scale", 0.05)),
        shared_prior=bool(cfg.get("shared_prior", False)),
    ).to(device)
    ck_name = "checkpoint_latest.pt" if bool(cfg.get("train_all_nontest", False)) else "checkpoint_best.pt"
    ck = torch.load(exp / ck_name, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    idx = list(range(len(samples)))
    raw, dark, target, cond = _batch(samples, idx, device, 0, cfg.get("condition_mode", "constant"))
    with torch.no_grad():
        pred, y = model(raw, dark, cond, (size, size))
    payload = {
        "pred": pred.cpu(), "target": target.cpu(), "measurement": y.cpu(),
        "objects": [samples[i]["object"] for i in idx],
        "ods": [samples[i]["od"] for i in idx],
        "checkpoint": str(exp / ck_name), "checkpoint_epoch": int(ck["epoch"]),
        "config": cfg,
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, a.output)
    print(json.dumps({"output": a.output, "samples": len(idx), "checkpoint_epoch": int(ck["epoch"])}))


if __name__ == "__main__":
    main()
