from pathlib import Path
import numpy as np
from PIL import Image
import torch


def load_real_dataset(root, pattern_tif, label_name="SI_AP.png"):
    root = Path(root)
    tif = Image.open(pattern_tif)
    frames = getattr(tif, "n_frames", 1)
    pattern_frames = []
    for i in range(frames):
        tif.seek(i)
        frame = np.asarray(tif, dtype=np.float32)
        # DMD files are commonly encoded as either 0/1 or 0/255.  Convert
        # both forms to the physical binary illumination values and never
        # reinterpret a positive-only pattern as a signed Hadamard pattern.
        lo, hi = float(frame.min()), float(frame.max())
        if hi <= lo:
            binary = np.full_like(frame, 1.0 if hi > 0 else 0.0, dtype=np.float32)
        else:
            binary = (frame > (lo + hi) * 0.5).astype(np.float32)
        pattern_frames.append(binary.reshape(-1))
    patterns = np.stack(pattern_frames)
    samples = []
    for obj in sorted(root.glob("bar*-20260830")):
        for od in sorted(obj.glob("OD*")):
            train = np.loadtxt(od / "traindata.txt", dtype=np.float32)
            assert train.size % 2 == 0
            dark, raw = train[0::2], train[1::2]
            assert len(raw) == len(patterns), (obj.name, od.name, len(raw), len(patterns))
            y = raw - dark
            y = (y - y.mean()) / max(float(y.std()), 1e-8)
            label = np.asarray(Image.open(obj / "SI" / label_name).convert("L"), dtype=np.float32) / 255.0
            samples.append({"object": obj.name, "od": od.name, "y_raw": raw, "y_dark": dark, "y": y, "label": label})
    return patterns, samples


def to_tensors(patterns, samples, device="cpu"):
    A = torch.from_numpy(patterns).to(device)
    y_raw = torch.from_numpy(np.stack([s["y_raw"] for s in samples])).to(device)
    y_dark = torch.from_numpy(np.stack([s["y_dark"] for s in samples])).to(device)
    labels = torch.from_numpy(np.stack([s["label"] for s in samples]))[:, None].to(device)
    return A, y_raw, y_dark, labels
