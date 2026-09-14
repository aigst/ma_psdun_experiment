"""Audited ground-truth preprocessing for the 64x64 reconstruction grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _foreground_mask(image: np.ndarray, threshold: float | None = None) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    if threshold is None:
        threshold = max(0.05, float(np.quantile(values, 0.75)))
    mask = values > threshold
    if int(mask.sum()) < max(4, values.size // 10000):
        mask = values > float(values.mean())
    return mask


def foreground_bbox(image: np.ndarray, threshold: float | None = None) -> tuple[int, int, int, int]:
    mask = _foreground_mask(image, threshold)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, int(image.shape[1]), int(image.shape[0]))
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def audit_label(image: np.ndarray, threshold: float | None = None) -> dict:
    """Return geometry and intensity diagnostics without modifying the label."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"label must be 2-D grayscale, got {values.shape}")
    height, width = values.shape
    x0, y0, x1, y1 = foreground_bbox(values, threshold)
    mask = _foreground_mask(values, threshold)
    ys, xs = np.where(mask)
    if len(xs) >= 2:
        coords = np.stack([xs - xs.mean(), ys - ys.mean()], axis=1)
        covariance = np.cov(coords, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        principal = eigenvectors[:, int(np.argmax(eigenvalues))]
        angle = float(np.degrees(np.arctan2(principal[1], principal[0])))
        centroid = [float(xs.mean() / max(width - 1, 1)), float(ys.mean() / max(height - 1, 1))]
    else:
        angle = 0.0
        centroid = [0.5, 0.5]
    return {
        "original_size": [int(width), int(height)],
        "foreground_bbox_xyxy": [x0, y0, x1, y1],
        "foreground_fraction": float(mask.mean()),
        "bbox_fraction": float((x1 - x0) * (y1 - y0) / max(width * height, 1)),
        "bbox_aspect_ratio": float((x1 - x0) / max(y1 - y0, 1)),
        "foreground_centroid_normalized": centroid,
        "center_offset_normalized": [float(centroid[0] - 0.5), float(centroid[1] - 0.5)],
        "touches_border": bool(x0 == 0 or y0 == 0 or x1 == width or y1 == height),
        "principal_axis_angle_deg": angle,
        "intensity_min": float(values.min()),
        "intensity_max": float(values.max()),
        "intensity_p01": float(np.quantile(values, 0.01)),
        "intensity_p99": float(np.quantile(values, 0.99)),
    }


def _square_crop(image: np.ndarray, crop: str, margin: float) -> tuple[np.ndarray, list[int]]:
    height, width = image.shape
    if crop == "none":
        return image, [0, 0, width, height]
    if crop == "center":
        side = min(height, width)
        x0 = (width - side) // 2
        y0 = (height - side) // 2
        return image[y0:y0 + side, x0:x0 + side], [x0, y0, x0 + side, y0 + side]
    if crop != "foreground":
        raise ValueError(f"unknown label crop mode: {crop}")
    x0, y0, x1, y1 = foreground_bbox(image)
    side = max(x1 - x0, y1 - y0, 1)
    side = int(np.ceil(side * (1.0 + 2.0 * max(margin, 0.0))))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    left, top = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    right, bottom = left + side, top + side
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - width), max(0, bottom - height)
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
        left += pad_left
        top += pad_top
    return image[top:top + side, left:left + side], [left - pad_left, top - pad_top, right - pad_left, bottom - pad_top]


def preprocess_label(
    path: str | Path,
    size: int,
    *,
    resample: str = "lanczos",
    crop: str = "center",
    contrast: str = "percentile",
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    foreground_margin: float = 0.05,
) -> tuple[np.ndarray, dict]:
    """Audit, crop, contrast-normalize and resize a label deterministically."""
    resampling = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    if resample not in resampling:
        raise ValueError(f"unknown label resampling: {resample}")
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    audit = audit_label(image)
    cropped, crop_box = _square_crop(image, crop, foreground_margin)
    if contrast == "none":
        normalized, contrast_bounds = cropped, [0.0, 1.0]
    elif contrast == "minmax":
        lo, hi = float(cropped.min()), float(cropped.max())
        normalized, contrast_bounds = (cropped - lo) / max(hi - lo, 1e-8), [lo, hi]
    elif contrast == "percentile":
        if not 0 <= percentile_low < percentile_high <= 100:
            raise ValueError("percentile bounds must satisfy 0 <= low < high <= 100")
        lo, hi = [float(x) for x in np.percentile(cropped, [percentile_low, percentile_high])]
        normalized, contrast_bounds = np.clip((cropped - lo) / max(hi - lo, 1e-8), 0.0, 1.0), [lo, hi]
    else:
        raise ValueError(f"unknown label contrast mode: {contrast}")
    pil = Image.fromarray(np.rint(np.clip(normalized, 0, 1) * 255).astype(np.uint8), mode="L")
    resized = np.asarray(pil.resize((size, size), resampling[resample]), dtype=np.float32) / 255.0
    audit.update({
        "crop_mode": crop,
        "crop_box_xyxy": crop_box,
        "cropped_size": [int(cropped.shape[1]), int(cropped.shape[0])],
        "contrast_mode": contrast,
        "contrast_bounds": contrast_bounds,
        "output_size": [int(size), int(size)],
        "resample": resample,
        "output_min": float(resized.min()),
        "output_max": float(resized.max()),
    })
    return resized, audit


__all__ = ["foreground_bbox", "audit_label", "preprocess_label"]
