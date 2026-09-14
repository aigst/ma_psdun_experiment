import torch
from pathlib import Path

from ma_psdun.core import structured_images
from new_sample_train import _truth_file


def test_structured_images_are_spatially_correlated_and_bounded():
    x = structured_images(4, 32, seed=9)
    assert x.shape == (4, 32 * 32)
    assert 0.0 <= float(x.min()) <= float(x.max()) <= 1.0
    img = x.reshape(4, 32, 32)
    neighbor_corr = torch.corrcoef(torch.stack([img[:, :, :-1].flatten(), img[:, :, 1:].flatten()]))[0, 1]
    assert float(neighbor_corr) > 0.2


def test_truth_file_accepts_export_typo_variants(tmp_path: Path):
    (tmp_path / "groud trurth.png").touch()
    assert _truth_file(tmp_path).name == "groud trurth.png"
    (tmp_path / "groud trurh.png").touch()
    # Deterministic sorting keeps the selection stable when an object contains
    # more than one accidental label spelling.
    assert _truth_file(tmp_path).name == "groud trurh.png"
