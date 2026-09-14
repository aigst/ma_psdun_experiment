import numpy as np
import torch

from ma_psdun.noise import (
    analyze_dark_sequence,
    build_noise_profile,
    load_dark_sequence,
    sample_noise_sequences,
)


def test_dark_analysis_reports_differenced_noise_and_psd():
    rng = np.random.default_rng(3)
    sequence = (0.2 + np.linspace(0.0, 0.05, 4096) + rng.normal(0, 0.01, 4096)).astype(np.float32)
    result = analyze_dark_sequence(sequence)
    assert result["count"] == 4096
    assert result["adjacent_diff_std"] > 0
    assert result["white_noise_std_estimate"] > 0
    assert 0 <= result["low_frequency_power_fraction"] <= 1
    assert len(result["normalized_residual"]) == 4096


def test_build_profile_and_replay_are_deterministic(tmp_path):
    root = tmp_path / "sample" / "obj1" / "OD1"
    root.mkdir(parents=True)
    values = np.empty(16, dtype=np.float32)
    values[0::2] = np.linspace(0.1, 0.2, 8)
    values[1::2] = values[0::2] + 0.03
    np.savetxt(root / "traindata.txt", values)
    assert load_dark_sequence(root / "traindata.txt").shape == (8,)
    profile = build_noise_profile(tmp_path, include_sequences=True)
    generator_a = torch.Generator(device="cpu").manual_seed(9)
    generator_b = torch.Generator(device="cpu").manual_seed(9)
    a = sample_noise_sequences(profile, 2, 5, generator=generator_a, device="cpu", od_labels=["OD1", "OD1"])
    b = sample_noise_sequences(profile, 2, 5, generator=generator_b, device="cpu", od_labels=["OD1", "OD1"])
    assert a.shape == (2, 5)
    assert torch.equal(a, b)


def test_noise_profile_can_be_restricted_to_fold_training_objects(tmp_path):
    for name, offset in (("train", 0.0), ("validation", 0.1), ("test", 0.2)):
        path = tmp_path / "sample" / name / "OD0"
        path.mkdir(parents=True)
        values = np.empty(32, dtype=np.float32)
        values[0::2] = np.linspace(0.1 + offset, 0.2 + offset, 16)
        values[1::2] = values[0::2] + 0.03
        np.savetxt(path / "traindata.txt", values)
    profile = build_noise_profile(
        tmp_path, include_sequences=True, include_objects=["train"],
    )
    assert profile["file_count"] == 1
    assert profile["source_objects"] == ["train"]
    assert profile["by_od"]["OD0"]["count"] == 1
    assert all("/train/" in source for source in profile["by_od"]["OD0"]["sources"])
