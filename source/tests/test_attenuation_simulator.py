from pathlib import Path

import numpy as np
import torch

from domain_bridge_train import (
    preserve_multi_od_amplitude,
    save_reload_synthetic_dataset,
    validate_synthetic_dataset,
)
from ma_psdun.attenuation import (
    OD_DENSITIES,
    fit_attenuation_distribution,
    fit_exponential_curve,
    sample_joint_attenuation,
)
from ma_psdun.model import MultiODTCM


def _write_capture(path: Path, response: float, amplitude: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    phase = np.linspace(0.0, 4.0 * np.pi, 128, dtype=np.float64)
    dark = np.full(phase.shape, 0.25)
    raw = dark + response + amplitude * np.sin(phase)
    interleaved = np.empty(raw.size * 2)
    interleaved[0::2] = dark
    interleaved[1::2] = raw
    np.savetxt(path, interleaved)


def _make_object(root: Path, name: str, a: float, k: float, b: float) -> None:
    responses = a * np.exp(-k * OD_DENSITIES) + b
    for index, response in enumerate(responses):
        _write_capture(root / name / f"OD{index}" / "traindata.txt", float(response))


def test_constrained_exponential_fit_recovers_known_curve():
    expected = {"A": 0.78, "k": 2.15, "b": 0.22}
    responses = expected["A"] * np.exp(-expected["k"] * OD_DENSITIES) + expected["b"]
    fit = fit_exponential_curve(responses)
    assert fit["A"] > 0.0 and fit["k"] > 0.0 and fit["b"] >= 0.0
    assert abs(fit["A"] - expected["A"]) < 2e-3
    assert abs(fit["k"] - expected["k"]) < 2e-3
    assert abs(fit["b"] - expected["b"]) < 2e-3
    assert fit["r2"] > 0.99999


def test_fold_profile_uses_only_requested_objects_and_joint_bootstraps(tmp_path):
    _make_object(tmp_path, "train-a", 0.9, 1.8, 0.1)
    _make_object(tmp_path, "train-b", 0.65, 3.0, 0.35)
    _make_object(tmp_path, "validation", 0.8, 2.4, 0.2)
    _make_object(tmp_path, "test", 0.7, 2.1, 0.3)
    profile = fit_attenuation_distribution(tmp_path, ["train-a", "train-b"])
    assert profile["source_objects"] == ["train-a", "train-b"]
    assert {row["object"] for row in profile["fits"]} == {"train-a", "train-b"}
    fitted_source_objects = {
        Path(path).parent.parent.name
        for row in profile["fits"]
        for path in row["source_data"]
    }
    assert fitted_source_objects == {"train-a", "train-b"}

    generator = torch.Generator().manual_seed(7)
    sampled = sample_joint_attenuation(
        profile, 64, generator=generator, device="cpu", amplitude_jitter_log_std=0.0,
    )
    population = {
        (round(row["A"], 6), round(row["k"], 6), round(row["b"], 6))
        for row in profile["sampling_population"]
    }
    tuples = zip(sampled["A"], sampled["k"], sampled["b"])
    assert all(tuple(round(float(value), 6) for value in row) in population for row in tuples)


def test_shared_od_normalization_preserves_amplitude_ratios_and_legacy_default():
    waveform = torch.sin(torch.linspace(0.0, 4.0 * torch.pi, 257))
    amplitudes = torch.tensor([1.0, 0.6, 0.3, 0.1])
    raw = amplitudes[None, :, None] * waveform[None, None, :]
    dark = torch.zeros_like(raw)

    shared = MultiODTCM(preserve_od_amplitude=True)
    _, _, shared_signal = shared.normalize_captures(raw, dark)
    shared_std = shared_signal.std(dim=-1)[0]
    assert torch.allclose(shared_std / shared_std[0], amplitudes, atol=1e-5)

    legacy = MultiODTCM()
    _, _, legacy_signal = legacy.normalize_captures(raw, dark)
    assert torch.allclose(legacy_signal.std(dim=-1), torch.ones(1, 4), atol=1e-5)


def test_real_multi_od_preprocessing_uses_one_stack_scale(tmp_path):
    paths = []
    for index, amplitude in enumerate((1.0, 0.55, 0.28, 0.12)):
        path = tmp_path / "obj" / f"OD{index}" / "traindata.txt"
        _write_capture(path, response=2.0 * amplitude, amplitude=amplitude)
        paths.append(str(path))
    samples = [{
        "object": "obj",
        "source_data": paths,
        "raw": np.zeros((4, 128), dtype=np.float32),
        "dark": np.zeros((4, 128), dtype=np.float32),
        "label": np.zeros((64, 64), dtype=np.float32),
    }]
    converted = preserve_multi_od_amplitude(samples, "zscore", 31, 40.0)
    std = converted[0]["raw"].std(axis=1)
    assert np.allclose(std / std[0], [1.0, 0.55, 0.28, 0.12], atol=1e-5)


def test_saved_synthetic_dataset_reloads_exactly_and_obeys_curve(tmp_path):
    profile = {
        "schema": "ma-psdun-attenuation-exp-v1",
        "source_objects": ["train-a"],
        "sampling_population": [{"object": "train-a", "A": 0.8, "k": 2.0, "b": 0.2}],
    }
    a = torch.tensor([0.8, 0.8])
    k = torch.tensor([2.0, 2.0])
    b = torch.tensor([0.2, 0.2])
    response = a[:, None] * torch.exp(-k[:, None] * torch.tensor(OD_DENSITIES).float()) + b[:, None]
    dataset = {
        "schema": "ma-psdun-synthetic-exp-attenuation-v1",
        "seed": 1,
        "count": 2,
        "train_objects": ["train-a"],
        "raw": torch.rand(2, 4, 16),
        "dark": torch.rand(2, 4, 16),
        "target": torch.rand(2, 1, 4, 4),
        "cond": torch.ones(2, 3),
        "A": a,
        "k": k,
        "b": b,
        "od_responses": response,
        "attenuation_source_index": torch.zeros(2, dtype=torch.long),
        "attenuation_source_object": ["train-a", "train-a"],
        "target_source_object": ["train-a", "procedural"],
    }
    validate_synthetic_dataset(dataset, profile)
    reloaded, audit = save_reload_synthetic_dataset(
        dataset, tmp_path / "synthetic_dataset.pt", profile,
    )
    assert audit["exact_tensor_reload"] is True
    assert audit["count"] == 2
    assert torch.equal(reloaded["raw"], dataset["raw"])
