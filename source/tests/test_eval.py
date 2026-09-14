import torch

from ma_psdun.eval import image_metrics


def test_image_metrics_have_expected_values_and_finite_outputs():
    target = torch.zeros(2, 1, 4, 4)
    pred = target.clone()
    pred[0, 0, 0, 0] = 1.0
    metrics = image_metrics(pred, target)
    assert set(metrics) == {"mse", "psnr", "ssim"}
    assert metrics["mse"] > 0
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()


def test_identical_images_have_zero_mse_and_perfect_ssim():
    x = torch.rand(2, 1, 8, 8)
    metrics = image_metrics(x, x)
    assert metrics["mse"] == 0.0
    assert metrics["ssim"] > 0.999


def test_ssim_is_computed_per_image_not_across_batch():
    target = torch.zeros(2, 1, 8, 8)
    pred = target.clone()
    pred[0] = 1.0
    metrics = image_metrics(pred, target)
    assert 0.0 < metrics["ssim"] < 1.0
