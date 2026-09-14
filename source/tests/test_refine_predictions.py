import torch

from refine_predictions import CANDIDATES, make_refiner, train_refiner


def test_all_refiners_start_as_identity_on_valid_predictions():
    image = torch.rand(2, 1, 16, 16) * 0.8 + 0.1
    for name in ("identity", "affine", "conv3", "multiscale", "residual8"):
        output = make_refiner(name)(image)
        assert torch.allclose(output, image, atol=1e-6), name


def test_identity_candidate_has_no_training_side_effect():
    candidate = next(item for item in CANDIDATES if item.name == "identity")
    pred = torch.rand(2, 1, 8, 8)
    target = torch.rand(2, 1, 8, 8)
    model = train_refiner(candidate, pred, target, seed=3)
    assert torch.equal(model(pred), pred)
