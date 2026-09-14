from ring_blind_train import validate_split


def test_20260903_split_is_object_disjoint_and_ring_blind():
    train = [
        "bar1-20260830", "bar10-20260831", "bar2-20260830",
        "bar3-20260830", "bar4-20260830", "bar5-20260830",
        "bar7-20260831", "bar8-20260831", "bar9-20260831",
        "obj11-20260901", "obj14-20260901", "obj22-20260902",
        "obj24-20260903", "obj25-20260903",
    ]
    validation = ["bar6-20260831", "obj13-20260901", "obj17-20260901"]
    test = ["obj18-20260901", "obj19-20260901", "obj20-20260901"]
    audit = ["obj15-20260901", "obj16-20260901"]
    forbidden = ["obj15-20260901", "obj16-20260901", "obj19-20260901"]
    available = sorted(set(train + validation + test + audit))

    result = validate_split(
        available, train, validation, test, audit, forbidden, True,
    )

    assert result["object_level_disjoint"] is True
    assert result["ring_blind_supervision"] is True
    assert result["forbidden_intersection"] == {"train": [], "validation": []}

