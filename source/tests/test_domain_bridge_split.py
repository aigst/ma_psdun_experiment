from domain_bridge_train import split_folds
from new_dataset_domain_bridge_train import (
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    DEFAULT_UNLABELLED,
    DEFAULT_VALIDATION,
)
from ring_blind_train import parse_names, validate_split


def test_deleted_object_protocol_uses_two_locked_tests_and_five_object_folds():
    objects = [f"obj{i}-20260901" for i in range(1, 23)]
    tests = ["obj18-20260901", "obj20-20260901"]
    folds = split_folds(objects, seed=20260902, test_objects=tests, n_folds=4)
    assert [len(fold) for fold in folds] == [5, 5, 5, 5]
    assert set().union(*map(set, folds)) == set(objects) - set(tests)
    assert not any(set(fold) & set(tests) for fold in folds)
    assert sum(len(set(a) & set(b)) for i, a in enumerate(folds) for b in folds[i + 1:]) == 0


def test_split_rejects_missing_locked_test_object():
    try:
        split_folds(["obj18-20260901"], test_objects=["obj18-20260901", "obj20-20260901"])
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing test object was accepted")


def test_new_dataset_domain_bridge_has_disjoint_locked_contract():
    train = parse_names(DEFAULT_TRAIN)
    validation = parse_names(DEFAULT_VALIDATION)
    test = parse_names(DEFAULT_TEST)
    unlabelled = parse_names(DEFAULT_UNLABELLED)
    available = sorted(train + validation + test + unlabelled)
    result = validate_split(
        available, train, validation, test, unlabelled, unlabelled, True,
    )
    assert result["object_level_disjoint"] is True
    assert len(train) == 12 and len(validation) == 5 and len(test) == 4
    assert len(unlabelled) == 4
    assert not (set(train) & (set(validation) | set(test) | set(unlabelled)))
