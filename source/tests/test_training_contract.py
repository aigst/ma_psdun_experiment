from train import split_ids


def test_split_ids_are_deterministic_and_disjoint():
    a = split_ids(n=100, seed=11)
    b = split_ids(n=100, seed=11)
    assert a == b
    assert len(a['train']) == 80
    assert len(a['val']) == 10
    assert len(a['test']) == 10
    assert not set(a['train']) & set(a['val'])
    assert not set(a['train']) & set(a['test'])
    assert not set(a['val']) & set(a['test'])
