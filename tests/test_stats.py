import random

from mvrag.stats import paired_bootstrap


def test_clear_gain_is_significant():
    rng = random.Random(1)
    a = [rng.random() for _ in range(400)]
    b = [x + 0.05 + rng.gauss(0, 0.02) for x in a]
    t = paired_bootstrap(a, b, n_resamples=2000)
    assert abs(t["diff"] - 0.05) < 0.01
    assert t["ci95"][0] > 0 and t["p"] < 0.01


def test_no_difference_is_not_significant():
    rng = random.Random(2)
    a = [rng.random() for _ in range(400)]
    b = [x + rng.gauss(0, 0.05) for x in a]
    t = paired_bootstrap(a, b, n_resamples=2000)
    assert t["ci95"][0] < 0 < t["ci95"][1] and t["p"] > 0.05


def test_groups_are_resampled_whole():
    t = paired_bootstrap([0, 0, 0, 0], [1, 1, 0, 0], groups=["q1", "q1", "q2", "q2"], n_resamples=500)
    assert t["groups"] == 2 and t["diff"] == 0.5
