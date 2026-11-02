"""E14, E16 and E17 on hand made calibration sets: the finite sample quantile, the block weights, the p value and the
Bonferroni and Holm verdicts."""
import math

import numpy as np

from servers.causal_diagnosis import conformal as co


def test_the_quantile_is_the_finite_sample_order_statistic_with_unit_weights():
    n = 19
    scores = np.arange(1, n + 1, dtype=float)[:, None]
    calibration = co.Calibration(scores, np.ones(n), n_blocks=n)
    for alpha in (0.05, 0.1, 0.5):
        q, capped = calibration.quantile(0, alpha)
        assert not capped and q == math.ceil((1 - alpha) * (n + 1))
    q, capped = calibration.quantile(0, 0.001)
    assert capped and q == n
    low, high = co.interval(10.0, 2.0, 3.0)
    assert (low, high) == (4.0, 16.0)


def test_blocks_share_one_unit_of_weight_whatever_their_length():
    scores = np.array([[1.0]] * 10 + [[5.0]])
    weights = np.array([0.1] * 10 + [1.0])
    calibration = co.Calibration(scores, weights, n_blocks=2)
    q, capped = calibration.quantile(0, 0.5)
    assert q == 5.0 and not capped  # the long block of low scores counts once, so the median block score is the second block's
    assert abs(calibration.p_value(0, 5.0) - (1 + 1.0) / 3) < 1e-12
    assert abs(calibration.p_value(0, 1.0) - (1 + 2.0) / 3) < 1e-12
    assert abs(calibration.p_value(0, 9.0) - 1 / 3) < 1e-12


def test_verdicts_by_bonferroni_and_holm():
    family = co.verdict({"a": 0.001, "b": 0.02, "c": 0.5}, 0.05)
    assert family["bonferroni"] == {"a": True, "b": False, "c": False} and family["holm"] == {"a": True, "b": True, "c": False}
    assert family["consistent"] is False and family["channels"] == 3
    assert co.verdict({"a": 0.2, "b": 0.9}, 0.05)["consistent"] is True
    assert co.verdict({}, 0.05)["consistent"] is None
    stepped = co.verdict({"a": 0.01, "b": 0.03, "c": 0.04}, 0.05)
    assert stepped["bonferroni"] == {"a": True, "b": False, "c": False} and stepped["holm"] == {"a": True, "b": False, "c": False}
