"""benchmarks/hydraulic_rig/stats.py on synthetic arrays, each expectation computed by hand from the formula."""
import math

import pytest

from stats import Z_95, cluster_bootstrap, cohens_kappa, holm, paired_cluster_bootstrap, rmst, spearman, wilson


def test_paired_cluster_bootstrap_resamples_whole_clusters():
    # two clusters of one episode each with differences 1 and 3: a resample draws two clusters with replacement, so
    # every resampled mean is 1, 2 or 3 and the 95 percent percentile interval spans the two extremes
    out = paired_cluster_bootstrap([2, 5], [1, 2], ["block_a", "block_b"], resamples=4000, seed=1)
    assert out["difference"] == 2 and out["clusters"] == 2 and out["episodes"] == 2
    assert out["low"] == 1 and out["high"] == 3
    assert 0 < out["p_value"] <= 1


def test_paired_cluster_bootstrap_keeps_episodes_of_one_cluster_together():
    # one cluster holding every episode: every resample is the whole sample, so the interval has zero width
    out = paired_cluster_bootstrap([1, 2, 6], [0, 0, 0], ["b", "b", "b"], resamples=200)
    assert out["difference"] == out["low"] == out["high"] == 3
    assert out["p_value"] == pytest.approx(2 * 1 / 201)


def test_paired_cluster_bootstrap_is_deterministic_and_empty_safe():
    first = paired_cluster_bootstrap([1, 0, 2, 1], [0, 0, 1, 1], ["x", "x", "y", "z"], resamples=300, seed=7)
    second = paired_cluster_bootstrap([1, 0, 2, 1], [0, 0, 1, 1], ["x", "x", "y", "z"], resamples=300, seed=7)
    assert first == second
    assert paired_cluster_bootstrap([], [], [])["difference"] is None
    with pytest.raises(ValueError):
        paired_cluster_bootstrap([1, 2], [1], ["a", "b"])


def test_cluster_bootstrap_of_any_statistic_resamples_whole_clusters():
    mean = lambda items: sum(items) / len(items)  # noqa: E731 - the statistic under test
    # two clusters of one item each: resampled means are 1, 2 or 3 and the interval spans the extremes
    out = cluster_bootstrap(mean, [1, 3], ["a", "b"], resamples=4000, seed=1)
    assert out["estimate"] == 2 and out["low"] == 1 and out["high"] == 3 and out["clusters"] == 2 and out["items"] == 2
    # kappa over pairs: a statistic that is None on some resamples drops them, and the estimate is the full sample kappa
    kappa = lambda pairs: cohens_kappa([a for a, _ in pairs], [b for _, b in pairs])  # noqa: E731
    pairs = [(True, True), (True, False), (False, False), (False, False)]
    out = cluster_bootstrap(kappa, pairs, ["a", "a", "b", "b"], resamples=400)
    assert out["estimate"] == pytest.approx(0.5) and out["low"] is not None
    assert cluster_bootstrap(mean, [], [])["estimate"] is None
    with pytest.raises(ValueError):
        cluster_bootstrap(mean, [1], ["a", "b"])


def test_holm_step_down_adjusts_in_sorted_order_and_stays_monotone():
    # sorted p values 0.01, 0.03, 0.04 with m = 3: 3 x 0.01 = 0.03, max(0.03, 2 x 0.03) = 0.06, max(0.06, 1 x 0.04) = 0.06
    out = holm([0.01, 0.04, 0.03], alpha=0.05)
    assert out["adjusted"] == pytest.approx([0.03, 0.06, 0.06])
    assert out["reject"] == [True, False, False]
    assert holm([0.5, 0.9])["adjusted"] == pytest.approx([1.0, 1.0])
    assert holm([]) == {"adjusted": [], "reject": []}


def test_wilson_interval_matches_the_closed_form():
    z = Z_95
    none_of_ten = wilson(0, 10)
    assert none_of_ten["proportion"] == 0 and none_of_ten["low"] == 0
    assert none_of_ten["high"] == pytest.approx(z * z / (10 + z * z))
    half = wilson(5, 10)
    center = (0.5 + z * z / 20) / (1 + z * z / 10)
    width = z * math.sqrt(0.25 / 10 + z * z / 400) / (1 + z * z / 10)
    assert half["low"] == pytest.approx(center - width) and half["high"] == pytest.approx(center + width)
    assert wilson(10, 10)["high"] == pytest.approx(1)
    assert wilson(0, 0) == {"proportion": None, "low": None, "high": None}


def test_rmst_is_the_area_under_the_kaplan_meier_curve():
    # events at turns 2 and 4 out of two episodes: survival 1 on [0, 2), 0.5 on [2, 4), 0 after, so the area is 3
    assert rmst([2, 4], [True, True], cap=10) == 3
    # one event at 2, one episode censored at the cap: survival 1 on [0, 2), 0.5 on [2, 10), area 2 + 4
    assert rmst([2, 10], [True, False], cap=10) == 6
    # no event at all: the whole cap
    assert rmst([10, 10], [False, False], cap=10) == 10
    # a censoring before an event leaves the risk set: event at 4 after a censoring at 3 drops survival to 0
    assert rmst([3, 4], [False, True], cap=10) == 4
    assert rmst([], [], cap=10) is None


def test_cohens_kappa_from_the_agreement_table():
    # observed agreement 3 of 4; chance 0.5 x 0.25 + 0.5 x 0.75 = 0.5; kappa (0.75 - 0.5) / (1 - 0.5)
    assert cohens_kappa([True, True, False, False], [True, False, False, False]) == pytest.approx(0.5)
    assert cohens_kappa(["a", "b"], ["a", "b"]) == 1
    assert cohens_kappa([True, True], [True, True]) is None
    assert cohens_kappa([], []) is None
    with pytest.raises(ValueError):
        cohens_kappa([True], [True, False])


def test_spearman_uses_average_ranks():
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    # a tie in x gets the average rank 1.5, 1.5, 3; with y ranks 1, 2, 3 the correlation is 0.5 x sqrt(3)
    assert spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(math.sqrt(3) / 2)
    assert spearman([1, 1], [1, 2]) is None and spearman([1], [2]) is None
