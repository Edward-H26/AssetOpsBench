"""CausalDiagnosis/evaluation/metrics/table_statistics.py on synthetic arrays, each expectation computed by hand from the formula, and the
bootstrap against a loop in plain Python over the resamples that the count matrix describes."""
import math
import statistics

import numpy as np
import pytest

from CausalDiagnosis.evaluation.metrics.table_statistics import (Z_95, binomial_bar, cluster_bootstrap, cluster_proportion, cohens_kappa, holm, kappa_of_counts,
                   paired_cluster_bootstrap, ratio, resample_counts, resampled_values, sign_flip_p, spearman, wilson)

# made-up items in four clusters of unequal size, the clusters not in sorted order: a number, a zero or one outcome, whether
# the item is counted, a second number for a ratio, and a second zero or one rating for kappa
ITEMS = [(2.5, 1, 1, 4.0, 1), (0.25, 0, 1, 1.0, 0), (3.0, 1, 0, 2.0, 1), (1.5, 0, 1, 0.5, 1), (4.0, 1, 1, 3.0, 0), (0.5, 1, 1, 2.5, 1),
         (2.0, 0, 0, 1.5, 0), (1.0, 1, 1, 0.75, 1), (3.5, 0, 1, 2.25, 0), (0.75, 1, 1, 1.25, 1)]
CLUSTERS = ["d", "b", "d", "a", "c", "d", "b", "c", "c", "c"]
RESAMPLES, SEED = 300, 5


def _resamples():
    """The items of every resample in plain Python: the items of the k-th cluster, in the sorted order of the labels, as often as
    row r of the count matrix says."""
    labels = sorted(set(CLUSTERS))
    for row in resample_counts(len(labels), RESAMPLES, SEED):
        yield [item for label, times in zip(labels, row) for item, cluster in zip(ITEMS, CLUSTERS) if cluster == label for _ in range(int(times))]


def _plain_kappa(items):
    first, second = [bool(item[1]) for item in items], [bool(item[4]) for item in items]
    return cohens_kappa(first, second) if len(set(first)) > 1 and len(set(second)) > 1 else None


# (the columns of the matrix form, its statistic, the same statistic on the items of a resample in plain Python, whether it is exact)
STATISTICS = {
    "mean": ([[x, 1] for x, *_ in ITEMS], ratio, lambda items: sum(item[0] for item in items) / len(items), False),
    "share": ([[hit * counted, counted] for _, hit, counted, _, _ in ITEMS], ratio,
              lambda items: sum(item[1] * item[2] for item in items) / sum(item[2] for item in items) if sum(item[2] for item in items) else None, True),
    "ratio": ([[x, y] for x, _, _, y, _ in ITEMS], ratio, lambda items: sum(item[0] for item in items) / sum(item[3] for item in items), False),
    "kappa": ([[a and b, a and not b, b and not a, not a and not b] for _, a, _, _, b in ITEMS], kappa_of_counts, _plain_kappa, False),
}


@pytest.mark.parametrize("name", sorted(STATISTICS))
def test_the_matrix_bootstrap_equals_a_loop_over_the_same_draws(name):
    """Every resampled value of the matrix form equals the value that plain Python computes on the items of the resample which
    the row of the count matrix describes, and has no value exactly where plain Python has none. A wrong axis of the product
    or a wrong order of the clusters gives other values."""
    columns, statistic, plain, exact = STATISTICS[name]
    values = resampled_values(columns, CLUSTERS, statistic, RESAMPLES, SEED)
    expected = [plain(items) for items in _resamples()]
    assert len(values) == len(expected) == RESAMPLES
    assert [bool(np.isfinite(v)) for v in values] == [e is not None for e in expected]
    for value, reference in zip(values, expected):
        if reference is not None:
            assert value == reference if exact else value == pytest.approx(reference, abs=1e-12)
    # the summary reads these values: the percentile interval and the standard deviation of those that are numbers
    kept = [e for e in expected if e is not None]
    out = cluster_bootstrap(columns, CLUSTERS, statistic, RESAMPLES, SEED)
    low, high = np.percentile(kept, [2.5, 97.5])
    assert (out["low"], out["high"], out["resamples_used"]) == (pytest.approx(low, abs=1e-12), pytest.approx(high, abs=1e-12), len(kept))
    assert out["estimate"] == pytest.approx(plain(ITEMS), abs=1e-12)


def test_the_standard_error_is_the_spread_of_the_resampled_values():
    """se is the standard deviation, with one less than their number below the line, of the resampled values, the bootstrap
    estimate of the standard error, here against the loop in plain Python. One cluster gives one value throughout and a spread of zero."""
    for columns, statistic, plain, _ in STATISTICS.values():
        kept = [value for items in _resamples() if (value := plain(items)) is not None]
        assert cluster_bootstrap(columns, CLUSTERS, statistic, RESAMPLES, SEED)["se"] == pytest.approx(statistics.stdev(kept), abs=1e-12)
    assert paired_cluster_bootstrap([1, 2, 6], [0, 0, 0], ["b", "b", "b"], resamples=200)["se"] == 0
    assert cluster_proportion([True, False], ["a", "b"], resamples=400, seed=1)["se"] == pytest.approx(
        statistics.stdev(count / 2 for count in resample_counts(2, 400, 1)[:, 0]), abs=1e-12)


def test_the_count_matrix_draws_as_many_clusters_as_there_are_and_repeats_for_the_same_seed():
    counts = resample_counts(7, 500, 3)
    assert counts.shape == (500, 7) and (counts.sum(axis=1) == 7).all() and counts.min() >= 0
    assert (counts == resample_counts(7, 500, 3)).all()
    assert not (counts == resample_counts(7, 500, 4)).all()
    # the resamples differ from each other, and every cluster is drawn about once per resample
    assert len({tuple(row) for row in counts}) > 100 and counts.mean(axis=0) == pytest.approx([1.0] * 7, abs=0.15)
    with pytest.raises(ValueError):
        counts[0, 0] = 9  # the matrix is shared by every statistic of the process, so nobody may write it


def test_a_resample_without_a_counted_item_is_dropped_and_counted():
    """Two clusters, of which b counts no item: a resample that draws b twice has no share, so it is dropped, and the result
    says how many resamples it used."""
    drawn_a = int((resample_counts(2, 400, 2)[:, 0] > 0).sum())
    out = cluster_bootstrap([[1, 1], [0, 0]], ["a", "b"], ratio, resamples=400, seed=2)
    assert 0 < drawn_a < 400 and out["resamples_used"] == drawn_a
    assert (out["estimate"], out["low"], out["high"], out["se"]) == (1, 1, 1, 0)
    assert cluster_proportion([True, True], ["a", "b"], counted=[True, False], resamples=400, seed=2)["proportion"] == 1


def test_every_bootstrap_result_names_its_clusters():
    """Each of the three results says over how many clusters it was resampled, so a table can refuse an interval over too few."""
    assert paired_cluster_bootstrap([1, 0, 2, 1], [0, 0, 1, 1], ["x", "x", "y", "z"], resamples=50)["clusters"] == 3
    assert cluster_proportion([True, False, True], ["a", "b", "b"], resamples=50)["clusters"] == 2
    assert cluster_proportion([True], ["a"], counted=[False], resamples=50)["clusters"] == 1
    assert cluster_bootstrap([[1, 1]] * 4, ["a", "b", "c", "c"], ratio, resamples=50)["clusters"] == 3
    assert cluster_bootstrap([], [], ratio)["clusters"] == 0


def test_kappa_of_counts_equals_cohens_kappa_and_has_no_value_where_a_rating_is_constant():
    first, second = [True, True, False, False], [True, False, False, False]
    counts = [[sum(a and b for a, b in zip(first, second)), sum(a and not b for a, b in zip(first, second)),
               sum(b and not a for a, b in zip(first, second)), sum(not a and not b for a, b in zip(first, second))]]
    assert kappa_of_counts(counts)[0] == cohens_kappa(first, second) == pytest.approx(0.5)
    # both yes throughout, the first rating constant, the second rating constant: no value
    assert all(math.isnan(v) for v in kappa_of_counts([[4, 0, 0, 0], [2, 2, 0, 0], [2, 0, 2, 0], [0, 0, 0, 0]]))
    assert ratio([[3, 2], [1, 0]])[0] == 1.5 and math.isnan(ratio([[3, 2], [1, 0]])[1])


def test_paired_cluster_bootstrap_resamples_whole_clusters():
    # two clusters of one episode each with differences 1 and 3: a resample draws two clusters with replacement, so
    # every resampled mean is 1, 2 or 3 and the 95 percent percentile interval spans the two extremes
    out = paired_cluster_bootstrap([2, 5], [1, 2], ["block_a", "block_b"], resamples=4000, seed=1)
    assert out["difference"] == 2 and out["clusters"] == 2 and out["episodes"] == 2
    assert out["low"] == 1 and out["high"] == 3 and "p_value" not in out


def test_paired_cluster_bootstrap_keeps_episodes_of_one_cluster_together():
    # one cluster holding every episode: every resample is the whole sample, so the interval has zero width
    out = paired_cluster_bootstrap([1, 2, 6], [0, 0, 0], ["b", "b", "b"], resamples=200)
    assert out["difference"] == out["low"] == out["high"] == 3


def test_paired_cluster_bootstrap_is_deterministic_and_empty_safe():
    first = paired_cluster_bootstrap([1, 0, 2, 1], [0, 0, 1, 1], ["x", "x", "y", "z"], resamples=300, seed=7)
    second = paired_cluster_bootstrap([1, 0, 2, 1], [0, 0, 1, 1], ["x", "x", "y", "z"], resamples=300, seed=7)
    assert first == second
    assert paired_cluster_bootstrap([], [], [])["difference"] is None
    with pytest.raises(ValueError):
        paired_cluster_bootstrap([1, 2], [1], ["a", "b"])


def test_cluster_bootstrap_of_a_statistic_of_sums_resamples_whole_clusters():
    # two clusters of one item each: resampled means are 1, 2 or 3 and the interval spans the extremes
    out = cluster_bootstrap([[1, 1], [3, 1]], ["a", "b"], ratio, resamples=4000, seed=1)
    assert out["estimate"] == 2 and out["low"] == 1 and out["high"] == 3 and out["clusters"] == 2 and out["items"] == 2
    # kappa over the counts of the rating pairs: a resample that draws one cluster twice has a constant rating and no value, so it
    # is dropped, every other resample is the full sample, and the estimate is the full sample kappa
    ratings = [(True, True), (True, False), (False, False), (False, False)]
    counts = [[a and b, a and not b, b and not a, not a and not b] for a, b in ratings]
    out = cluster_bootstrap(counts, ["a", "a", "b", "b"], kappa_of_counts, resamples=400)
    assert out["estimate"] == out["low"] == out["high"] == pytest.approx(0.5) and 0 < out["resamples_used"] < 400
    assert cluster_bootstrap([], [], ratio)["estimate"] is None
    with pytest.raises(ValueError):
        cluster_bootstrap([[1, 1]], ["a", "b"], ratio)
    with pytest.raises(ValueError):
        cluster_bootstrap([1, 3], ["a", "b"], ratio)  # one column per summed quantity, not a plain list of numbers


def test_cluster_proportion_widens_against_wilson_when_the_episodes_of_a_block_agree():
    # four blocks of five episodes, two passing every episode and two failing every one: the share is one half, and a resample draws
    # four whole blocks, so the share of a resample is the number of passing blocks drawn over four, zero with probability 1/16 and one
    # with 1/16, both above the 2.5 percent tail, so the 95 percent interval spans 0 to 1, where Wilson over twenty independent
    # episodes reads 0.299 to 0.701
    hits = [block < 2 for block in range(4) for _ in range(5)]
    blocks = [f"b{block}" for block in range(4) for _ in range(5)]
    out = cluster_proportion(hits, blocks, resamples=4000, seed=1)
    narrow = wilson(10, 20)
    assert out["proportion"] == 0.5 and (out["low"], out["high"]) == (0.0, 1.0)
    assert out["high"] - out["low"] > narrow["high"] - narrow["low"] and narrow["low"] == pytest.approx(0.299, abs=1e-3)
    # twenty blocks of one episode each: the clusters are the episodes, and the interval comes close to Wilson's
    single = cluster_proportion(hits, [f"e{i}" for i in range(20)], resamples=4000, seed=1)
    assert abs((single["high"] - single["low"]) - (narrow["high"] - narrow["low"])) < 0.1


def test_cluster_proportion_counts_the_hits_among_the_counted_items_and_is_empty_without_one():
    # precision as A8 reads it: three declarations on three blocks, two of them correct, and a correct flag outside the counted items
    # never counts
    out = cluster_proportion([True, False, True, True], ["a", "b", "c", "d"], counted=[True, True, True, False], resamples=500, seed=0)
    assert out["proportion"] == pytest.approx(2 / 3) and 0 <= out["low"] <= 2 / 3 <= out["high"] <= 1
    assert cluster_proportion([False], ["a"], counted=[False]) == {"proportion": None, "low": None, "high": None, "se": None, "clusters": 1}
    with pytest.raises(ValueError):
        cluster_proportion([True, False], ["a"])


def test_holm_step_down_adjusts_in_sorted_order_and_stays_monotone():
    # sorted p values 0.01, 0.03, 0.04 with m = 3: 3 x 0.01 = 0.03, max(0.03, 2 x 0.03) = 0.06, max(0.06, 1 x 0.04) = 0.06
    out = holm([0.01, 0.04, 0.03], alpha=0.05)
    assert out["adjusted"] == pytest.approx([0.03, 0.06, 0.06])
    assert out["reject"] == [True, False, False]
    assert holm([0.5, 0.9])["adjusted"] == pytest.approx([1.0, 1.0])
    assert holm([]) == {"adjusted": [], "reject": []}


def test_the_binomial_bar_is_the_smallest_count_whose_upper_tail_under_the_constant_answer_is_at_most_the_level():
    """The bar of the gate: with 48 episodes and a constant answer right in 18 the upper tail is 0.028 at 25 and 0.052 at 24, so the
    bar is 25; a constant answer that is never right is beaten by one right episode, and one that is always right by no count."""
    def tail(n, right, k):
        p = right / n
        return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))
    assert binomial_bar(48, 18) == 25 and round(tail(48, 18, 25), 3) == 0.028 and round(tail(48, 18, 24), 3) == 0.052
    assert binomial_bar(5, 1) == 4 and binomial_bar(5, 0) == 1 and binomial_bar(5, 5) is None and binomial_bar(0, 0) is None
    for n, right in ((48, 18), (5, 1), (47, 9), (30, 3)):
        bar = binomial_bar(n, right)
        assert tail(n, right, bar) <= 0.05 < tail(n, right, bar - 1)


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


def test_sign_flip_p_enumerates_every_assignment_of_signs_when_the_clusters_are_few():
    """Three clusters with sums 1, 2 and 3 (the second of two episodes): of the eight assignments of signs, the sums +-6 reach the
    observed 6 and +-4 do not, so the exact p value is 2 / 8; every cluster of one sign gives 2 / 2**G, one cluster gives 1."""
    assert sign_flip_p([1, 1, 1, 3], ["a", "b", "b", "c"], flips=8) == pytest.approx(2 / 8)
    assert sign_flip_p([1] * 5, [f"b{i}" for i in range(5)], flips=10_000) == pytest.approx(2 / 32)
    assert sign_flip_p([1, 2, 6], ["b", "b", "b"]) == 1.0
    # no difference at all: every assignment ties zero
    assert sign_flip_p([0, 0, 0], ["a", "b", "c"]) == 1.0
    # sums 3 and -3 cancel: the observed mean is zero, so every assignment is at least as far from it
    assert sign_flip_p([3, -3], ["a", "b"]) == 1.0
    assert sign_flip_p([], []) is None
    with pytest.raises(ValueError):
        sign_flip_p([1, 2], ["a"])


def test_sign_flip_p_draws_random_assignments_with_an_add_one_share_when_the_clusters_are_many():
    """Twelve clusters of one sign with fractional sums, an outcome that is no whole number, under 100 flips: 2**12 assignments
    exceed the flips, so random ones are drawn, at most a few reach the observed sum, the share is add one and never zero, and
    the same seed gives the same p value."""
    clusters = [f"b{i}" for i in range(12)]
    p = sign_flip_p([0.5] * 12, clusters, flips=100, seed=3)
    assert p == sign_flip_p([0.5] * 12, clusters, flips=100, seed=3) and 1 / 101 <= p <= 5 / 101
    assert sign_flip_p([0.5] * 12, clusters, flips=4096) == pytest.approx(2 / 4096)


def test_sign_flip_p_drops_the_clusters_whose_sum_is_zero():
    """7 clusters of sum 1 and 40 of sum 0: a zero sum changes no assignment, so the test runs over the 7 and is exact at
    2 / 2**7 under the default flips, where 2**47 assignments would have been drawn at random. A cluster whose differences
    cancel is dropped like one that holds only zeros."""
    clusters = [f"b{i}" for i in range(47)]
    assert sign_flip_p([1.0] * 7 + [0.0] * 40, clusters, flips=10_000) == 2 / 2 ** 7
    assert sign_flip_p([1.0] * 7 + [0.0] * 40, clusters, flips=10_000, seed=9) == 2 / 2 ** 7
    assert sign_flip_p([1.5, 2.5, 3.0, -3.0, 0.0], ["a", "b", "c", "c", "d"], flips=4) == 2 / 4


def test_sign_flip_p_of_whole_number_sums_is_exact_by_convolution_at_any_count_of_clusters():
    """With whole set the share is counted over every assignment whatever the number of clusters: 47 clusters of sum 1 give
    exactly 2 / 2**47, and 20 clusters with sums between -5 and 5 give the count over all 2**20 assignments, listed here by
    doubling. Neither moves with the flips or the seed."""
    clusters = [f"b{i}" for i in range(47)]
    assert sign_flip_p([1] * 47, clusters, whole=True) == 2 / 2 ** 47
    assert sign_flip_p([1] * 47, clusters, flips=10, seed=4, whole=True) == 2 / 2 ** 47
    sums = [3, -1, 4, 1, -5, 2, 0, 5, -3, 1, 2, -2, 4, 1, -1, 3, 5, -4, 2, 1]
    totals = np.zeros(1, dtype=np.int64)
    for s in sums:
        totals = np.concatenate([totals + s, totals - s])
    expected = int((np.abs(totals) >= abs(sum(sums))).sum()) / 2 ** 20
    # each cluster of two episodes, whose differences add up to its sum
    differences = [d for s in sums for d in (s - 1, 1)]
    labels = [f"c{i}" for i in range(20) for _ in range(2)]
    assert len(totals) == 2 ** 20 and 0 < expected < 1
    assert sign_flip_p(differences, labels, whole=True) == expected
    assert sign_flip_p(differences, labels, flips=7, seed=11, whole=True) == expected


def test_sign_flip_p_is_one_when_no_cluster_sum_differs_from_zero():
    for whole in (False, True):
        assert sign_flip_p([0, 0, 0], ["a", "b", "c"], whole=whole) == 1.0
        assert sign_flip_p([1, -1, 2, -2], ["a", "a", "b", "b"], whole=whole) == 1.0


def test_sign_flip_p_gives_the_exact_values_of_one_sided_clusters_and_of_ten_against_one_beside_clusters_that_sum_to_zero():
    """By hand: seven clusters of +1 give 2 / 2**7 = 0.015625 and ten give 2 / 2**10, about 0.001953, since only the two assignments of
    one sign reach the observed total. Ten clusters of +1 beside one of -5 give 1278 / 2**11, about 0.624: the total is 5, and 639
    assignments reach a total of size 5 or more with the -5 cluster positive and 639 with it negative. A cluster whose sum is zero, one
    zero or a +1 and a -1, enters no assignment, so two of them beside each case leave every value as it is, with and without whole."""
    cases = (([1] * 7, 2 / 2 ** 7), ([1] * 10, 2 / 2 ** 10), ([1] * 10 + [-5], 1278 / 2 ** 11))
    for sums, exact in cases:
        differences = [*sums, 0, 1, -1]
        clusters = [f"c{i}" for i in range(len(sums))] + ["zero", "even", "even"]
        for whole in (False, True):
            assert sign_flip_p(differences, clusters, whole=whole) == pytest.approx(exact, abs=1e-12)
    assert [round(exact, 6) for _, exact in cases] == [0.015625, 0.001953, 0.624023]


def test_sign_flip_p_refuses_a_fractional_sum_under_whole():
    """whole is set for a zero or one outcome, whose sums are whole numbers, so a fractional sum is a caller's mistake."""
    with pytest.raises(ValueError):
        sign_flip_p([0.5, 1], ["a", "b"], whole=True)
    assert sign_flip_p([0.5, 0.5, 1], ["a", "a", "b"], whole=True) == 2 / 4
