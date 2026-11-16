"""The offline table of the CausalDiagnosis MCP on a hand made world: the tiny asset of the asset layer tests, whose recordings
move the mean of a channel by the degradation rank of every component the channel indicates, fitted as the rig is. No value
of the rig enters this test, and the file in the repository holds the header of the table alone."""
import numpy as np
import pytest

from CausalDiagnosis.datasets.tests.conftest import ROOT, tiny, tiny_data, tiny_recordings  # noqa: F401
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import offline_table
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import Models
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import asset_view, start_sets
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.conformal_intervals import DEFAULT_ALPHA
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.effect_model import COVARIANCE_ESTIMATORS, DEFAULT_COVARIANCE, combo_key, fit_episode
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models import fit

VERDICTS = ("fewer_wrong", "gap_two_sided", "gap_one_sided")


def test_the_offline_table_of_a_hand_made_world_follows_what_its_readings_show_and_ships_its_header_alone(tiny, tiny_recordings):
    """The four single fault starts of the tiny sampler at the ten entry offsets give 40 episodes per row. The prehistory holds
    every summary, and so does the second live reading, of the full group, so every such episode names its start and repairs
    it, under the selected pairs and under the main effects fit alike. The first live reading, of the basic group, shows the fan
    alone, so it leaves wrong first candidates and wrong repairs, and E17 rejects none of them, since with 11 calibration blocks
    the smallest conformal p value, 1/12, lies above alpha 0.05. Every row carries the calibration check against both main
    effects references, each verdict the comparison of the row's own numbers."""
    models = Models(fit(tiny_recordings, asset_id=tiny.asset_id, resamples=0))
    view = asset_view(tiny.asset_id)
    rows = offline_table.table(models, view)
    assert rows == offline_table.table(models, view, starts=start_sets(view, models.blocks)[1])  # the starts of the test set by default
    assert [(r["prehistory"], r["readings"]) for r in rows] == [(p, k) for p in (5, 0) for k in offline_table.READINGS if p or k]
    assert all(r["episodes"] == 40 and (r["e14_covers_truth"] is None if r["readings"] == 0 else 0 <= r["e14_covers_truth"] <= 1) for r in rows)
    for row in rows:
        if row["prehistory"] or row["readings"] > 1:
            assert (row["top1_rate"], row["exact_wrong"], row["wrong_repairs"], row["e17_rejects_wrong_repair"]) == (1.0, 0, 0, None)
            assert row["top_mass"] > 0.99 and row["main_effects_top1_rate"] == 1.0
            assert (row["main_effects_exact_wrong"], row["guard_reference_exact_wrong"], row["fewer_wrong"], row["guard_fewer_wrong"]) == (0, 0, 0, 0)
    basic = next(r for r in rows if (r["prehistory"], r["readings"]) == (0, 1))
    assert basic["exact_wrong"] == round(basic["episodes"] * (1 - basic["top1_rate"])) > 0
    assert 0 < basic["wrong_repairs"] <= basic["exact_wrong"] and basic["e17_rejects_wrong_repair"] == 0.0
    for row in rows:
        assert tuple(row) == offline_table.COLUMNS
        assert abs(row["calibration_gap"] - (row["top_mass"] - row["top1_rate"])) < 1e-12
        for prefix in ("main_effects", "guard_reference"):
            assert abs(row[f"{prefix}_calibration_gap"] - (row[f"{prefix}_top_mass"] - row[f"{prefix}_top1_rate"])) < 1e-12
            assert row[f"{prefix}_exact_wrong"] == round(row["episodes"] * (1 - row[f"{prefix}_top1_rate"]))
        like = offline_table.comparison(row["exact_wrong"], row["calibration_gap"], row["main_effects_exact_wrong"], row["main_effects_calibration_gap"])
        assert {name: row[name] for name in VERDICTS} == like
        guard = offline_table.comparison(row["exact_wrong"], row["calibration_gap"], row["guard_reference_exact_wrong"],
                                         row["guard_reference_calibration_gap"])
        assert (row["guard_fewer_wrong"], row["guard_gap_one_sided"]) == (guard["fewer_wrong"], guard["gap_one_sided"])
    assert offline_table.COLUMNS[:11] == ("prehistory", "readings", "episodes", "top_mass", "top1_rate", "exact_wrong", "e14_covers_truth",
                                          "e17_rejects_wrong_repair", "wrong_repairs", "main_effects_top_mass", "main_effects_top1_rate")
    shipped = (ROOT / "CausalDiagnosis" / "results" / "offline_mcp.csv").read_text(encoding="utf-8")
    assert shipped == ",".join(offline_table.COLUMNS) + "\n"


def test_the_calibration_check_states_both_tests_of_the_gap():
    """Fewer wrong first candidates than the reference, and the gap between the top mass and the accuracy against the size of the
    reference's gap: two sided as first registered, one sided as amended, so a belief that is less sure than it is right passes
    the one sided test and can fail the two sided one, and a belief that is too sure fails both."""
    assert offline_table.comparison(3, -0.2, 5, 0.1) == {"fewer_wrong": 1, "gap_two_sided": 0, "gap_one_sided": 1}
    assert offline_table.comparison(3, 0.05, 5, -0.1) == {"fewer_wrong": 1, "gap_two_sided": 1, "gap_one_sided": 1}
    assert offline_table.comparison(3, 0.2, 5, -0.1) == {"fewer_wrong": 1, "gap_two_sided": 0, "gap_one_sided": 0}
    assert offline_table.comparison(5, 0.1, 5, 0.1) == {"fewer_wrong": 0, "gap_two_sided": 1, "gap_one_sided": 1}


def test_the_like_for_like_reference_takes_the_estimator_of_the_start_and_the_other_reference_the_default_one(tiny, tiny_recordings):
    """The main effects fit that the selected pairs are compared with is built under the covariance estimator of the start's
    form, so the two differ in the pairs alone. The second reference is built under the registered default estimator, as the
    unit test over the starts of the test set builds it, and the two references are one fit when the form holds the default."""
    raw = fit(tiny_recordings, asset_id=tiny.asset_id, resamples=0)
    start = start_sets(asset_view(tiny.asset_id), Models(raw).blocks)[1][0]
    key = combo_key(start)
    for estimator in COVARIANCE_ESTIMATORS:
        models = Models({**raw, "selections": {**raw["selections"], key: {**raw["selections"][key], "covariance": estimator}}})
        fits = offline_table.start_fits(models, start, DEFAULT_ALPHA)
        assert fits.main_effects.pairs == [] and fits.guard_reference.pairs == []
        assert np.array_equal(fits.main_effects.sigma_within, fit_episode(models.recorded, [], start, 0, 0, covariance=estimator).effect.sigma_within)
        assert np.array_equal(fits.guard_reference.sigma_within, fit_episode(models.recorded, [], start, 0, 0).effect.sigma_within)
        assert (fits.guard_reference is fits.main_effects) == (estimator == DEFAULT_COVARIANCE)
        assert np.array_equal(fits.episode.effect.sigma_within[0], fit_episode(models.recorded, models.pairs_without(start), start, 0, 0,
                                                                                covariance=estimator).effect.sigma_within[0])


def test_the_table_reads_the_development_starts_when_asked_and_the_command_keeps_them_out_of_the_table_of_the_test_set(tiny, tiny_recordings):
    """A report that is read before the freeze takes the development starts of the asset, which share no combination with the
    starts of the test set, and the command refuses to write them into the file that holds the table of the test set."""
    models = Models(fit(tiny_recordings, asset_id=tiny.asset_id, resamples=0))
    view = asset_view(tiny.asset_id)
    development, test = start_sets(view, models.blocks)
    assert development and test and not set(development) & set(test)
    rows = offline_table.table(models, view, starts=development)
    assert [(r["prehistory"], r["readings"]) for r in rows] == [(p, k) for p in (5, 0) for k in offline_table.READINGS if p or k]
    assert all(r["episodes"] == 10 * len(development) and tuple(r) == offline_table.COLUMNS for r in rows)
    with pytest.raises(SystemExit):
        offline_table.main(["--starts", "development"])


def test_a_start_set_without_a_recorded_start_is_refused_by_name(tiny, tiny_recordings):
    """A row of the table divides by the count of the summaries it scored, so a start set that holds no recorded start of the
    asset is refused before any row is built."""
    models = Models(fit(tiny_recordings, asset_id=tiny.asset_id, resamples=0))
    with pytest.raises(ValueError, match="the start set holds no recorded start of the asset"):
        offline_table.table(models, asset_view(tiny.asset_id), starts=[])


def test_the_table_holds_the_row_of_the_record_before_the_episode_alone(tiny, tiny_recordings, tmp_path):
    """The row of prehistory 5 and no live reading is the evidence of the two rows without a clock: it is built, its coverage
    cell is empty in the file, since it released no live summary, and the pair of no prehistory and no live reading, which
    holds no evidence, is not built. The count of rows follows READINGS."""
    models = Models(fit(tiny_recordings, asset_id=tiny.asset_id, resamples=0))
    rows = offline_table.table(models, asset_view(tiny.asset_id))
    assert offline_table.READINGS[0] == 0 and len(rows) == 2 * len(offline_table.READINGS) - 1
    record = [r for r in rows if r["readings"] == 0]
    assert [(r["prehistory"], r["e14_covers_truth"]) for r in record] == [(5, None)]
    path = tmp_path / "offline.csv"
    offline_table.write(rows, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    cells = dict(zip(offline_table.COLUMNS, lines[1 + rows.index(record[0])].split(",")))
    assert (cells["prehistory"], cells["readings"], cells["e14_covers_truth"]) == ("5", "0", "")


def test_the_marked_readings_of_the_table_give_the_observations_of_the_unmarked_ones(tiny, tiny_recordings, monkeypatch):
    """Every reading of an episode of the table carries the mark of a record and has a cycle of its own, so the readings without
    the mark give the same observations and the same rows."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import Reading

    models = Models(fit(tiny_recordings, asset_id=tiny.asset_id, resamples=0))
    view = asset_view(tiny.asset_id)
    start = start_sets(view, models.blocks)[1][0]
    marked = offline_table.table(models, view, starts=[start])
    monkeypatch.setattr(offline_table, "Reading", lambda *values, recorded=False: Reading(*values))
    assert offline_table.table(models, view, starts=[start]) == marked
