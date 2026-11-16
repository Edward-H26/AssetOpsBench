"""CausalDiagnosis/evaluation/metrics/realized_read_elimination.py: the paid reads of the released evidence, their realized elimination and
their cost of E21 on a stub of the filter whose effective number of hypotheses halves with every distinct reading, the
forecasts kept for calibration and those left out, the error without models.json, and on the fitted models the replay
of the server's own filter, whose belief after each read is the one the server answered with."""
import json
from pathlib import Path

import pytest

from CausalDiagnosis.evaluation.metrics import realized_read_elimination as read_elimination
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import build_blocks, default_facts_path, load_cycles

PRICED = {"read": {"g1": 1, "g2": 3}, "parts": {"pump": 30}, "repair_charge": 20}
FREE = {"read": {"g1": 0, "g2": 0}, "parts": {"pump": 0}, "repair_charge": 0}
FULL = {"f1": 1.0, "f2": 2.0}


class StubStick:
    """N(b) is 16 halved once per distinct reading of the evidence, the loss rate is 2 per cycle at every belief, and every
    sensor group has the fields f1 and f2."""
    digest = "stub"

    def __init__(self):
        self.rate_times = []

    def view(self, released):
        return released["rules"]

    def effective(self, view, episode):
        return 16.0 / 2 ** len({(r.t, tuple(sorted(r.values.items()))) for r in episode.readings})

    def rate(self, view, episode):
        self.rate_times.append(episode.time)
        return 2.0

    def group_fields(self, group):
        return {"f1", "f2"}


def _read(group, t_start, t_end, cost, served):
    return {"event": "read", "t_start": t_start, "t_end": t_end, "cost": cost, "sensor_group": group, "cycles": len(served),
            "served": [{"n": n, "t": t, "values": values} for n, t, values in served]}


# the clock keys of the advanced real-time row, whose reads play cycles as the fixtures' reads do, from its composed preset
CLOCK = {k: v for k, v in load_config("advanced_real_time")[1].to_json().items() if k in ("clock_mode", "durations")}


def _released(prices, events):
    return {"rules": {"prices": prices, **CLOCK}, "events": events, "status": {"time": max([0] + [e.get("t_end", 0) for e in events])}}


def test_a_priced_read_costs_its_price_and_its_played_cycles_at_the_rate_of_the_belief_it_met():
    released = _released(PRICED, [
        _read("g1", 0, 1, 1, [(1, 0, FULL)]),
        _read("g1", 1, 1, 0, [(2, 0, FULL)]),  # a re-read of the cycle paid before: free, so no paid read, and no new evidence
        _read("g2", 1, 3, 6, [(3, 1, {"f1": 3.0, "f2": 4.0}), (4, 2, {"f1": 5.0, "f2": 6.0})]),
    ])
    stick = StubStick()
    records = read_elimination.read_records(released, stick)
    assert [r["index"] for r in records] == [0, 2]
    first, second = records
    assert (first["effective_before"], first["effective_after"], first["eliminated"]) == (16.0, 8.0, 8.0)
    assert first["cost"] == 1 + 1 * 2.0  # the price of the event and one played cycle at the rate 2
    assert (second["effective_before"], second["effective_after"], second["eliminated"]) == (8.0, 2.0, 6.0)
    assert second["cost"] == 6 + 2 * 2.0  # two played cycles
    assert stick.rate_times == [0, 1]  # the rate of the belief at the cycle each read started, before its own readings
    measures = read_elimination.episode_measures(records)
    assert measures["paid_reads"] == 2 and measures["realized_elimination_per_read"] == 7.0
    assert measures["realized_elimination_per_cost"] == 14.0 / 13.0 and measures["calibrated_reads"] == 0


def test_every_read_of_the_free_base_row_costs_one_call_and_an_empty_re_read_eliminates_nothing():
    released = _released(FREE, [_read("g1", 0, 0, 0, [(1, 0, FULL)]), _read("g1", 0, 0, 0, [(2, 0, FULL)])])
    stick = StubStick()
    records = read_elimination.read_records(released, stick)
    assert [(r["eliminated"], r["cost"], r["rate"]) for r in records] == [(8.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    assert stick.rate_times == []  # a call has no production loss
    measures = read_elimination.episode_measures(records)
    assert measures["realized_elimination_per_read"] == 4.0 and measures["realized_elimination_per_cost"] == 4.0


def test_evidence_without_a_read_gives_no_record_and_loads_no_model(monkeypatch):
    monkeypatch.setattr(read_elimination, "yardstick", lambda *a: pytest.fail("no read, so no model is loaded"))
    assert read_elimination.read_records({}) == []
    assert read_elimination.read_records(_released(PRICED, [{"event": "fix_history", "t_start": 0, "t_end": 0, "cost": 0}])) == []
    assert read_elimination.episode_measures([])["realized_elimination_per_read"] is None


def _answer(observations, closures, rows):
    return {"evidence_ids": [f"obs/{n}/f1" for n in observations] + [f"wo/{c}" for c in closures],
            "elimination": [{"sensor_group": g, "cycles": 1, "effective_before": before, "expected_elimination": expected}
                            for g, before, expected in rows]}


def test_a_forecast_counts_only_when_it_saw_the_same_evidence_and_started_from_the_same_belief():
    released = _released(PRICED, [
        _read("g1", 0, 1, 1, [(1, 0, FULL)]),
        _read("g2", 1, 2, 3, [(2, 1, {"f1": 3.0, "f2": 4.0})]),
        {"event": "work_order_closed", "t_start": 2, "t_end": 2, "cost": 0, "id": "wo-1", "component": "pump"},
        _read("g1", 2, 3, 1, [(3, 2, {"f1": 5.0, "f2": 6.0})]),
        _read("g2", 3, 4, 3, [(4, 3, {"f1": 7.0})]),
        _read("g1", 4, 6, 2, [(5, 4, {"f1": 8.0, "f2": 9.0}), (6, 5, {"f1": 10.0, "f2": 11.0})]),
        _read("g2", 6, 7, 3, [(7, 6, {"f1": 12.0, "f2": 13.0})]),
    ])
    stick = StubStick()
    records = read_elimination.read_records(released, stick)
    answers = [
        {"evidence_ids": [], "elimination": [{"sensor_group": "g1", "cycles": 2, "effective_before": 16.0, "expected_elimination": 9.0}]},
        _answer([], [], [("g1", 16.0, 7.0)]),  # before the first read, on its belief: kept
        _answer([1], [], [("g2", 8.0, 3.0)]),  # after the first read, on the belief the second read met: kept
        _answer([1, 2], [], [("g1", 4.0, 1.5)]),  # before the closure that precedes the third read: left out
        _answer([1, 2, 3], ["wo-1"], [("g2", 2.0, 0.5)]),  # the fourth read served one field of its group: left out
        _answer([1, 2, 3, 4], ["wo-1"], [("g1", 1.0, 0.2)]),  # the fifth read played two cycles: left out
        _answer([1, 2, 3, 4, 5, 6], ["wo-1"], [("g2", 99.0, 0.1)]),  # another belief than the sixth read met: left out
    ]
    read_elimination.calibrate(records, released, answers, stick)
    assert [r.get("expected") for r in records] == [7.0, 3.0, None, None, None, None]
    measures = read_elimination.episode_measures(records)
    assert measures["calibrated_reads"] == 2 and measures["expected_elimination"] == 5.0
    assert measures["realized_elimination_calibrated"] == 6.0 and measures["calibration_gap"] == 1.0


def test_the_yardstick_without_models_json_names_the_fit_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models"):
        read_elimination.yardstick(tmp_path / "models.json", (100, 80, 0, 90))


@pytest.mark.skipif(not Path(default_facts_path()).exists(), reason="no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")
def test_the_yardstick_needs_the_start_whose_block_the_fits_of_the_episode_leave_out():
    with pytest.raises(ValueError, match="left_out"):
        read_elimination.yardstick()


@pytest.mark.skipif(not Path(default_facts_path()).exists(), reason="no data file in this clone, as ASSETS_WITHOUT_DATA=1 states")
def test_the_replay_ends_each_read_on_the_belief_the_server_answered_with_and_keeps_its_forecasts(tmp_path):
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import CausalTools
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.fitted_models import load_models
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.fixture_episode import FixtureEpisode

    cycles = load_cycles(default_facts_path())
    episode = FixtureEpisode(tmp_path, cycles, build_blocks(cycles), (100, 80, 0, 90), seed=0)
    tools = CausalTools(load_models().without(episode.start), episode.released_path)
    answers = [tools.rank_root_causes(3).model_dump()]
    for group in ("tier1", "tier2", "tier3"):
        episode.read(group)
        answers.append(json.loads(json.dumps(tools.rank_root_causes(3).model_dump())))
    released = json.loads(episode.released_path.read_text(encoding="utf-8"))
    records = read_elimination.read_records(released, left_out=episode.start)
    assert len(records) == 3
    for record, answer in zip(records, answers[1:]):
        assert record["effective_after"] == pytest.approx(answer["effective_hypotheses"], rel=1e-9)
        assert record["eliminated"] == pytest.approx(record["effective_before"] - record["effective_after"])
        assert record["cost"] > released["rules"]["prices"]["read"][record["sensor_group"]]  # one played cycle at a positive rate
    read_elimination.calibrate(records, released, answers, left_out=episode.start)
    kept = [(r["index"], r["expected"]) for r in records if "expected" in r]
    assert kept, "no forecast of the server was made on the belief a read met"
    for index, expected in kept:
        forecast = {row["sensor_group"]: row for row in answers[index]["elimination"]}[records[index]["sensor_group"]]
        assert expected == forecast["expected_elimination"]
