"""The causal-diagnosis server: static knowledge, the posterior replayed from the released answers, the worked S2 example of
docs/hydraulic_rig/causal-diagnosis-server.md, the rule that no tool opens a hidden file, the released answers of the environment-
action server, the generated models, and the MCP server over stdio when the mcp package is installed. No test states a
value computed from the rig data: each checks what must hold whatever the data, such as probabilities that sum to one,
bands that contain their prediction and residuals that equal the observation minus the prediction."""
import asyncio
import builtins
import importlib.util
import io
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, LEVELS, default_facts_path
from servers.causal_diagnosis import fit
from servers.causal_diagnosis import knowledge as kb
from servers.causal_diagnosis.adapter import (CONFIDENCE, MISS_RATE, CausalTools, binomial_tail, load_models, misses_tolerated,
                                              outcome_consistent)
from servers.causal_diagnosis.models import ErrorResult, TimeWindow
from servers.hydraulic_rig import episode
from servers.hydraulic_rig.names import tier_fields

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)

START, SEED = (100, 80, 0, 90), 0
S2_TEXT = ("Hydraulic unit hyd_1 was flagged by the performance monitor. Exactly two components are off their healthy levels. "
           "Find them and restore the unit within the budget.")
HIDDEN_FILES = ("episode.json", "log.json", "score.json")
HIDDEN_KEYS = ("seed", "allowed", "cycle", "hidden", "initial", "combo_after", "unnecessary", "facts_path")
STATIC_TOOLS = ("get_asset_context", "query_fmea_paths")
EPISODE_TOOLS = ("get_symptom_report", "get_episode_evidence", "rank_root_causes", "estimate_intervention_effect", "verify_outcome")
PS1, FS1 = "PS1_Pressure_bar_100Hz_mean", "FS1_VolumeFlow_l_per_min_10Hz_mean"
needs_mcp = pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="needs the mcp package (pinned in the fork's uv.lock)")


@contextmanager
def hidden_files_sealed():
    """Every open of the episode's state file, its log or score, or of the facts file fails, so a tool that reads a hidden
    source fails its test (docs/hydraulic_rig/causal-diagnosis-server.md, design rule "Only what the agent has seen")."""
    facts = Path(default_facts_path()).resolve()
    real_open = builtins.open

    def guarded(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            p = Path(file)
            if p.name in HIDDEN_FILES or p.resolve() == facts:
                raise PermissionError(f"a tool opened the hidden file {p.name}")
        return real_open(file, *args, **kwargs)

    builtins.open = io.open = guarded
    try:
        yield
    finally:
        builtins.open = io.open = real_open


def _start(tmp_path, seed=SEED, **overrides):
    kwargs = episode.default_env_kwargs()
    kwargs.update(overrides)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "episode.json"
    episode.start(path, seed, kwargs, default_facts_path())
    return path


def _act(path, tool, arguments, action):
    return json.loads(episode.call(path, tool, arguments, action))


def _measure(path, tier):
    return _act(path, "measure", {"tier": tier}, {"type": "measure", "tier": tier})


@pytest.fixture
def s2(tmp_path):
    """The worked example's episode: start (100, 80, 0, 90), the rule exactly two faulty, seed 0, and its tools."""
    path = _start(tmp_path, min_faulty=2, max_faulty=2, allowed=[START])
    return path, CausalTools(load_models(), episode.released_path(path))


def test_asset_context_states_components_channels_and_codes():
    tools = CausalTools(load_models())
    with hidden_files_sealed():
        ctx = tools.get_asset_context("hyd_1")
    assert ctx.asset_type == "HYDRAULIC_PUMP" and [c.name for c in ctx.components] == list(COMPONENTS)
    assert {c.name: (c.levels, c.healthy) for c in ctx.components} == {c: (list(LEVELS[c]), h) for c, h in [("cooler", 100), ("valve", 100), ("pump", 0), ("accumulator", 130)]}
    assert {c.name: c.n_fault_codes for c in ctx.components} == {"cooler": 2, "valve": 3, "pump": 2, "accumulator": 3}
    assert len(ctx.channels) == 17 and ctx.n_failure_codes == 16 and ctx.n_fault_codes == 10
    ps1 = next(c for c in ctx.channels if c.channel == "PS1")
    assert (ps1.field, ps1.unit, ps1.rate, ps1.group, ps1.indicates) == ("PS1_Pressure_bar_100Hz", "bar", "100Hz", "tier3", ["valve", "accumulator"])
    eps1 = next(c for c in ctx.channels if c.channel == "EPS1")
    assert (eps1.field, eps1.unit) == ("P1_MotorPower_W_100Hz", "W")
    fs1 = next(c for c in ctx.channels if c.channel == "FS1")
    assert (fs1.unit, fs1.rate, fs1.group) == ("l_per_min", "10Hz", "tier2")
    assert {c.group for c in ctx.channels if c.channel in ("TS1", "VS1", "CE")} == {"tier1"}
    assert ctx.time_window.unit == "static" and "assoc/helwig2015condition/PS1_valve" in ctx.evidence_ids
    assert isinstance(tools.get_asset_context("Chiller 6"), ErrorResult)


def test_fmea_paths_for_a_component_a_code_and_a_channel():
    tools = CausalTools(load_models())
    with hidden_files_sealed():
        acc = tools.query_fmea_paths("hyd_1", "accumulator")
        code = tools.query_fmea_paths("hyd_1", "valve_switching_critical")
        ts1 = tools.query_fmea_paths("hyd_1", "TS1")
        p1 = tools.query_fmea_paths("hyd_1", "P1_MotorPower_W_100Hz")
        ps4 = tools.query_fmea_paths("hyd_1", "PS4")
    assert acc.node_kind == "component" and {p.code: p.recommended_action for p in acc.paths} == {
        "accumulator_pressure_reduced": "monitor", "accumulator_pressure_severely_reduced": "service_accumulator",
        "accumulator_pressure_low_critical": "inspect_accumulator;service_or_replace"}
    first = acc.paths[0]
    assert [c.channel for c in first.channels] == ["PS1", "PS2", "PS3", "EPS1", "FS1", "SE"]
    assert all(0 < c.correlation <= 1 for c in first.channels)
    assert set(first.held_out_accuracy) == {"tier1", "tier2", "tier3"} and all(0 <= a <= 1 for a in first.held_out_accuracy.values())
    assert first.evidence_ids[0] == "fmea/industryasseteqa_hydraulic_facts/accumulator_pressure_low_critical"
    assert "assoc/helwig2015condition/EPS1_accumulator" in first.evidence_ids and acc.time_window.unit == "static"
    assert code.node_kind == "failure_mode" and [(p.component, p.level, p.severity) for p in code.paths] == [("valve", 73, "critical")]
    assert ts1.node_kind == "channel" and {p.component for p in ts1.paths} == {"cooler"} and len(ts1.paths) == 2
    assert {p.component for p in p1.paths} == {"valve", "accumulator"} and len(p1.paths) == 6
    assert ps4.paths == [] and "indicates no component" in ps4.message
    assert isinstance(tools.query_fmea_paths("hyd_1", "motor"), ErrorResult)
    assert isinstance(tools.query_fmea_paths("Chiller 6", "valve"), ErrorResult)


def test_static_server_has_no_episode_tools():
    tools = CausalTools(load_models())
    for call in (tools.get_symptom_report, lambda: tools.get_episode_evidence("all"), lambda: tools.rank_root_causes(3),
                 lambda: tools.estimate_intervention_effect("monitor"), lambda: tools.verify_outcome("pred/1")):
        assert isinstance(call(), ErrorResult)


def test_symptom_report_states_the_task_text_and_the_public_rules(s2, tmp_path):
    path, tools = s2
    with hidden_files_sealed():
        report = tools.get_symptom_report()
    assert report.task_text == S2_TEXT
    assert (report.report_time, report.fault_count_min, report.fault_count_max, report.n_candidates) == (0, 2, 2, 37)
    assert (report.budget, report.step_cap, report.repairs) == (150, 200, 2)
    assert report.prices["parts"]["accumulator"] == 20 and report.prices["downtime"] == 20 and report.prices["measure"]["3"] == 10
    assert report.status["phase"] == "investigate" and report.evidence_ids == [] and report.time_window == TimeWindow(start=0, end=0, unit="measurement")
    s1 = _start(tmp_path / "s1", min_faulty=1, max_faulty=1)
    assert CausalTools(load_models(), episode.released_path(s1)).get_symptom_report().task_text.endswith(
        "Exactly one component is off its healthy level. Find it and restore the unit within the budget.")
    default = _start(tmp_path / "default")
    text = CausalTools(load_models(), episode.released_path(default)).get_symptom_report()
    assert "Between one and two components are off their healthy levels. Find them" in text.task_text and text.n_candidates == 47
    told = tmp_path / "told"
    told.mkdir()
    episode.start(told / "episode.json", SEED, episode.default_env_kwargs(), default_facts_path(), task_text="Unit hyd_1 runs hot.")
    assert CausalTools(load_models(), episode.released_path(told / "episode.json")).get_symptom_report().task_text == "Unit hyd_1 runs hot."


def _ranking_is_coherent(ranking):
    """What every ranking must satisfy whatever the data: ordered probabilities in [0, 1], level distributions that sum to
    one, the flag and the unseparated components set by the confidence threshold, the separating channels of exactly the
    unseparated components, and faults that name the components a candidate sets off their healthy levels."""
    probs = [c.probability for c in ranking.candidates]
    assert probs == sorted(probs, reverse=True) and all(0 <= p <= 1 for p in probs)
    assert set(ranking.component_fault_probability) == set(COMPONENTS)
    assert all(0 <= p <= 1 for p in ranking.component_fault_probability.values())
    assert all(abs(sum(levels.values()) - 1) < 1e-9 for levels in ranking.level_probability.values())
    assert ranking.insufficient_evidence == (probs[0] < CONFIDENCE)
    assert ranking.unseparated == [c for c in COMPONENTS if max(ranking.level_probability[c].values()) < CONFIDENCE]
    assert list(ranking.separating_channels) == ranking.unseparated
    for cand in ranking.candidates:
        off = [(c, v) for c, v in cand.combination.items() if v != HEALTHY[c]]
        assert [(f.component, f.level) for f in cand.faults] == off
        assert all(f.code == kb.level_code(f.component, f.level) for f in cand.faults)


def test_worked_s2_example_of_the_design_document(s2):
    path, tools = s2
    with hidden_files_sealed():
        empty = tools.rank_root_causes(3)
    assert empty.n_candidates == 37 and empty.insufficient_evidence and empty.unseparated == list(COMPONENTS)
    assert empty.evidence_ids == [] and empty.time_window == TimeWindow(start=0, end=0, unit="measurement")
    assert all(abs(c.probability - 1 / empty.n_candidates) < 1e-12 for c in empty.candidates)
    _measure(path, 1)
    with hidden_files_sealed():
        evidence = tools.get_episode_evidence("tier1")
        first = tools.rank_root_causes(3)
    assert evidence.n_measurements == 1 and len(evidence.items) == 24
    ce = next(i for i in evidence.items if i.field == "CE_CoolingEfficiency_percent_1Hz_mean")
    assert (ce.evidence_id, ce.band_healthy, ce.release_time) == ("obs/1/CE_CoolingEfficiency_percent_1Hz_mean", ["cooler"], 1)
    assert (ce.unit, ce.rate, ce.channel, ce.statistic) == ("percent", "1Hz", "CE", "mean")
    assert ce.band_low <= ce.band_high and ce.band_cycles > 0 and ce.inside_band == (ce.band_low <= ce.value <= ce.band_high)
    _ranking_is_coherent(first)
    assert first.time_window == TimeWindow(start=1, end=1, unit="measurement") and len(first.evidence_ids) == 24
    _measure(path, 3)
    with hidden_files_sealed():
        second = tools.rank_root_causes(37)
        effect = tools.estimate_intervention_effect("replace:accumulator")
    _ranking_is_coherent(second)
    assert len(second.candidates) == second.n_candidates and abs(sum(c.probability for c in second.candidates) - 1) < 1e-9
    assert dict(zip(COMPONENTS, START)) in [c.combination for c in second.candidates]
    top = second.candidates[0]
    assert all(i.startswith("obs/") for i in top.evidence_ids)
    assert (effect.prediction_id, effect.price) == ("pred/1", 40)
    assert abs(effect.addresses_fault - second.component_fault_probability["accumulator"]) < 1e-9 and 0 <= effect.recovery <= 1
    assert effect.time_window == TimeWindow(start=2, end=3, unit="measurement") and effect.evidence_ids[-1] == "pred/1" and len(effect.predicted) == 17
    assert all(s.band_low <= s.mean <= s.band_high for s in effect.predicted)
    _act(path, "submit", top.combination, {"type": "submit", "diagnosis": top.combination})
    _act(path, "replace", {"component": "accumulator"}, {"type": "replace", "component": "accumulator"})
    with hidden_files_sealed():
        pending = tools.verify_outcome("pred/1")
    assert pending.status.startswith("not yet observable") and pending.recovered is False and pending.residuals == []
    _measure(path, 3)
    with hidden_files_sealed():
        verified = tools.verify_outcome("pred/1")
        third = tools.rank_root_causes(3)
        second_effect = tools.estimate_intervention_effect("replace:valve")
    predicted = {s.field: s for s in effect.predicted}
    residual = {r.field: r for r in verified.residuals}
    assert verified.status == "verified" and verified.observation == 3 and set(residual) == set(predicted)
    for field, r in residual.items():
        assert r.predicted == predicted[field].mean and abs(r.residual - (r.observed - r.predicted)) < 1e-9
        assert r.inside_band == (predicted[field].band_low <= r.observed <= predicted[field].band_high)
    n_outside = sum(not r.inside_band for r in verified.residuals)
    assert verified.recovered is False and verified.n_outside == n_outside and verified.n_outside_tolerated == 1
    assert verified.consistent == outcome_consistent(n_outside, 17, False, effect.recovery) and 0.0 < verified.p_value_outside <= 1.0
    assert verified.time_window == TimeWindow(start=3, end=3, unit="measurement") and verified.evidence_ids[:2] == ["pred/1", f"obs/3/{sorted(residual)[0]}"]
    _ranking_is_coherent(third)
    assert all(c.combination["accumulator"] == HEALTHY["accumulator"] for c in third.candidates) and 0 < third.n_candidates <= 37
    assert second_effect.prediction_id == "pred/2"
    assert abs(second_effect.addresses_fault - third.component_fault_probability["valve"]) < 1e-9 and 0 <= second_effect.recovery <= 1
    end = _act(path, "replace", {"component": "valve"}, {"type": "replace", "component": "valve"})
    assert end["episode_over"] and end["repairs_left"] == 0
    with hidden_files_sealed():
        last = tools.verify_outcome("pred/2")
        after_end = tools.estimate_intervention_effect("monitor")
    assert last.status.startswith("not yet observable") and last.recovered is None and last.recovery_status.startswith("not observable")
    assert after_end.prediction_id == "pred/3" and "the episode is over" in " ".join(after_end.assumptions)


def test_a_repair_that_the_episode_outlives_rules_out_the_healthy_combination(tmp_path):
    start = (100, 100, 1, 130)  # class S1, the pump at level 1
    path = _start(tmp_path, min_faulty=1, max_faulty=1, allowed=[start])
    tools = CausalTools(load_models(), episode.released_path(path))
    _measure(path, 1)
    _act(path, "submit", dict(zip(COMPONENTS, start)), {"type": "submit", "diagnosis": dict(zip(COMPONENTS, start))})
    assert not _act(path, "replace", {"component": "cooler"}, {"type": "replace", "component": "cooler"})["episode_over"]
    with hidden_files_sealed():
        ranking = tools.rank_root_causes(10)
        watch = tools.estimate_intervention_effect("monitor")
    _ranking_is_coherent(ranking)
    assert dict(HEALTHY) not in [c.combination for c in ranking.candidates] and watch.recovery == 0.0
    assert _act(path, "replace", {"component": "pump"}, {"type": "replace", "component": "pump"})["episode_over"]
    with hidden_files_sealed():
        ended = tools.rank_root_causes(10)
    assert dict(HEALTHY) in [c.combination for c in ended.candidates]  # a recovery and a used-up repair limit look the same


def test_evidence_groups_and_identifiers(s2):
    path, tools = s2
    _measure(path, 1)
    _measure(path, 3)
    with hidden_files_sealed():
        everything = tools.get_episode_evidence("all")
        tier2 = tools.get_episode_evidence("tier2")
        ps1 = tools.get_episode_evidence("PS1")
        p1 = tools.get_episode_evidence("P1_MotorPower_W_100Hz")
        unknown = tools.get_episode_evidence("tier4")
    assert len(everything.items) == 24 + 68 and everything.time_window == TimeWindow(start=1, end=2, unit="measurement")
    assert len(tier2.items) == 24 + 36 and {i.release_time for i in tier2.items} == {1, 2}
    assert [i.evidence_id for i in ps1.items] == [f"obs/2/PS1_Pressure_bar_100Hz_{s}" for s in ("max", "mean", "min", "std")]
    assert [i.channel for i in p1.items] == ["EPS1"] * 4 and all(i.unit == "W" for i in p1.items)
    assert isinstance(unknown, ErrorResult)
    assert not any("cycle" in i.evidence_id for i in everything.items) and all(i.band_low <= i.band_high for i in everything.items)


def test_prediction_bookkeeping_and_errors(s2):
    path, tools = s2
    _measure(path, 2)
    with hidden_files_sealed():
        assert isinstance(tools.verify_outcome("pred/1"), ErrorResult)
        assert isinstance(tools.estimate_intervention_effect("replace:motor"), ErrorResult)
        assert isinstance(tools.rank_root_causes(0), ErrorResult)
        watch = tools.estimate_intervention_effect("monitor")
    assert (watch.prediction_id, watch.component, watch.addresses_fault, watch.recovery, watch.price) == ("pred/1", None, 0.0, 0.0, 0.0)
    assert "the environment accepts a repair" not in " ".join(watch.assumptions)
    stored = json.loads((path.parent / "predictions.json").read_text())
    assert [p["id"] for p in stored] == ["pred/1"] and stored[0]["events"] == 1 and stored[0]["measurements"] == 1
    _measure(path, 3)
    with hidden_files_sealed():
        verified = tools.verify_outcome("pred/1")
    assert verified.status == "verified" and verified.observation == 2 and verified.recovered is None and verified.recovery_status.startswith("not applicable")
    assert len(verified.residuals) == 17 and verified.n_outside_tolerated == 1
    assert verified.consistent == (verified.n_outside <= 1)
    assert abs(verified.p_value_outside - binomial_tail(17, verified.n_outside)) < 1e-12


def test_a_replacement_between_prediction_and_measurement_makes_the_prediction_not_comparable(tmp_path):
    path = _start(tmp_path, min_faulty=2, max_faulty=2, allowed=[START], max_repairs=3)
    tools = CausalTools(load_models(), episode.released_path(path))
    _measure(path, 3)
    with hidden_files_sealed():
        effect = tools.estimate_intervention_effect("replace:accumulator")
        watch = tools.estimate_intervention_effect("monitor")
    _act(path, "submit", dict(zip(COMPONENTS, START)), {"type": "submit", "diagnosis": dict(zip(COMPONENTS, START))})
    for component in ("pump", "accumulator"):  # the pump is healthy at the start, so neither repair ends the episode
        assert not _act(path, "replace", {"component": component}, {"type": "replace", "component": component})["episode_over"]
    _measure(path, 3)
    with hidden_files_sealed():
        replaced, monitored = tools.verify_outcome(effect.prediction_id), tools.verify_outcome(watch.prediction_id)
    assert replaced.status == "not comparable: replace:pump came between the prediction and the measurement"
    assert monitored.status.startswith("not comparable: replace:pump, replace:accumulator")
    assert all(v.residuals == [] and v.observation is None and v.consistent is None for v in (replaced, monitored))


def test_verdict_tolerates_one_missed_band():
    # a calibrated prediction keeps all 17 summaries inside their bands with probability (1 - MISS_RATE) ** 17
    assert abs(binomial_tail(17, 1) - (1 - (1 - MISS_RATE) ** 17)) < 1e-12 and abs(binomial_tail(17, 0) - 1) < 1e-12
    for n in range(1, 41):
        k = misses_tolerated(n)
        assert binomial_tail(n, k + 1) <= 0.05 and (k == 0 or binomial_tail(n, k) > 0.05)
    assert outcome_consistent(1, 17, None, 0.0) and not outcome_consistent(2, 17, None, 0.0)
    assert outcome_consistent(0, 17, False, 0.2) and not outcome_consistent(0, 17, True, 0.2)


def test_released_answers_hold_only_what_the_agent_saw(tmp_path):
    path = _start(tmp_path, min_faulty=2, max_faulty=2, allowed=[START])
    released = json.loads(episode.released_path(path).read_text())
    assert released["events"] == [] and released["asset_id"] == "hyd_1" and released["task_text"] is None
    assert released["rules"]["tiers"]["1"] == tier_fields(1) and released["rules"]["prices"]["measure"] == {"1": 1, "2": 3, "3": 10}
    assert released["status"] == {"phase": "investigate", "budget_left": 150, "repairs_left": 2, "steps_left": 200, "episode_over": False}
    _measure(path, 1)
    _act(path, "measure", {"tier": 4}, {"type": "measure", "tier": 4})
    _act(path, "submit", dict(zip(COMPONENTS, START)), {"type": "submit", "diagnosis": dict(zip(COMPONENTS, START))})
    _act(path, "replace", {"component": "valve"}, {"type": "replace", "component": "valve"})
    text = episode.released_path(path).read_text()
    for key in HIDDEN_KEYS:
        assert f'"{key}"' not in text, key
    released = json.loads(text)
    observed = episode.replay(episode.load_state(path)).observation()
    assert released["events"] == [episode._agent_view(e) for e in observed["history"]]
    assert [e["event"] for e in released["events"]] == ["measure", "refused", "submit", "replace"]
    assert set(released["events"][0]) == {"event", "tier", "cost", "result", "repeated"} and set(released["events"][3]) == {"event", "component", "cost"}
    assert released["status"]["phase"] == "repair" and released["status"]["repairs_left"] == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["episode.json", "released.json"]


def test_models_json_is_the_fit_of_the_facts_file():
    stored = json.loads(fit.MODELS_PATH.read_text(encoding="utf-8"))
    fresh = json.loads(json.dumps(fit.fit(default_facts_path())))
    assert stored.pop("generated") and fresh.pop("generated")
    assert fresh == stored


@needs_mcp
def test_server_offers_seven_flat_tools_over_stdio(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from servers.causal_diagnosis.main import build_server

    path = _start(tmp_path, min_faulty=2, max_faulty=2, allowed=[START])
    _measure(path, 1)
    shutil.copy(Path(__file__).resolve().parents[1] / "launcher.py", tmp_path / "causal_server.py")
    src = str(Path(__file__).resolve().parents[3])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p)}

    async def session_calls():
        params = StdioServerParameters(command=sys.executable, args=[str(tmp_path / "causal_server.py")], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {t.name: t.inputSchema for t in (await session.list_tools()).tools}
                context = await session.call_tool("get_asset_context", {"asset_id": "hyd_1"})
                ranking = await session.call_tool("rank_root_causes", {"top_k": 2})
                bad = await session.call_tool("estimate_intervention_effect", {"action": "replace:motor"})
                return tools, context, ranking, bad

    tools, context, ranking, bad = asyncio.run(session_calls())
    assert set(tools) == set(STATIC_TOOLS) | set(EPISODE_TOOLS)
    assert all(schema.get("required", []) == list(schema["properties"]) for schema in tools.values())
    assert not any("default" in p for schema in tools.values() for p in schema["properties"].values())
    assert tools["rank_root_causes"]["properties"]["top_k"]["type"] == "integer" and tools["get_symptom_report"]["properties"] == {}
    assert all(p["type"] == "string" for name in ("get_asset_context", "query_fmea_paths", "get_episode_evidence", "estimate_intervention_effect", "verify_outcome")
               for p in tools[name]["properties"].values())
    assert not context.isError and json.loads(context.content[0].text)["asset_type"] == "HYDRAULIC_PUMP"
    assert not ranking.isError and len(json.loads(ranking.content[0].text)["candidates"]) == 2
    assert not bad.isError and "error" in json.loads(bad.content[0].text)
    assert {t.name for t in asyncio.run(build_server(None).list_tools())} == set(STATIC_TOOLS)
