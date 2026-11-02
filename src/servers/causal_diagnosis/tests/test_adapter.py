"""The four tools over released.json format 2 written by the fixture episode: a worked episode from the empty ranking to a
verified prediction, the rule that no tool opens a hidden file, the three ranking models, the bookkeeping of predictions
and the error answers. No test states a value computed from the rig data: each checks what must hold whatever the data,
such as probabilities that sum to one, intervals that contain their mean, p values in (0, 1] and residual identities."""
import builtins
import io
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from envs.hydraulic_rig.classifiers import COMBOS
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, build_blocks, default_facts_path, load_cycles
from servers.causal_diagnosis import adapter
from servers.causal_diagnosis.adapter import CausalTools, load_models
from servers.causal_diagnosis.effect_model import GROUP_FIELDS, SUMMARY_FIELDS, n_faulty
from servers.causal_diagnosis.models import ErrorResult
from servers.causal_diagnosis.posterior import DEFAULT_RANKING_MODEL, RANKING_MODELS
from servers.causal_diagnosis.tests.fixture_episode import FixtureEpisode, default_rules

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)

START = (100, 80, 0, 90)  # a start with two faulty components, the valve and the accumulator
HIDDEN_FILES = ("episode.json", "log.json", "score.json", "oracle.json", "verdict.json")
HIDDEN_KEYS = ("seed", "allowed", "cycle", "hidden", "initial", "combination_after", "unnecessary", "facts_path", "schedule", "external_events")
N_CANDIDATES = sum(1 for h in COMBOS if 1 <= n_faulty(h) <= 2)


@contextmanager
def hidden_files_sealed():
    """Every open of a hidden episode file, of anything under a hidden folder or of the facts file fails, so a tool that
    reads a hidden source fails its test."""
    facts = Path(default_facts_path()).resolve()
    real_open = builtins.open

    def guarded(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            p = Path(file)
            if p.name in HIDDEN_FILES or "hidden" in p.parts or p.resolve() == facts:
                raise PermissionError(f"a tool opened the hidden file {p}")
        return real_open(file, *args, **kwargs)

    builtins.open = io.open = guarded
    try:
        yield
    finally:
        builtins.open = io.open = real_open


@pytest.fixture(scope="module")
def world():
    cycles = load_cycles(default_facts_path())
    return cycles, build_blocks(cycles)


@pytest.fixture
def episode(tmp_path, world):
    cycles, blocks = world
    return FixtureEpisode(tmp_path, cycles, blocks, START, seed=0)


def tools_for(episode, **kwargs):
    return CausalTools(load_models(), episode.released_path, **kwargs)


def ranking_is_coherent(ranking, n_candidates=N_CANDIDATES):
    """What every ranking must satisfy whatever the data."""
    probs = [c.probability for c in ranking.candidates]
    assert ranking.n_candidates == n_candidates and probs == sorted(probs, reverse=True) and all(0 <= p <= 1 for p in probs)
    assert set(ranking.component_fault_probability) == set(COMPONENTS) == set(ranking.current_fault_probability)
    assert all(0 <= p <= 1 for p in ranking.component_fault_probability.values())
    assert all(ranking.current_fault_probability[c] <= ranking.component_fault_probability[c] + 1e-9 for c in COMPONENTS)
    assert all(abs(sum(levels.values()) - 1) < 1e-9 for levels in ranking.level_probability.values())
    for cand in ranking.candidates:
        off = [(c, v) for c, v in cand.combination.items() if v != HEALTHY[c]]
        assert [(f.component, f.level) for f in cand.faults] == off and all(f.code for f in cand.faults)
        assert all(cand.current_combination[c] in (cand.combination[c], HEALTHY[c]) for c in COMPONENTS)
    for item in ranking.attribution:
        assert 0 <= item.fault_probability <= 1 and item.explained_anomaly >= 0
        assert all(0 <= ch.share <= 1 and ch.anomaly_score >= 0 for ch in item.channels)
    assert ranking.insufficient_evidence == (min(ranking.expected_loss_test.values()) < ranking.expected_loss_act)
    assert (ranking.separating_test is not None) == ranking.insufficient_evidence
    if ranking.separating_test is not None:
        assert ranking.separating_test.sensor_group == min(ranking.expected_loss_test, key=lambda g: (ranking.expected_loss_test[g], g))
        assert ranking.separating_test.cost >= ranking.separating_test.price
    assert abs(ranking.expected_saving - (ranking.expected_loss_act - min(ranking.expected_loss_test.values()))) < 1e-9
    assert set(ranking.separating_channels) == set(ranking.unseparated)
    assert set(ranking.expected_loss_test) == set(ranking.monte_carlo.standard_error) == {"tier1", "tier2", "tier3"}
    assert ranking.ranking_model == DEFAULT_RANKING_MODEL == "joint_block" or ranking.ranking_model in RANKING_MODELS
    if ranking.stability is not None:
        assert 0 <= ranking.stability.interval_low <= ranking.stability.top_marginal + 1e-9
        assert ranking.stability.top_marginal - 1e-9 <= ranking.stability.interval_high <= 1 and 0 <= ranking.stability.top_share <= 1
    assert "no fixed confidence cut" in " ".join(ranking.assumptions) and "0.95" not in " ".join(ranking.assumptions)
    assert "abstention_rule decision" in " ".join(ranking.assumptions)


def test_worked_episode_from_the_empty_ranking_to_a_verified_prediction(episode):
    tools = tools_for(episode)
    with hidden_files_sealed():
        empty = tools.rank_root_causes(3)
        no_evidence = tools.get_episode_evidence("all")
    ranking_is_coherent(empty)
    assert all(abs(c.probability - 1 / N_CANDIDATES) < 1e-12 for c in empty.candidates) and empty.evidence_ids == []
    assert empty.time_window.unit == "cycle" and empty.time_window.start == empty.time_window.end == 0
    assert no_evidence.items == [] and no_evidence.n_observations == 0
    episode.read("tier1")
    with hidden_files_sealed():
        evidence = tools.get_episode_evidence("tier1")
        first = tools.rank_root_causes(3)
    assert evidence.n_observations == 1 and len(evidence.items) == len(episode.observations[0]["values"])
    modeled = [i for i in evidence.items if i.anomaly_score is not None]
    assert len(modeled) == len(GROUP_FIELDS["tier1"]) and all(i.anomaly_score >= 0 for i in modeled)
    means = [i for i in evidence.items if i.statistic == "mean"]
    assert all(i.band_low is not None and i.band_low <= i.band_high and i.inside_band == (i.band_low <= i.value <= i.band_high) for i in means)
    assert all(i.inside_band is None for i in evidence.items if i.statistic != "mean")
    ce = next(i for i in evidence.items if i.field == "CE_CoolingEfficiency_percent_1Hz_mean")
    assert (ce.evidence_id, ce.channel, ce.unit, ce.rate, ce.n, ce.t, ce.sensor_group) == ("obs/1/CE_CoolingEfficiency_percent_1Hz_mean", "CE", "percent", "1Hz", 1, 1, "tier1")
    ranking_is_coherent(first)
    assert first.time_window.start == 1 and first.time_window.end == 1 and len(first.evidence_ids) == len(episode.observations[0]["values"])
    episode.read("tier3", cycles=2)
    with hidden_files_sealed():
        second = tools.rank_root_causes(N_CANDIDATES)
        effect = tools.estimate_intervention_effect("replace:accumulator")
        pending = tools.verify_outcome("pred/1")
    ranking_is_coherent(second)
    assert len(second.candidates) == N_CANDIDATES and abs(sum(c.probability for c in second.candidates) - 1) < 1e-9
    assert dict(zip(COMPONENTS, START)) in [c.combination for c in second.candidates]
    assert second.stability is not None and second.stability.resamples == len(load_models().effect.resampled)
    rules = episode.rules
    assert (effect.prediction_id, effect.price) == ("pred/1", rules["prices"]["parts"]["accumulator"] + rules["prices"]["repair_charge"])
    assert effect.duration == rules["durations"]["lead_time"]["accumulator"] + rules["durations"]["repair_downtime"]["accumulator"]
    assert effect.downtime_during_action == rules["downtime_rate"]["stopped"] * rules["durations"]["repair_downtime"]["accumulator"]
    assert abs(effect.addresses_fault - second.current_fault_probability["accumulator"]) < 1e-9 and 0 <= effect.recovery <= 1
    assert effect.recovery_interval_low <= effect.recovery <= effect.recovery_interval_high and effect.dead_band >= 0
    assert len(effect.predicted) == len(SUMMARY_FIELDS) and all(p.low <= p.mean <= p.high and p.quantile >= 0 for p in effect.predicted)
    assert effect.alpha == adapter.DEFAULT_ALPHA and effect.direction in ("lower", "higher", "no change")
    assert abs(effect.risk_before - (1 - sum(c.probability for c in second.candidates if c.current_combination == dict(HEALTHY)))) < 1e-9
    assert effect.risk_after == 1 - effect.recovery and effect.evidence_ids[-1] == "pred/1"
    assert effect.time_window.start == episode.time and effect.time_window.end == episode.time + effect.duration
    assert "a diagnosis or a declaration" in " ".join(effect.assumptions)
    assert pending.status.startswith("not yet observable") and pending.channels == [] and pending.consistent is None
    top = second.candidates[0].combination
    episode.submit_diagnosis(top, evidence_ids=second.evidence_ids[:3])
    order_id = episode.create_work_order("accumulator", prediction_id="pred/1")
    with hidden_files_sealed():
        waiting = tools.verify_outcome("pred/1")
        with_order = tools.estimate_intervention_effect("replace:accumulator")
    assert waiting.status == "not yet observable: the work order is open and has not closed"
    assert "is open" in " ".join(with_order.assumptions) and with_order.prediction_id == "pred/2"
    episode.advance(effect.duration)
    assert episode.events[-2]["event"] == "work_order_closed" and episode.events[-2]["id"] == order_id
    with hidden_files_sealed():
        unread = tools.verify_outcome("pred/1")
    assert unread.status == "not yet observable: no reading since the action"
    episode.read("tier3")
    with hidden_files_sealed():
        verified = tools.verify_outcome("pred/1")
        third = tools.rank_root_causes(3)
        evidence_after = tools.get_episode_evidence("all")
    assert verified.status == "verified" and verified.observation == 4 and verified.n_channels == len(SUMMARY_FIELDS)
    predicted = {p.field: p for p in effect.predicted}
    observed = episode.observations[-1]["values"]
    for check in verified.channels:
        assert check.predicted == predicted[check.field].mean and check.observed == observed[check.field]
        assert (check.low, check.high) == (predicted[check.field].low, predicted[check.field].high)
        assert 0 < check.p_value <= 1 and check.score >= 0 and (not check.bonferroni or check.holm)
    assert verified.consistent == (not any(c.bonferroni for c in verified.channels)) and verified.bonferroni_rejections <= verified.holm_rejections
    assert verified.alpha == effect.alpha and 0 <= verified.recovery_posterior <= 1 and verified.recovery_predicted == effect.recovery
    assert verified.time_window.start == episode.events[-2]["t_end"] and verified.time_window.end == episode.time
    assert verified.evidence_ids == ["pred/1"] + [f"obs/4/{c.field}" for c in verified.channels]
    assert [c.field for c in verified.channels] == list(SUMMARY_FIELDS)
    ranking_is_coherent(third)
    assert all(c.current_combination["accumulator"] == HEALTHY["accumulator"] for c in third.candidates)
    assert third.current_fault_probability["accumulator"] == 0 and f"wo/{order_id}" in third.evidence_ids
    assert [w.id for w in evidence_after.work_orders_closed] == [order_id] and f"wo/{order_id}" in evidence_after.evidence_ids
    with hidden_files_sealed():
        watch = tools.estimate_intervention_effect("monitor")
    assert (watch.prediction_id, watch.component, watch.addresses_fault, watch.price) == ("pred/3", None, 0.0, 0.0)
    assert watch.duration == rules["durations"]["advance_per_cycle"] and watch.downtime_during_action >= 0
    episode.read("tier1")
    with hidden_files_sealed():
        watched = tools.verify_outcome("pred/3")
    assert watched.status == "verified" and watched.n_channels == len(GROUP_FIELDS["tier1"]) and watched.recovery_posterior is None
    episode.finish("done")
    with hidden_files_sealed():
        after_end = tools.estimate_intervention_effect("replace:valve")
    assert "the episode is over" in " ".join(after_end.assumptions)
    assert sorted(p.name for p in episode.folder.iterdir()) == ["predictions.json", "released.json"]
    stored = json.loads((episode.folder / "predictions.json").read_text())
    assert [p["id"] for p in stored] == ["pred/1", "pred/2", "pred/3", "pred/4"]
    assert stored[0]["events"] == 2 and stored[0]["observations"] == 3 and set(stored[0]["predicted"]) == set(SUMMARY_FIELDS)


def test_a_second_closure_before_the_reading_makes_the_prediction_not_comparable(episode):
    tools = tools_for(episode)
    episode.read("tier3")
    with hidden_files_sealed():
        effect = tools.estimate_intervention_effect("replace:accumulator")
    episode.submit_diagnosis(dict(zip(COMPONENTS, START)))
    episode.create_work_order("accumulator", prediction_id=effect.prediction_id)
    episode.create_work_order("valve")
    episode.advance(10)
    episode.read("tier3")
    with hidden_files_sealed():
        result = tools.verify_outcome(effect.prediction_id)
    assert result.status.startswith("not comparable: replace:valve") and result.channels == [] and result.consistent is None


def test_an_external_replacement_enters_the_posterior_as_a_closed_order(episode):
    tools = tools_for(episode)
    episode.external_replacement("valve", at_t=2)
    episode.read("tier3", cycles=3)
    with hidden_files_sealed():
        ranking = tools.rank_root_causes(3)
        evidence = tools.get_episode_evidence("all")
    ranking_is_coherent(ranking)
    assert ranking.current_fault_probability["valve"] == 0 and "wo/wo-ext-1" in ranking.evidence_ids
    assert [(w.id, w.source, w.t) for w in evidence.work_orders_closed] == [("wo-ext-1", "external", 2)]


@pytest.mark.parametrize("ranking_model", RANKING_MODELS)
def test_every_ranking_model_answers_coherently(episode, ranking_model):
    tools = tools_for(episode, ranking_model=ranking_model)
    episode.read("tier1")
    episode.read("tier3")
    with hidden_files_sealed():
        ranking = tools.rank_root_causes(5)
        effect = tools.estimate_intervention_effect("replace:valve")
    ranking_is_coherent(ranking)
    assert ranking.ranking_model == ranking_model and len(ranking.candidates) == 5
    assert abs(effect.addresses_fault - ranking.current_fault_probability["valve"]) < 1e-9
    if ranking_model == "discriminant_urn":
        assert ranking.stability is None and "discriminant_urn" in " ".join(ranking.assumptions)


def test_evidence_groups_identifiers_and_bands(episode):
    tools = tools_for(episode)
    episode.read("tier1")
    episode.read("tier3")
    with hidden_files_sealed():
        everything = tools.get_episode_evidence("all")
        tier2 = tools.get_episode_evidence("tier2")
        ps1 = tools.get_episode_evidence("PS1")
        p1 = tools.get_episode_evidence("P1_MotorPower_W_100Hz")
        unknown = tools.get_episode_evidence("tier4")
    per_observation = [len(o["values"]) for o in episode.observations]
    assert len(everything.items) == sum(per_observation) and everything.time_window.start == 1 and everything.time_window.end == 2
    tier2_channels = set(episode.rules["sensor_groups"]["tier2"])
    assert all(adapter.channel_field(i.field) in tier2_channels for i in tier2.items) and {i.n for i in tier2.items} == {1, 2}
    assert [i.evidence_id for i in ps1.items] == [f"obs/2/PS1_Pressure_bar_100Hz_{s}" for s in ("mean", "std")]
    assert all(i.channel == "EPS1" and i.unit == "W" for i in p1.items) and len(p1.items) == len(episode.rules["statistics"])
    assert isinstance(unknown, ErrorResult)
    assert not any("cycle" in i.evidence_id for i in everything.items)
    text = episode.released_path.read_text()
    for key in HIDDEN_KEYS:
        assert f'"{key}"' not in text, key


def test_errors_and_the_server_without_an_episode(episode, tmp_path):
    tools = tools_for(episode)
    episode.read("tier2")
    with hidden_files_sealed():
        assert isinstance(tools.verify_outcome("pred/9"), ErrorResult)
        assert isinstance(tools.estimate_intervention_effect("replace:motor"), ErrorResult)
        assert isinstance(tools.estimate_intervention_effect("inspect"), ErrorResult)
        assert isinstance(tools.rank_root_causes(0), ErrorResult)
        assert isinstance(tools.rank_root_causes(True), ErrorResult)
    none = CausalTools(load_models())
    for call in (lambda: none.get_episode_evidence("all"), lambda: none.rank_root_causes(3), lambda: none.estimate_intervention_effect("monitor"),
                 lambda: none.verify_outcome("pred/1")):
        assert isinstance(call(), ErrorResult)
    stale = tmp_path / "stale" / "released.json"
    stale.parent.mkdir()
    stale.write_text(json.dumps({"format": 1}))
    assert "format" in CausalTools(load_models(), stale).rank_root_causes(1).error
    with pytest.raises(ValueError):
        CausalTools(load_models(), episode.released_path, ranking_model="oracle")


def test_models_json_is_the_fit_of_the_facts_file_in_format_2():
    from servers.causal_diagnosis import fit

    stored = json.loads(fit.MODELS_PATH.read_text(encoding="utf-8"))
    assert stored["format"] == fit.MODELS_FORMAT and set(stored["ranking_models"]) == set(RANKING_MODELS)
    assert set(stored["pool"]["combinations"]).isdisjoint(stored["pool"]["evaluation"])
    assert len(stored["pool"]["combinations"]) + len(stored["pool"]["evaluation"]) == len(COMBOS)
    fresh = json.loads(json.dumps(fit.fit(default_facts_path(), resamples=2)))
    for key in ("facts_sha256", "records", "pool", "components", "levels", "healthy", "discriminant_urn"):
        assert fresh[key] == stored[key], key
    assert fresh["effect"]["beta"] == stored["effect"]["beta"] and fresh["effect"]["interactions"] == stored["effect"]["interactions"]
    assert fresh["effect"]["resampled"] == stored["effect"]["resampled"][:2]


def test_the_default_ranking_model_is_joint_block_and_the_models_file_names_it(episode):
    ranking = tools_for(episode).rank_root_causes(1)
    assert ranking.ranking_model == "joint_block" and "joint_block" in " ".join(ranking.assumptions)
    assert load_models().raw["default_ranking_model"] == "joint_block"


def test_the_fixed_cut_abstention_rule_reproduces_v1_for_ablation_a8(episode):
    cut = 0.9
    tools = tools_for(episode, abstention_rule="fixed_cut", abstention_cut=cut)
    episode.read("tier1")
    with hidden_files_sealed():
        ranking = tools.rank_root_causes(3)
    top = ranking.candidates[0].probability
    assert ranking.insufficient_evidence == (top < cut)
    cheapest = min(episode.rules["prices"]["read"], key=lambda g: (episode.rules["prices"]["read"][g], g))
    if ranking.insufficient_evidence:
        assert ranking.separating_test.sensor_group == cheapest
    assert ranking.unseparated == [c for c in COMPONENTS if max(ranking.level_probability[c].values()) < cut]
    assert all(fields == list(GROUP_FIELDS[cheapest]) for fields in ranking.separating_channels.values())
    text = " ".join(ranking.assumptions)
    assert "abstention_rule fixed_cut" in text and f"cut {cut:g}" in text and "ablation A8" in text
    assert ranking.expected_loss_act >= 0 and set(ranking.expected_loss_test) == {"tier1", "tier2", "tier3"}
    with pytest.raises(ValueError):
        tools_for(episode, abstention_rule="coin")
    with pytest.raises(ValueError):
        tools_for(episode, abstention_cut=1.5)
