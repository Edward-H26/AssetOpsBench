"""The four tools over released.json format 3 written by the fixture episode: a worked episode from the empty ranking to a
verified prediction, the rule that no tool opens a hidden file, the three ranking models, the bookkeeping of predictions,
the fields of version 7 (the belief over every combination, its effective number and cycle, the information gain and the
expected elimination of every read and the read batch), the time budget of a call and the error answers. No test states
a value computed from the rig data: each checks what must hold whatever the data, such as probabilities that sum to one,
intervals that contain their mean, p values in (0, 1] and residual identities."""
import builtins
import io
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

import time

from CausalDiagnosis.datasets.dataset_loader.recordings import DataFileError
from CausalDiagnosis.environment.reference_policies.tier_classifiers import COMBOS
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY, build_blocks, default_facts_path, load_cycles
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import tools as adapter
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import CausalTools
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.fitted_models import load_models
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import profile_view
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.troubleshooting_plans import plan_completion, plan_terms
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tool_result_schemas import ErrorResult, RootCauses
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import DEFAULT_RANKING_MODEL, RANKING_MODELS, Posterior
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.fixture_episode import FixtureEpisode, assert_compact_shortens, default_rules, episode_models, with_misdiagnosis

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)

START = (100, 80, 0, 90)  # a start with two faulty components, the valve and the accumulator
HIDDEN_FILES = ("episode.json", "log.json", "score.json", "oracle.json", "verdict.json")
HIDDEN_KEYS = ("seed", "allowed", "cycle", "hidden", "initial", "combination_after", "unnecessary", "facts_path", "schedule")
N_CANDIDATES = len(COMBOS)  # every combination of the profile, whatever the fault rule of the starts


def fields_of(group=None):
    space = load_models().space
    return space.fields if group is None else space.groups[group]


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
    return CausalTools(episode_models(episode.start), episode.released_path, **kwargs)


def ranking_is_coherent(ranking, n_candidates=N_CANDIDATES):
    """What every ranking must satisfy whatever the data: the one read of the rule of the server at the top level, and E10 and
    E22 in the block e10_rule of the long forms. The compact reply, the default, holds none of the keys it leaves out, one
    assumption text and no edge cutting score."""
    compact = ranking.assumptions == [adapter.A_COMPACT]
    probs = [c.probability for c in ranking.candidates]
    assert ranking.n_candidates == n_candidates and probs == sorted(probs, reverse=True) and all(0 <= p <= 1 for p in probs)
    assert set(ranking.current_fault_probability) == set(COMPONENTS) and all(0 <= p <= 1 for p in ranking.current_fault_probability.values())
    assert compact or all(abs(sum(levels.values()) - 1) < 1e-9 for levels in ranking.level_probability.values())
    for cand in ranking.candidates:
        off = [(c, v) for c, v in cand.combination.items() if v != HEALTHY[c]]
        assert [(f.component, f.level) for f in cand.faults] == off and all(f.code for f in cand.faults)
        assert cand.current_combination == cand.combination  # the hypothesis is the combination now
    for item in ranking.attribution or []:
        assert item.current_fault_probability == ranking.current_fault_probability[item.component] and item.explained_anomaly >= 0
        assert all(0 <= ch.share <= 1 and ch.anomaly_score >= 0 for ch in item.channels)
    assert 1 - 1e-9 <= ranking.effective_hypotheses <= n_candidates + 1e-9 and ranking.n_states >= 1
    assert ranking.belief_time == ranking.time_window.end and (compact or f"{ranking.n_states} augmented states" in " ".join(ranking.assumptions))
    assert [e.sensor_group for e in ranking.elimination] == sorted(ranking.expected_loss_test)
    for e in ranking.elimination:
        assert e.cycles == 1 and e.cost == ranking.costs["test_cost"][e.sensor_group] > 0 and e.standard_error >= 0
        assert e.cost_unit == {"money": "money", "calls": "call"}[ranking.costs["unit"]]
        assert abs(e.effective_before - e.expected_effective_after - e.expected_elimination) < 1e-9 or e.expected_elimination == 0
        assert abs(e.elimination_per_cost - e.expected_elimination / e.cost) < 1e-12
        assert e.information_gain >= 0 and abs(e.ig_per_cost - e.information_gain / e.cost) < 1e-12
        if compact:  # the edge cutting scores are computed in the long forms alone
            assert e.ec2 is None and e.ec2_per_cost is None
        else:  # an edge weight between classes is at most one half
            assert 0 <= e.ec2 <= 0.5 and abs(e.ec2_per_cost - e.ec2 / e.cost) < 1e-12
    rows = {e.sensor_group: e for e in ranking.elimination}
    if ranking.read_batch:  # the one read of the rule over every cycle of the group it names
        assert ranking.read_first
        assert [(b.sensor_group, b.cycles) for b in ranking.read_batch] == [(ranking.separating_test.sensor_group, ranking.separating_test.cycles)]
    rule = ranking.e10_rule
    if rule is not None and rule.read_batch:
        assert rule.read_first and rule.batch_cap is None and all(b.cycles == 1 for b in rule.read_batch)
        head = rows[rule.read_batch[0].sensor_group]
        assert head.ig_per_cost > adapter.IG_EPSILON and head.ig_per_cost == max(e.ig_per_cost for e in ranking.elimination)
    for b in ranking.read_batch + (rule.read_batch if rule is not None else []):
        assert (b.information_gain, b.ig_per_cost, b.elimination_per_cost) == (rows[b.sensor_group].information_gain, rows[b.sensor_group].ig_per_cost,
                                                                             rows[b.sensor_group].elimination_per_cost)
    joined = " ".join(ranking.assumptions)
    assert "fault rule" not in joined and "min_faulty" not in joined and "max_faulty" not in joined
    assert compact or "the server never reads a seed or the split, and no result names the start" in joined  # one text on both tables
    # the flag is the one read of the rule, so the flag, the test, the batch and the advice say one thing
    assert ranking.read_first == bool(ranking.read_batch)
    assert ranking.advice is None or (ranking.advice.action == "read") == ranking.read_first
    assert (ranking.separating_test is not None) == ranking.read_first
    if ranking.separating_test is not None:
        assert ranking.separating_test.cost >= ranking.separating_test.price
    if rule is not None:  # E10 with the margin of the card, here the default margin of zero
        assert rule.margin == 0.0 and rule.read_first == (min(ranking.expected_loss_test.values()) < ranking.expected_loss_act)
        assert (rule.separating_test is not None) == rule.read_first
        if rule.separating_test is not None:
            assert rule.separating_test.cycles == 1
            assert rule.separating_test.sensor_group == min(ranking.expected_loss_test, key=lambda g: (ranking.expected_loss_test[g], g))
    assert abs(ranking.expected_saving - (ranking.expected_loss_act - min(ranking.expected_loss_test.values()))) < 1e-9
    assert set(ranking.separating_channels) == set(ranking.unseparated)
    assert set(ranking.expected_loss_test) == {"tier1", "tier2", "tier3"}
    assert compact or set(ranking.monte_carlo.standard_error) == set(ranking.expected_loss_test)
    assert ranking.ranking_model == DEFAULT_RANKING_MODEL == "joint_block" or ranking.ranking_model in RANKING_MODELS
    if ranking.stability is not None:  # a range over the resampled fits of the call, which need not hold the marginal of the main fit
        assert 0 <= ranking.stability.interval_low <= ranking.stability.interval_high <= 1 and 0 <= ranking.stability.top_share <= 1
        assert ranking.stability.resamples == ranking.resampled_fits
    assert "0.95" not in joined and (compact or "abstention_rule decision" in joined)
    assert not compact or all(getattr(ranking, key) is None for key in (*adapter.COMPACT_LEFT_OUT, "e10_rule"))


def test_worked_episode_from_the_empty_ranking_to_a_verified_prediction(episode):
    tools = tools_for(episode)
    with hidden_files_sealed():
        empty = tools.rank_root_causes(3)
        no_evidence = tools.get_episode_evidence("all")
    ranking_is_coherent(empty)
    assert all(abs(c.probability - 1 / N_CANDIDATES) < 1e-12 for c in empty.candidates) and empty.evidence_ids == []
    assert abs(empty.effective_hypotheses - N_CANDIDATES) < 1e-6 and empty.belief_time == 0 and empty.n_states == N_CANDIDATES
    assert empty.time_window.unit == "cycle" and empty.time_window.start == empty.time_window.end == 0
    assert no_evidence.items == [] and no_evidence.n_observations == 0
    episode.read("tier1")
    with hidden_files_sealed():
        evidence = tools.get_episode_evidence("tier1")
        first = tools.rank_root_causes(3)
    assert evidence.n_observations == 1 and len(evidence.items) == len(episode.observations[0]["values"])
    modeled = [i for i in evidence.items if i.anomaly_score is not None]
    assert len(modeled) == len(fields_of("tier1")) and all(i.anomaly_score >= 0 for i in modeled)
    means = [i for i in evidence.items if i.statistic == "mean"]
    assert all(i.normal_low is not None and i.normal_low <= i.normal_high and i.in_normal_range == (i.normal_low <= i.value <= i.normal_high) for i in means)
    assert all(i.in_normal_range is None for i in evidence.items if i.statistic != "mean")
    ce = next(i for i in evidence.items if i.field == "CE_CoolingEfficiency_percent_1Hz_mean")
    assert (ce.evidence_id, ce.channel, ce.unit, ce.rate, ce.n, ce.t, ce.sensor_group) == ("obs/1/CE_CoolingEfficiency_percent_1Hz_mean", "CE", "percent", "1Hz", 1, 0, "tier1")
    ranking_is_coherent(first)
    assert first.time_window.start == 0 and first.time_window.end == 1 and len(first.evidence_ids) == len(episode.observations[0]["values"])
    assert first.effective_hypotheses < empty.effective_hypotheses
    episode.read("tier3", cycles=2)
    with hidden_files_sealed():
        second = tools.rank_root_causes(N_CANDIDATES, "summary")
        effect = tools.estimate_intervention_effect("replace:accumulator")
        pending = tools.verify_outcome("pred/1")
    ranking_is_coherent(second)
    # a top_k of every combination returns every combination with positive mass and leaves none out
    assert second.more_candidates.count == 0 and len(second.candidates) <= N_CANDIDATES and all(c.probability > 0 for c in second.candidates)
    assert abs(sum(c.probability for c in second.candidates) - 1) < 1e-9
    assert dict(zip(COMPONENTS, START)) in [c.combination for c in second.candidates]
    assert second.stability is not None and second.stability.resamples == second.resampled_fits == load_models().fit["resamples"]
    rules = episode.rules
    price = rules["prices"]["parts"]["accumulator"] + rules["prices"]["repair_charge"] + rules["prices"].get("visit", 0)
    assert (effect.prediction_id, effect.price) == ("pred/1", price)
    assert effect.duration == rules["durations"]["lead_time"]["accumulator"] + rules["durations"]["repair_downtime"]["accumulator"]
    rate = second.costs["expected_downtime_rate"]
    assert abs(effect.downtime_during_action - (rate * rules["durations"]["lead_time"]["accumulator"]
                                                + rules["downtime_rate"]["stopped"] * rules["durations"]["repair_downtime"]["accumulator"])) < 1e-9
    assert abs(effect.addresses_fault - second.current_fault_probability["accumulator"]) < 1e-9 and 0 <= effect.recovery <= 1
    assert effect.recovery_interval_low <= effect.recovery <= effect.recovery_interval_high and effect.dead_band >= 0
    assert len(effect.predicted) == len(fields_of()) and all(p.low <= p.mean <= p.high and p.quantile >= 0 for p in effect.predicted)
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
    assert verified.status == "verified" and verified.observation == 4 and verified.n_channels == len(fields_of())
    predicted = {p.field: p for p in effect.predicted}
    observed = episode.observations[-1]["values"]
    for check in verified.channels:
        assert check.predicted == predicted[check.field].mean and check.observed == observed[check.field]
        assert (check.low, check.high) == (predicted[check.field].low, predicted[check.field].high)
        assert check.score >= 0
    # one combined score per reading, the largest standardized deviation, and one conformal p value
    assert verified.combined_score == max(c.score for c in verified.channels) and 0 < verified.p_value <= 1
    assert verified.consistent == (verified.p_value > verified.alpha) and verified.uncertainty.kind == "conformal_p_value"
    assert verified.alpha == effect.alpha and 0 <= verified.recovery_posterior <= 1 and verified.recovery_predicted == effect.recovery
    # the predictions are those of the most probable combination after the repair, which the verification names
    named = tuple(verified.predicted_combination[c] for c in COMPONENTS)
    mu, index = tools.models.effect.mu(named), tools.models.space.field_index
    assert all(p.mean == mu[index[p.field]] for p in effect.predicted) and 0 < verified.predicted_mass <= 1
    named_text = ", ".join(f"{c} {v}" for c, v in verified.predicted_combination.items())
    # every prediction is named as one in the messages a reader sees
    assert f"for ({named_text}), the predicted combination, the most probable after the action, with mass {verified.predicted_mass:.3f}:" in verified.message
    assert f"predicted recovery {effect.recovery:.3f}" in effect.message and f"predicted risk {effect.risk_after:.3f} after the action" in effect.message
    assert f"{named_text}), the predicted combination, the most probable after the action, with mass" in effect.message
    assert verified.time_window.start == episode.events[-2]["t_end"] and verified.time_window.end == episode.observations[-1]["t"]
    assert verified.evidence_ids == ["pred/1"] + [f"obs/4/{c.field}" for c in verified.channels]
    assert [c.field for c in verified.channels] == list(fields_of())
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
    assert watched.status == "verified" and watched.n_channels == len(fields_of("tier1")) and watched.recovery_posterior is None
    episode.finish("done")
    with hidden_files_sealed():
        after_end = tools.estimate_intervention_effect("replace:valve")
    assert "the episode is over" in " ".join(after_end.assumptions)
    assert sorted(p.name for p in episode.folder.iterdir()) == ["predictions.json", "predictions.json.lock", "released.json"]
    # the tool's own records keep their numbering beside the records of the troubleshooting plans, which carry a branch key
    stored = [p for p in json.loads((episode.folder / "predictions.json").read_text()) if "branch" not in p]
    assert [p["id"] for p in stored] == ["pred/1", "pred/2", "pred/3", "pred/4"]
    assert stored[0]["events"] == 2 and stored[0]["observations"] == 3 and set(stored[0]["predicted"]) == set(fields_of())


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


@pytest.mark.parametrize("ranking_model", RANKING_MODELS)
def test_every_ranking_model_answers_coherently(episode, ranking_model):
    tools = tools_for(episode, ranking_model=ranking_model)
    episode.read("tier1")
    episode.read("tier3")
    with hidden_files_sealed():
        ranking = tools.rank_root_causes(5, "summary")
        effect = tools.estimate_intervention_effect("replace:valve")
    ranking_is_coherent(ranking)
    assert ranking.ranking_model == ranking_model and len(ranking.candidates) == 5
    assert abs(effect.addresses_fault - ranking.current_fault_probability["valve"]) < 1e-9
    if ranking_model == "discriminant_urn":
        assert ranking.stability is None and "discriminant_urn" in " ".join(ranking.assumptions)


def test_the_fixture_releases_the_normal_ranges_of_the_environment(world):
    """The fixture's normal range of every channel mean is the one the environment releases, computed on the served values,
    so no served healthy cycle lies outside its normal range."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.fixture_episode import normal_ranges

    cycles, blocks = world
    assert normal_ranges(cycles, blocks) == HydraulicRigEnv(cycles, blocks).normal_ranges


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
    assert len(everything.items) == sum(per_observation) and everything.time_window.start == 0 and everything.time_window.end == 2
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
    none = CausalTools(episode_models(START))
    for call in (lambda: none.get_episode_evidence("all"), lambda: none.rank_root_causes(3), lambda: none.estimate_intervention_effect("monitor"),
                 lambda: none.verify_outcome("pred/1")):
        assert isinstance(call(), ErrorResult)
    stale = tmp_path / "stale" / "released.json"
    stale.parent.mkdir()
    stale.write_text(json.dumps({"format": 1}))
    assert "format" in CausalTools(episode_models(START), stale).rank_root_causes(1).error
    unlaunched = CausalTools(None, episode.released_path)
    for call in (lambda: unlaunched.get_episode_evidence("all"), lambda: unlaunched.rank_root_causes(1), lambda: unlaunched.verify_outcome("pred/1")):
        assert "launched without the block its fits leave out" in call().error
    with pytest.raises(ValueError):
        CausalTools(episode_models(START), episode.released_path, ranking_model="oracle")


def test_models_json_holds_the_blocks_the_folds_and_the_form_of_each_recorded_combination_in_format_8():
    """The fits of an episode are built at launch, so models.json stores what they read: the stable cycles of every recorded
    combination, the folds of the cross fit, the development blocks, which hold no start of the test set, and the form of every
    start with the blocks each choice left out, which the rule recomputes on the facts file, with where each part of the form
    was chosen and the hash of the chosen forms. No fit mode, held out fault count or stored fit is left, and the two rule texts
    of the file are the ones that the fit writes."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import effect_model as em
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import fit_models as fit
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import LIVE_WINDOW, development_blocks, folds_of, recorded_blocks, start_sets

    stored = json.loads(fit.MODELS_PATH.read_text(encoding="utf-8"))
    assert stored["format"] == fit.MODELS_FORMAT == 8 and set(stored["ranking_models"]) == set(RANKING_MODELS)
    assert set(stored) == {"format", "generated", "facts_sha256", "records", "fit", "space", "ranking_models", "default_ranking_model", "folds",
                           "blocks", "selections", "form", "code"}
    assert set(stored["fit"]) == {"rule", "fitted", "cycles", "folds", "fold_seed", "resamples", "resample_seed"}
    assert stored["fit"]["fitted"] == len(COMBOS) == len(stored["blocks"]) == len(stored["selections"]) == len(stored["folds"])
    assert "fault rule" not in stored["fit"]["rule"] and "evaluation" not in json.dumps(stored["fit"])
    assert stored["facts_sha256"] == fit.facts_sha256(default_facts_path())
    view = profile_view()
    blocks, _records = recorded_blocks(view)
    assert all(np.array_equal(np.asarray(stored["blocks"][em.combo_key(h)]), blocks[h]) for h in blocks)
    fold_of = folds_of(view, em.FOLDS, em.FOLD_SEED)
    assert {key: fold for key, fold in stored["folds"].items()} == {em.combo_key(h): fold_of[h] for h in blocks}
    space, form = load_models().space, stored["form"]
    assert set(form) == {"rule", "window", "covariances", "block_effects", "development", "parts", "sha256"}
    assert (form["window"], form["covariances"], form["block_effects"]) == (LIVE_WINDOW, list(em.COVARIANCE_ESTIMATORS), list(em.BLOCK_EFFECTS))
    assert (form["rule"], stored["fit"]["rule"]) == (fit.FORM_RULE, fit.RULE)
    development = [em.parse_combo(key) for key in form["development"]]
    starts, test = start_sets(view, blocks)
    assert development == development_blocks(view, blocks) and set(starts) < set(development) and not set(test) & set(development)
    assert sorted([*development, *test]) == sorted(blocks)
    identified = em.model_form.development_identifies(space, development, fold_of)
    assert form["parts"] == {"pairs": {"chosen": "development" if identified else "per_start", "development_identifies": identified},
                             "covariance": {"chosen": "development"}, "block_effect": {"chosen": "registered", "value": DEFAULT_RANKING_MODEL}}
    assert form["sha256"] == fit.form_sha256(stored) != fit.form_sha256({**stored, "form": {**form, "development": form["development"][1:]}})
    assert not identified  # the worked instance: the rig's pairs are chosen once per start

    def named(pairs):
        return [[space.components[a], space.components[b]] for a, b in pairs]

    for start in (START, view.healthy_combination(), (3, 73, 2, 90)):
        selection = stored["selections"][em.combo_key(start)]
        second = em.recorded_blocks.second_block(start, [h for h in blocks if h != start])
        pairs, log = em.select_pool_pairs(space, blocks, [start, second], fold_of)
        assert selection["pairs"] == named(pairs) and selection["log"] == json.loads(json.dumps(log))
        assert selection["left_out"] == [em.combo_key(start), em.combo_key(second)]
        assert em.fit_episode(load_models().recorded, [], start, 0, 0).second == second
        own = [start] if start in development else []
        chosen = em.select_form(space, blocks, [h for h in development if h != start], fold_of, form["window"])
        assert {k: selection[k] for k in ("covariance", "fallback", "held_out_window_dof", "forms", "block_effects")} == json.loads(json.dumps(
            {k: chosen[k] for k in ("covariance", "fallback", "held_out_window_dof", "forms", "block_effects")}))
        assert selection["covariance_left_out"] == [em.combo_key(h) for h in own] and selection["report_pairs"] == named(chosen["pairs"])
    assert stored["selections"][em.combo_key(START)]["covariance_left_out"] == [] and len(stored["selections"][em.combo_key((3, 73, 2, 90))]["covariance_left_out"]) == 1


def test_models_fitted_on_another_data_file_are_refused_and_never_refitted_in_place(tmp_path, monkeypatch):
    """A models file of the current format that names another data file than the one in use is refused with the sentence
    that says how it is fitted again, and it is neither fitted nor written in place, so that no stale file is served and none
    is replaced unseen."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import fit_models as fit

    stored = json.loads(fit.MODELS_PATH.read_text(encoding="utf-8"))
    copy = tmp_path / "models.json"
    copy.write_text(json.dumps({**stored, "facts_sha256": "0" * 64}), encoding="utf-8")
    before, calls = copy.read_bytes(), []
    monkeypatch.setattr(fit, "fit", lambda *args, **kwargs: calls.append("fit"))
    monkeypatch.setattr(fit, "write_models", lambda *args, **kwargs: calls.append("write_models"))
    with pytest.raises(DataFileError) as refused:
        fit.ensure_models(copy)
    assert fit.REFIT in str(refused.value) and "0" * 64 in str(refused.value)
    assert calls == [] and copy.read_bytes() == before


def test_ensure_models_fits_a_missing_file_on_the_data_file_it_is_given(tmp_path, monkeypatch):
    """A missing models file is fitted on the data file the caller names, and the file it writes names that data file, so a
    second call compares and fits nothing."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import fit_models as fit

    given = tmp_path / "recordings.jsonl"
    given.write_text("{}\n", encoding="utf-8")
    asked = []

    def fitted(facts_path, **options):
        asked.append(facts_path)
        return {"format": fit.MODELS_FORMAT, "facts_sha256": fit.facts_sha256(facts_path)}

    monkeypatch.setattr(fit, "fit", fitted)
    path = tmp_path / "models.json"
    assert fit.ensure_models(path, facts_path=given) == path and asked == [given]
    assert json.loads(path.read_text(encoding="utf-8")) == {"format": fit.MODELS_FORMAT, "facts_sha256": fit.facts_sha256(given),
                                                      "code": fit.code_digest()}
    assert fit.ensure_models(path, facts_path=given) == path and asked == [given]


def test_the_default_ranking_model_is_joint_block_and_the_models_file_names_it(episode):
    ranking = tools_for(episode).rank_root_causes(1, "summary")
    assert ranking.ranking_model == "joint_block" and "joint_block" in " ".join(ranking.assumptions)
    assert load_models().raw["default_ranking_model"] == "joint_block"


def test_the_fixed_cut_abstention_rule_reproduces_v1_for_the_ablation_of_the_stop_rule(episode):
    cut = 0.9
    tools = tools_for(episode, abstention_rule="fixed_cut", abstention_cut=cut)
    episode.read("tier1")
    with hidden_files_sealed():
        ranking = tools.rank_root_causes(3, "summary")
    top = ranking.candidates[0].probability
    assert ranking.read_first == (top < cut)
    cheapest = min(episode.rules["prices"]["read"], key=lambda g: (episode.rules["prices"]["read"][g], g))
    if ranking.read_first:  # the read of the rule while the top mass lies below the cut, the cheapest group once none is left
        assert ranking.separating_test.sensor_group == (ranking.read_batch[0].sensor_group if ranking.read_batch else cheapest)
    assert ranking.unseparated == [c for c in COMPONENTS if max(ranking.level_probability[c].values()) < cut]
    assert all(fields == list(fields_of(cheapest)) for fields in ranking.separating_channels.values())
    text = " ".join(ranking.assumptions)
    assert "abstention_rule fixed_cut" in text and f"cut {cut:g}" in text
    assert ranking.expected_loss_act >= 0 and set(ranking.expected_loss_test) == {"tier1", "tier2", "tier3"}
    with pytest.raises(ValueError):
        tools_for(episode, abstention_rule="coin")
    with pytest.raises(ValueError):
        tools_for(episode, abstention_cut=1.5)


@pytest.mark.parametrize("processes", [{}, {"hazard": {"mode": "stochastic"}}, {"schedule": {"mode": "seeded"}}])
def test_a_ranking_runs_every_registered_fit_and_prints_its_state_count(world, tmp_path, processes):
    """The work of a call is fixed by the evidence and has no bound of its own: the filter carries
    every registered resampled fit of E6, which the state text of the reply states, after many cycles under the heaviest rule
    cards too. The seconds of the call go to
    the log of the test, since a wall clock reads the load of the machine and not the code."""
    cycles, blocks = world
    config = EpisodeConfig().to_json()
    rules = default_rules(cycles, blocks, processes={key: {**config[key], **value} for key, value in processes.items()})
    episode = FixtureEpisode(tmp_path, cycles, blocks, START, seed=0, rules=rules)
    for step in range(6):
        episode.read("tier3" if step % 2 else "tier1")
        episode.advance(4)
    tools = tools_for(episode)
    started = time.perf_counter()
    ranking = tools.rank_root_causes(5, "summary")
    elapsed = time.perf_counter() - started
    ranking_is_coherent(ranking)
    print(f"rank_root_causes answered in {elapsed:.2f} s with {ranking.n_states} states and {ranking.resampled_fits} resampled fits")
    assert ranking.resampled_fits == tools.models.effect.resample_count > 0
    assert not hasattr(adapter, "RESAMPLE_WORK") and any("every registered one" in a for a in ranking.assumptions)
    if processes:
        assert ranking.n_states > N_CANDIDATES and any("from the released rules" in a for a in ranking.assumptions)


def free_rules(cycles, blocks):
    """The fixture's rules on the free base row of the environment table: every price zero, the downtime rates kept."""
    priced = default_rules(cycles, blocks)
    free = {"read": {g: 0 for g in priced["prices"]["read"]}, "parts": {c: 0 for c in priced["prices"]["parts"]}, "repair_charge": 0}
    return default_rules(cycles, blocks, prices=free)


def test_the_read_cost_of_e21_follows_the_row_of_the_environment_table(world, tmp_path):
    """From the priced row on, the information gain and the expected elimination of a read are divided by
    the test cost of E7 in money, and on the free base row every cost of E7 to E10 counts calls, so a read costs one call, a missed fault the
    calls left before the step cap, a repair plan one call per order, no downtime is charged, and an act off the truth costs
    the calls left once, what a failed episode costs, while the priced row names the price of a wrong diagnosis it read from
    the rules."""
    cycles, blocks = world
    for name, rules, unit in (("priced", default_rules(cycles, blocks), "money"), ("base", free_rules(cycles, blocks), "call")):
        episode = FixtureEpisode(tmp_path / name, cycles, blocks, START, seed=0, rules=rules)
        episode.read("tier1")
        ranking = tools_for(episode).rank_root_causes(2, "summary")
        ranking_is_coherent(ranking)
        assert {e.cost_unit for e in ranking.elimination} == {unit} and all(b.cost_unit == unit for b in ranking.read_batch)
        if unit == "call":
            steps_left = json.loads(episode.released_path.read_text(encoding="utf-8"))["status"]["steps_left"]
            assert ranking.costs["unit"] == "calls" and ranking.costs["remaining_horizon"] == steps_left
            assert all(e.cost == 1.0 and e.price == 1.0 for e in ranking.elimination) and ranking.costs["expected_downtime_rate"] == 0
            for branch in ranking.branches:
                plan = branch.repair_plan
                assert plan.price == len(plan.steps) and plan.waiting_cost == plan.downtime_cost == 0
            assert any("E7 to E10 in calls" in a and f"costs the {steps_left} calls left before the step cap, what a failed episode costs, once" in a
                       for a in ranking.assumptions)
        else:
            assert any("a wrong diagnosis, a combination off the truth in any level, costs 0," in a for a in ranking.assumptions)


def test_a_priced_wrong_diagnosis_raises_the_loss_of_acting_by_its_price_times_the_chance_the_top_is_wrong(world, tmp_path):
    """The ranking reads prices.misdiagnosis from the released rules, so on the same evidence acting now
    costs that price times one minus the top mass more than under rules without it, the decision stays coherent, and
    the cost text names the price."""
    cycles, blocks = world
    price = profile_view().visit_charge()
    rankings = {}
    for key in (None, price):
        episode = FixtureEpisode(tmp_path / f"price_{key}", cycles, blocks, START, seed=0,
                                 rules=with_misdiagnosis(default_rules(cycles, blocks), key))
        episode.read("tier1")
        with hidden_files_sealed():
            rankings[key] = tools_for(episode).rank_root_causes(3, "summary")
    plain, priced = rankings[None], rankings[price]
    ranking_is_coherent(priced)
    top = priced.candidates[0].probability
    assert price > 0 and top == plain.candidates[0].probability
    assert abs(priced.expected_loss_act - plain.expected_loss_act - price * (1 - top)) < 1e-9
    assert any(f"a wrong diagnosis, a combination off the truth in any level, costs {price:g}," in a for a in priced.assumptions)


def test_the_free_base_row_bounds_the_read_batch_by_the_calls_left(world, tmp_path):
    """On the free base row the status carries no budget, so the reads of a batch and the reserve of the top repair plan,
    one call per order, must fit the calls left before the step cap."""
    cycles, blocks = world
    episode = FixtureEpisode(tmp_path, cycles, blocks, START, seed=0, rules=free_rules(cycles, blocks))
    episode.read("tier1")
    released = json.loads(episode.released_path.read_text(encoding="utf-8"))
    released["status"].pop("budget_left")
    for steps_left, bounded in ((released["status"]["steps_left"], False), (1, True)):
        released["status"]["steps_left"] = steps_left
        episode.released_path.write_text(json.dumps(released), encoding="utf-8")
        ranking = tools_for(episode).rank_root_causes(3)
        ranking_is_coherent(ranking)
        if bounded:  # one call left cannot pay a read and the reserve of a plan of at least one order
            assert ranking.read_batch == []
        else:
            assert bool(ranking.read_batch) == ranking.read_first


def test_the_view_is_built_for_the_asset_the_released_file_names(episode, monkeypatch):
    """write_released puts asset_id beside the rules, not inside them, so the adapter hands it to profile_view, and a
    released file that names no asset reads as the default asset."""
    asked = []
    real = adapter.causal_tools.profile_view
    monkeypatch.setattr(adapter.causal_tools, "profile_view", lambda rules=None: asked.append(dict(rules)) or real(rules))
    raw = json.loads(episode.released_path.read_text(encoding="utf-8"))
    assert "asset_id" not in raw["rules"]
    tools_for(episode)._released()
    assert asked[-1]["asset_id"] == raw["asset_id"]
    del raw["asset_id"]
    episode.released_path.write_text(json.dumps(raw), encoding="utf-8")
    view = tools_for(episode)._released().view
    assert "asset_id" not in asked[-1] and view.asset_id == real().asset_id


def test_a_status_without_a_budget_does_not_stop_the_read_batch(episode):
    """A row whose released status carries no budget, which the row presets allow, leaves the batch bounded by the
    horizon and the batch cap only."""
    episode.read("tier1")
    released = json.loads(episode.released_path.read_text(encoding="utf-8"))
    released["status"]["budget_left"] = None
    episode.released_path.write_text(json.dumps(released), encoding="utf-8")
    ranking = tools_for(episode).rank_root_causes(3)
    ranking_is_coherent(ranking)
    assert bool(ranking.read_batch) == ranking.read_first


def test_the_card_reserve_adds_the_cheapest_other_plan_and_the_defaults_are_todays_constants(episode):
    """The server fields of the decision threshold card keep today's constants by default (no margin, the batch cap, the
    top plan alone in the reserve). Under reserve_plans 2 the reserve adds the price and the cycles of the cheapest repair
    plan of another combination of the credible set, the plan backtracking turns to, and a healthy combination adds no plan."""
    tools = tools_for(episode)
    assert (tools.e10_margin, tools.batch_cap, tools.reserve_plans) == (0.0, None, 1)
    view = profile_view(episode.rules)
    ranked_combinations = [(100, 80, 0, 90), (100, 100, 0, 90), (3, 100, 0, 130), (100, 100, 0, 130)]
    post = Posterior.uniform(episode_models(START).effect, support=ranked_combinations)
    ranked = [post.candidates.index(h) for h in ranked_combinations]
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import verifying_read_price
    # each plan's price holds the read that verifies it, here for an agent told no indicating channel (verifying_read_price)
    top = (plan_terms(view, ["valve", "accumulator"])[0] + verifying_read_price(view, ["valve", "accumulator"], []),
           plan_terms(view, ["valve", "accumulator"])[1], plan_completion(view, ["valve", "accumulator"]).calls)  # price, cycles, calls
    assert tools._reserve_terms(view, post, ranked, []) == (top[0], top[2], top[1])
    cheapest = min((plan_terms(view, [c])[0] + verifying_read_price(view, [c], []), plan_terms(view, [c])[1],
                    plan_completion(view, [c], commit=False).calls - 1) for c in ("accumulator", "cooler"))
    assert tools_for(episode, reserve_plans=2)._reserve_terms(view, post, ranked, []) == (top[0] + cheapest[0], top[2] + cheapest[2], top[1] + cheapest[1])


@pytest.mark.parametrize("told", [False, True])
def test_verify_outcome_judges_a_record_row_with_what_the_agent_was_told(tmp_path, world, told):
    """Told equals done: on a row that reads a record the failure modes carry no channels, but an answer of
    lookup_fix names the indicating channels of its component, and the agent then verifies the repair on a group that holds one
    of them (Board.verifies). verify_outcome judges with the same knowledge (ProfileView.told_indicating_channels): after the
    manual's answer a reading of such a group verifies the plan, and without it only a reading of every channel would."""
    from CausalDiagnosis.utils.public_rules.repair_completion import verifies
    cycles, blocks = world
    episode = FixtureEpisode(tmp_path, cycles, blocks, START, seed=0, rules=default_rules(cycles, blocks, tools=[{"server": "iot", "name": "history"}]))
    tools, view = tools_for(episode), profile_view(episode.rules)
    assert view.reads_record and not any(view.told_indicating_channels([{"event": "failure_modes", "component": ""}]).values())
    episode.read("tier3")
    ranking = tools.rank_root_causes(3)
    effect = tools.estimate_intervention_effect("replace:accumulator")
    episode.submit_diagnosis(ranking.candidates[0].combination, evidence_ids=ranking.evidence_ids[:3])
    episode.create_work_order("accumulator", prediction_id=effect.prediction_id)
    if told:
        episode._event("fix_lookup", episode.time, "read", 0, component="accumulator")
    episode.advance(effect.duration)
    manual = view.told_indicating_channels([{"event": "fix_lookup", "component": "accumulator"}])
    group = next(g for g, channels in view.sensor_groups.items() if verifies(channels, ["accumulator"], manual, view.sensor_groups)
                 and not verifies(channels, ["accumulator"], {}, view.sensor_groups))
    episode.read(group)
    status = tools.verify_outcome(effect.prediction_id).status
    assert status == "verified" if told else status.startswith("not yet observable")


def test_the_fits_of_an_episode_take_the_covariance_estimator_of_the_start_s_form():
    """The fits of an episode read the covariance estimator from the form of their start in models.json, whichever registered
    estimator that form holds."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.effect_model import COVARIANCE_ESTIMATORS, combo_key, fit_episode

    models = load_models()
    key = combo_key(START)
    assert models.covariance_without(START) == models.raw["selections"][key]["covariance"] in COVARIANCE_ESTIMATORS
    for estimator in COVARIANCE_ESTIMATORS:
        raw = {**models.raw, "selections": {**models.raw["selections"], key: {**models.raw["selections"][key], "covariance": estimator}}}
        served = adapter.Models(raw).without(START)
        direct = fit_episode(models.recorded, models.pairs_without(START), START, 0, 0, covariance=estimator)
        assert np.array_equal(served.effect.sigma_within, direct.effect.sigma_within) and np.array_equal(served.sigma, direct.sigma)
        assert np.array_equal(served.effect.mu_table, direct.effect.mu_table)
        top = served.effect.sigma_within[0]
        assert (np.count_nonzero(top - np.diag(np.diag(top))) == 0) == (estimator == "diagonal")


def test_the_notes_say_how_the_fits_leave_out_two_blocks_and_name_no_estimator():
    """The two notes of every result that describe the fits: each combination is scored by a fit that leaves out two blocks, the
    form was chosen before any episode, and no note names a covariance estimator, so the notes stay one text for every start."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.effect_model import COVARIANCE_ESTIMATORS

    assert "leaves out two blocks" in adapter.A_RULE and "one second block" in adapter.A_RULE
    assert "chosen before any episode by a registered rule that reads no recording of that block" in adapter.A_RULE
    assert "the covariance estimator of the form of the model" in adapter.A_E1
    for note in (adapter.A_RULE, adapter.A_E1):
        assert not any(name in note for name in COVARIANCE_ESTIMATORS) and "Ledoit" not in note


def test_two_stores_at_once_keep_both_predictions_under_two_identifiers(episode, monkeypatch):
    """Two server processes of one episode can store at once. A second store runs while the first stands between its read of
    predictions.json and its write, which waits until the second has had time to finish: under the lock the second waits for
    the first, so the two predictions take two identifiers and the file holds both records."""
    import threading

    tools = tools_for(episode)
    released = tools._released()
    flt, post = tools._belief(released)
    predicted = tools._predicted_action(released.view, flt, post, ["valve"])
    real_write, finished, ids, caller = adapter.causal_tools.write_json_atomically, threading.Event(), {}, threading.current_thread()

    def other():
        ids["other"] = tools._store_prediction(released, "replace:valve", ["valve"], predicted)
        finished.set()

    second = threading.Thread(target=other)

    def held_write(path, obj, indent=None):
        if threading.current_thread() is caller and not second.is_alive() and not finished.is_set():
            second.start()
            finished.wait(timeout=2)  # the other store runs now, unless the lock keeps it out until this write
        real_write(path, obj, indent=indent)

    monkeypatch.setattr(adapter.causal_tools, "write_json_atomically", held_write)
    ids["first"] = tools._store_prediction(released, "replace:valve", ["valve"], predicted)
    second.join()
    stored = json.loads((episode.folder / "predictions.json").read_text(encoding="utf-8"))
    assert ids["first"] != ids["other"] and sorted(p["id"] for p in stored) == sorted(ids.values())


def test_a_branch_prediction_that_another_process_stored_meanwhile_is_reused(episode):
    """Two rankings of one released evidence that both found no stored branch for an action before either stored it, as two
    server processes of one episode can, keep one record of that action, and both get its identifier."""
    tools = tools_for(episode)
    released = tools._released()
    flt, post = tools._belief(released)
    predicted = tools._predicted_action(released.view, flt, post, ["valve"])
    assert tools._stored_branch("replace:valve", released) is None
    first = tools._store_prediction(released, "replace:valve", ["valve"], predicted, branch=1)
    again = tools._store_prediction(released, "replace:valve", ["valve"], predicted, branch=2)
    stored = json.loads((episode.folder / "predictions.json").read_text(encoding="utf-8"))
    assert first == again == "pred/b1" and [p["id"] for p in stored] == ["pred/b1"] and stored[0]["branch"] == 1


def test_a_monitor_prediction_is_checked_against_a_reading_recorded_at_or_after_it(episode):
    """On the AssetOpsBench rows a later read can release an older recorded cycle, so the reading that checks a monitor
    prediction is the first one released after it and recorded at or after its cycle. With only an older cycle released
    after the prediction the check is not yet observable, and once a later cycle is released it compares that one."""
    episode.read("tier1")
    episode.read("tier1")
    tools = tools_for(episode)
    watch = tools.estimate_intervention_effect("monitor")
    made_at = json.loads((episode.folder / "predictions.json").read_text(encoding="utf-8"))[-1]["time"]
    episode.read("tier2")
    episode.observations[-1]["t"] = made_at - 1  # the summaries of a cycle recorded before the prediction, released after it
    episode.write()
    waiting = tools.verify_outcome(watch.prediction_id)
    assert waiting.status == "not yet observable: no reading since the action" and waiting.observation is None
    episode.read("tier1")
    later = tools.verify_outcome(watch.prediction_id)
    assert later.status == "verified" and later.observation == episode.observations[-1]["n"] and later.observed_t >= made_at


def test_the_error_of_a_missing_models_file_names_the_fit_command_and_the_data_variable_of_the_asset(tmp_path):
    """A server started before its models file exists answers with the fit command, which fits on the release file of the
    asset or on the file that the environment variable of its profile names, in words that hold for any asset."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import data_variable

    with pytest.raises(FileNotFoundError) as missing:
        load_models(tmp_path / "models.json")
    text = str(missing.value)
    assert text.startswith("models.json is generated from the recordings of an asset and is not part of the repository")
    assert "`python -m CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models`" in text and f"the file that {data_variable()} names when it is set" in text


def test_every_ranking_holds_the_rule_at_the_top_and_e10_in_the_long_forms(world, tmp_path):
    """Both forms of a ranking after one read, on the rules of the fixture, which hold prices, and on the free base row: the
    top level of each states the one read of the rule and is the same in both, the long form adds e10_rule with E10 and
    E22, and the compact form holds no e10_rule and shortens the rest as assert_compact_shortens states."""
    cycles, blocks = world
    for name, rules in (("priced", default_rules(cycles, blocks)), ("free", free_rules(cycles, blocks))):
        episode = FixtureEpisode(tmp_path / name, cycles, blocks, START, seed=0, rules=rules)
        episode.read("tier1")
        with hidden_files_sealed():
            long, short = tools_for(episode).rank_root_causes(3, "summary"), tools_for(episode).rank_root_causes(3, "compact")
        ranking_is_coherent(long)
        ranking_is_coherent(short)
        assert long.e10_rule is not None and short.e10_rule is None and "e10_rule" not in short.model_dump()
        assert_compact_shortens(short, long)


def test_the_direction_is_undetermined_when_the_fits_are_too_few_for_the_declared_level(episode, monkeypatch):
    """The range of R resampled fits holds a further fit with probability (R - 1) / (R + 1), so the direction of E15 needs at
    least fits_for_level(alpha) fits. With the fits of the fixture the direction follows the dead band, half the range. With
    two fits the direction is undetermined and no dead band exists, while the two ends of the range and the number of fits
    are reported."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import effect_model as em
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.conformal_intervals import fits_for_level

    episode.read("tier1")
    full = tools_for(episode).estimate_intervention_effect("replace:valve")
    assert full.resampled_fits >= fits_for_level(full.alpha) and full.direction in ("lower", "higher", "no change")
    assert abs(full.dead_band - (full.recovery_interval_high - full.recovery_interval_low) / 2) < 1e-12
    monkeypatch.setattr(em.EffectModel, "resample_count", property(lambda self: 2))
    few = tools_for(episode).estimate_intervention_effect("replace:valve")
    assert few.resampled_fits == 2 < fits_for_level(few.alpha)
    assert few.direction == "undetermined" and few.dead_band is None
    assert few.recovery_interval_low is not None and few.recovery_interval_low <= few.recovery_interval_high


def test_the_results_say_that_the_calibration_of_the_fixture_reaches_the_level(episode):
    """The estimate, the verification and every plan of a ranking state whether the calibration blocks carry alpha, which the
    blocks of the fixture do."""
    episode.read("tier1")
    tools = tools_for(episode)
    ranking = tools.rank_root_causes(3)
    estimate = tools.estimate_intervention_effect("replace:valve")
    verification = tools.verify_outcome(estimate.prediction_id)
    assert estimate.level_reachable is True and verification.level_reachable is True
    assert ranking.branches and all(b.predicted_outcome.level_reachable is True for b in ranking.branches)


def test_the_default_reply_is_compact_and_the_other_details_keep_their_keys(episode):
    """The default call is branch_detail compact. After a read of two cycles compact gives each candidate one identifier per
    summary beside the closed orders, where summary names both cycles, and leaves out what assert_compact_shortens names,
    while every other field is that of summary. Summary and full hold every key of the reply."""
    episode.read("tier3", cycles=2)
    tools = tools_for(episode)
    with hidden_files_sealed():
        default = tools.rank_root_causes(3)
        compact, summary, full = (tools.rank_root_causes(3, detail) for detail in ("compact", "summary", "full"))
    assert default == compact
    ranking_is_coherent(compact)
    assert_compact_shortens(compact, summary)
    for cand in compact.candidates:
        fields = [i.split("/", 2)[2] for i in cand.evidence_ids if i.startswith("obs/")]
        assert len(fields) == len(set(fields))
    assert any(len(c.evidence_ids) < len(s.evidence_ids) for c, s in zip(compact.candidates, summary.candidates))
    assert set(summary.model_dump()) == set(full.model_dump()) == set(RootCauses.model_fields)


def test_a_prediction_whose_order_is_created_later_is_verified_for_the_cycle_of_its_creation(world, tmp_path):
    """Thinking time or other calls between an estimate and its work order move the closures of the visit, and the rule cards
    act on the cycles in between. The verification makes the prediction again for the cycle the order was created, from the
    evidence the estimate held, which is what an estimate made at that cycle on the same evidence predicts, and says so."""
    cycles, blocks = world
    config = EpisodeConfig().to_json()
    rules = default_rules(cycles, blocks, processes={"hazard": {**config["hazard"], "mode": "stochastic"}})
    episode = FixtureEpisode(tmp_path, cycles, blocks, START, seed=0, rules=rules)
    episode.read("tier3")
    tools = tools_for(episode)
    early = tools.estimate_intervention_effect("replace:accumulator")
    episode.submit_diagnosis(dict(zip(COMPONENTS, START)))
    episode.advance(5)  # the thinking time of the next call, on a row that charges it
    late = tools.estimate_intervention_effect("replace:accumulator")
    created = episode.time
    episode.create_work_order("accumulator", prediction_id=early.prediction_id)
    episode.advance(late.duration)
    episode.read("tier3")
    verified = tools.verify_outcome(early.prediction_id)
    assert verified.status == "verified" and created == early.time_window.start + 5
    assert verified.recovery_predicted == pytest.approx(late.recovery, rel=1e-9) and late.recovery != early.recovery
    expected = {s.field: s.mean for s in late.predicted}
    assert verified.channels and all(c.predicted == pytest.approx(expected[c.field], rel=1e-9) for c in verified.channels)
    assert f"made again for cycle {created}, when its work orders were created" in verified.message


def test_the_models_file_carries_the_code_that_fitted_it_and_a_read_only_process_fits_nothing(tmp_path, monkeypatch):
    """The models file records the digest of the server's code: models_problems names a missing file and a file of other code,
    and with CD_TABLES_READ_ONLY=1 ensure_models stops by name before it fits, the file left as it was."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import fit_models as fit
    from CausalDiagnosis.utils.atomic_writes_and_locks import READ_ONLY_TABLES, GeneratedTableMissing

    given = tmp_path / "recordings.jsonl"
    given.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(fit, "fit", lambda facts_path, **options: {"format": fit.MODELS_FORMAT, "facts_sha256": fit.facts_sha256(facts_path)})
    path = tmp_path / "models.json"
    assert fit.models_problems(path, facts_path=given) == [f"no models file at {path}"]
    fit.ensure_models(path, facts_path=given)
    assert fit.models_problems(path, facts_path=given) == []
    path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), "code": "other code"}), encoding="utf-8")
    assert fit.models_problems(path, facts_path=given) == ["models.json was fitted by other code"]
    monkeypatch.setenv(READ_ONLY_TABLES, "1")
    before = path.read_bytes()
    with pytest.raises(GeneratedTableMissing, match="precompute_tables.sbatch"):
        fit.ensure_models(path, facts_path=given)
    assert path.read_bytes() == before
