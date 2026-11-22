"""The troubleshooting plans of rank_root_causes on the fixture episode: one branch per top combination with its mass, a confirm
test that separates it from the others, a repair plan over its faulty set as one technician visit (one visit charge, the
repairs in priority order after the longest lead time, the dependencies of every step, the cycles to restored of
schedule_visit) whose predictions verify_outcome knows and the intervention tool would store the same way, costs that add
up from the public prices, the price of a wrong diagnosis in the loss of acting on each branch, the decision fields
untouched, and the same branches on a replayed call; and on the real
environment, a plan of two work orders that both cite the plan's prediction and are verified by one reading after the
last closure. No test states a value computed from the rig data, except that the plan of the true start is verified as
consistent on its own block."""
from CausalDiagnosis.utils.paths import CONFIGS, preset_files, preset_path
import itertools
import json
from pathlib import Path

import pytest

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_4_repair_rules import schedule_visit
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY, build_blocks, default_facts_path, load_cycles
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import tools as adapter
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import CausalTools
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import profile_view
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.troubleshooting_plans import SEPARATION, plan_terms, repair_plan, restored, visit_schedule
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.decision_rule import costs
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import Episode, Filter
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tool_result_schemas import ErrorResult
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import RANKING_MODELS
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.fixture_episode import FixtureEpisode, default_rules, episode_models, with_misdiagnosis
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.test_tools import N_CANDIDATES, START, fields_of, hidden_files_sealed
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as rig
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tool_contract import load_contract

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)


@pytest.fixture(scope="module")
def world():
    cycles = load_cycles(default_facts_path())
    return cycles, build_blocks(cycles)


@pytest.fixture
def episode(tmp_path, world):
    cycles, blocks = world
    return FixtureEpisode(tmp_path, cycles, blocks, START, seed=0)


def faulty_now(combination):
    return [c for c in COMPONENTS if combination[c] != HEALTHY[c]]


def returned_sets(ranking):
    """{faulty set: [ranks]} of the candidates the ranking returns (the first top_k by mass), in the order of each set's first
    candidate: what the branches of a ranking must hold, one per set."""
    sets = {}
    for candidate in ranking.candidates:
        sets.setdefault(tuple(faulty_now(candidate.current_combination)), []).append(candidate.rank)
    return sets


def set_mass(ranking, faulty_set):
    return next(row.mass for row in ranking.set_masses if row.faulty_set == sorted(faulty_set))


def stored_records(episode):
    return json.loads((episode.folder / "predictions.json").read_text(encoding="utf-8"))


def test_branches_follow_the_candidates_and_add_up_from_the_public_prices(episode):
    tools = CausalTools(episode_models(episode.start), episode.released_path)
    episode.read("tier1")
    with hidden_files_sealed():
        ranking = tools.rank_root_causes(branch_detail="full")
    assert len(ranking.candidates) == adapter.DEFAULT_TOP_K and all(c.probability > 0 for c in ranking.candidates)
    sets = returned_sets(ranking)
    assert [(tuple(b.faulty_set), b.ranks) for b in ranking.branches] == list(sets.items())
    rules = episode.rules
    rate = ranking.costs["expected_downtime_rate"]
    for branch in ranking.branches:
        candidate = ranking.candidates[branch.rank - 1]
        assert (branch.rank, branch.combination) == (candidate.rank, candidate.combination) and branch.rank == branch.ranks[0]
        assert abs(branch.probability - set_mass(ranking, branch.faulty_set)) < 1e-12  # the set's mass over every combination
        assert branch.current_combination == candidate.current_combination
        faulty = faulty_now(branch.current_combination)
        assert branch.faulty_set == faulty and [f.component for f in branch.faults] == faulty and all(f.code for f in branch.faults)
        test = branch.confirm_test
        assert test.sensor_group in rules["sensor_groups"] and test.cycles == 1
        assert test.price == rules["prices"]["read"][test.sensor_group] and test.duration == rules["durations"]["read_per_cycle"]
        assert test.cost == ranking.costs["test_cost"][test.sensor_group] and test.cost >= test.price
        assert 0 <= test.mass_if_false <= test.mass_if_true <= 1 and abs(test.gap - (test.mass_if_true - test.mass_if_false)) < 1e-12
        assert test.standard_error >= 0 and set(test.gap_by_group) == set(rules["sensor_groups"]) and test.gap == max(test.gap_by_group.values())
        if branch.two_step is not None:
            two = branch.two_step
            assert len(two.sensor_groups) == two.cycles == 2 and set(two.sensor_groups) <= set(rules["sensor_groups"])
            assert two.price == sum(rules["prices"]["read"][g] for g in two.sensor_groups) and two.duration == 2 * rules["durations"]["read_per_cycle"]
            assert two.cost == sum(ranking.costs["test_cost"][g] for g in two.sensor_groups)
            assert 0 <= two.mass_if_false <= two.mass_if_true <= 1
            assert two.gap - test.gap > SEPARATION * (two.standard_error ** 2 + test.standard_error ** 2) ** 0.5
        plan = branch.repair_plan
        view = profile_view(rules)
        assert [s.component for s in plan.steps] == sorted(faulty, key=view.priority)  # the technician's order
        for step in plan.steps:
            assert step.part_price == rules["prices"]["parts"][step.component] and step.repair_charge == rules["prices"]["repair_charge"]
            assert step.lead_time == rules["durations"]["lead_time"][step.component] and step.repair_downtime == rules["durations"]["repair_downtime"][step.component]
            assert step.downtime_cost == rules["downtime_rate"]["stopped"] * step.repair_downtime
            assert step.depends_on == list(view.dependencies(step.component)) and set(step.dependencies_in_plan) == set(step.depends_on) & set(faulty)
            assert step.prediction_id.startswith("pred/b")
        if len(plan.steps) == 1:
            assert plan.plan_prediction_id == plan.steps[0].prediction_id
        else:
            assert plan.plan_prediction_id.startswith("pred/b") and plan.plan_prediction_id not in {s.prediction_id for s in plan.steps}
        visit_charge = rules["prices"].get("visit", 0)
        assert plan.visits == 1 and plan.visit_charge == visit_charge and plan.lead_time == max(s.lead_time for s in plan.steps)
        assert plan.price == sum(s.part_price + s.repair_charge for s in plan.steps) + visit_charge
        assert abs(plan.waiting_cost - rate * plan.lead_time) < 1e-9 and abs(plan.downtime_cost - plan.waiting_cost - sum(s.downtime_cost for s in plan.steps)) < 1e-9
        assert plan.cycles_to_restored == plan.steps[-1].close_offset + 1 == plan.lead_time + sum(s.repair_downtime for s in plan.steps) + 1
        outcome = branch.predicted_outcome
        assert len(outcome.predicted) == len(fields_of()) and all(p.low <= p.mean <= p.high and p.quantile >= 0 for p in outcome.predicted)
        # the plan restores every combination whose faulty set it repairs, returned or not: the masses of those sets
        covered = sum(row.mass for row in ranking.set_masses if set(row.faulty_set) <= set(faulty))
        assert abs(outcome.recovery_if_acted - covered) < 1e-9 and outcome.restored_if_true and outcome.alpha == adapter.DEFAULT_ALPHA
        plan_record = next(p for p in stored_records(episode) if p["id"] == plan.plan_prediction_id)
        assert plan_record["recovery"] == outcome.recovery_if_acted and plan_record["predicted"][fields_of()[0]][0] == outcome.predicted[0].mean
        assert plan_record.get("components", [plan_record["component"]]) == [s.component for s in plan.steps]  # in the order of the visit
        assert branch.expected_loss_if_acted >= 0 and branch.expected_loss_if_tested == ranking.expected_loss_test[test.sensor_group]
        # the outlay holds every replacement of the plan, so act_now lies at or below the plan's money plus E8
        assert branch.cost_to_restored.act_now <= plan.price + plan.downtime_cost + branch.expected_loss_if_acted + 1e-9
        assert branch.cost_to_restored.act_now >= plan.price and branch.cost_to_restored.test_first >= test.cost
    first = {}
    for branch in ranking.branches:  # testing first reads the confirm test and then acts on what is first, whatever the branch
        assert first.setdefault(branch.confirm_test.sensor_group, branch.cost_to_restored.test_first) == branch.cost_to_restored.test_first
    if ranking.branches[0].ranks == [1]:  # a set of the first candidate alone acts as the ranking's act
        assert ranking.branches[0].expected_loss_if_acted == ranking.expected_loss_act
    text = " ".join(ranking.assumptions)
    assert "branches:" in text and "not the cost the environment charges" in text and f"{ranking.monte_carlo.samples} draws" in text


def test_the_cost_of_acting_now_holds_the_outlay_of_a_plan_once(tmp_path, world):
    """Acting on a branch repairs its faulty set and diagnoses its combination, and the outlay of its plan holds every
    replacement of that set. So act_now is the price and the downtime cost of the plan plus the belief times the faults the
    plan leaves and the price of a wrong diagnosis, computed here from the rules, and it lies below the plan's money plus E8
    by the expected unnecessary replacements; a branch keyed by a faulty set prices the combination a diagnosis submits, the
    set's most probable candidate, as testing first prices the combination on top after the reading. With prices.misdiagnosis
    the expected loss if acted and act_now of every branch grow by that price times one minus the mass of that candidate, while
    the mass, the faulty set, the plan's price and the confirm group stay as they are without the price."""
    cycles, blocks = world
    price = profile_view().visit_charge()
    rankings = {}
    for key in (None, price):
        rules = with_misdiagnosis(default_rules(cycles, blocks), key)
        episode = FixtureEpisode(tmp_path / f"price_{key}", cycles, blocks, START, seed=0, rules=rules)
        episode.read("tier1")
        with hidden_files_sealed():
            rankings[key] = CausalTools(episode_models(episode.start), episode.released_path).rank_root_causes(N_CANDIDATES)
    assert price > 0 and len(rankings[price].branches) == len(rankings[None].branches) > 1
    for plain, priced in zip(rankings[None].branches, rankings[price].branches):
        assert (priced.combination, priced.probability, priced.faulty_set) == (plain.combination, plain.probability, plain.faulty_set)
        assert priced.repair_plan.price == plain.repair_plan.price and priced.confirm_test.sensor_group == plain.confirm_test.sensor_group
        added = price * (1 - rankings[price].candidates[priced.ranks[0] - 1].probability)
        assert abs(priced.expected_loss_if_acted - plain.expected_loss_if_acted - added) < 1e-9
        assert abs(priced.cost_to_restored.act_now - plain.cost_to_restored.act_now - added) < 1e-9
    for key, ranking in rankings.items():
        rules = ranking.costs
        rates = {c: {int(level): rate for level, rate in default_rules(cycles, blocks)["downtime_rate"][c].items()} for c in COMPONENTS}
        belief = [(c.current_combination, c.probability) for c in ranking.candidates]
        for branch in ranking.branches:
            missed = wrong = unnecessary = 0.0
            for r, w in ((branch.ranks[0], 1.0),):  # the set's most probable candidate, the one a diagnosis submits
                acted = ranking.candidates[r - 1].current_combination
                missed += w * sum(p * sum(rates[c][h[c]] * rules["remaining_horizon"] for c in COMPONENTS if acted[c] == HEALTHY[c] and h[c] != HEALTHY[c])
                                  for h, p in belief)
                wrong += w * (key or 0) * sum(p for h, p in belief if h != acted)
                unnecessary += w * sum(p * sum(rules["unnecessary_replacement"][c] for c in COMPONENTS if acted[c] != HEALTHY[c] and h[c] == HEALTHY[c])
                                       for h, p in belief)
            plan = branch.repair_plan
            assert abs(branch.cost_to_restored.act_now - (plan.price + plan.downtime_cost + missed + wrong)) < 1e-6 * max(1.0, branch.cost_to_restored.act_now)
            assert abs(plan.price + plan.downtime_cost + branch.expected_loss_if_acted - branch.cost_to_restored.act_now - unnecessary) < 1e-6 * max(1.0, unnecessary)


def test_plan_predictions_are_stored_as_the_intervention_tool_stores_them_and_verified_after_the_repair(episode):
    tools = CausalTools(episode_models(episode.start), episode.released_path)
    episode.read("tier3")
    with hidden_files_sealed():
        ranking = tools.rank_root_causes(3)
        pending = [tools.verify_outcome(s.prediction_id) for b in ranking.branches for s in b.repair_plan.steps]
        effect = tools.estimate_intervention_effect("replace:" + ranking.branches[0].repair_plan.steps[0].component)
    assert all(not isinstance(v, ErrorResult) and v.status.startswith("not yet observable") for v in pending)
    assert effect.prediction_id == "pred/1"
    stored = stored_records(episode)
    own, branch_records = [p for p in stored if "branch" not in p], [p for p in stored if "branch" in p]
    cited = {s.prediction_id for b in ranking.branches for s in b.repair_plan.steps} | {b.repair_plan.plan_prediction_id for b in ranking.branches}
    assert [p["id"] for p in own] == ["pred/1"] and {p["id"] for p in branch_records} == cited
    assert all(p["id"].startswith("pred/b") and p["branch"] >= 1 for p in branch_records)
    assert all(p["action"] == "replace:" + ",".join(p.get("components") or [p["component"]]) for p in branch_records)
    assert all((p["component"] is None) == ("components" in p) for p in branch_records)
    # one record per action at one released state: branches that share a component or a plan cite the same record
    assert len({p["action"] for p in branch_records}) == len(branch_records)
    first = next(p for p in branch_records if p["id"] == ranking.branches[0].repair_plan.steps[0].prediction_id)
    for key in ("action", "component", "events", "observations", "time", "addresses", "recovery", "recovery_interval", "alpha", "ranking_model", "after", "predicted"):
        assert first[key] == own[0][key], key
    assert first["predicted"][fields_of()[0]] == [effect.predicted[0].mean, effect.predicted[0].low, effect.predicted[0].high, effect.predicted[0].quantile]
    step = ranking.branches[0].repair_plan.steps[0]
    episode.submit_diagnosis(ranking.branches[0].combination, evidence_ids=ranking.evidence_ids[:2])
    episode.create_work_order(step.component, prediction_id=step.prediction_id)
    episode.advance(step.lead_time + step.repair_downtime)
    episode.read("tier3")
    with hidden_files_sealed():
        verified = tools.verify_outcome(step.prediction_id)
        after = tools.rank_root_causes(2)
    assert verified.status == "verified" and verified.n_channels == len(fields_of()) and verified.consistent is not None
    assert verified.recovery_predicted == first["recovery"] and 0 <= verified.recovery_posterior <= 1
    assert all(b.current_combination[step.component] == HEALTHY[step.component] and step.component not in b.faulty_set for b in after.branches)
    # new released evidence, new records; the earlier ones stay
    later = stored_records(episode)
    assert [p["id"] for p in later[:len(stored)]] == [p["id"] for p in stored] and len(later) > len(stored)


def test_the_summary_detail_leaves_out_the_two_largest_fields_and_changes_nothing_else(episode):
    tools = CausalTools(episode_models(episode.start), episode.released_path)
    episode.read("tier1")
    with hidden_files_sealed():
        summary, full = tools.rank_root_causes(3, "summary"), tools.rank_root_causes(3, "full")
        default, blank = tools.rank_root_causes(3), tools.rank_root_causes(3, "")
        assert isinstance(tools.rank_root_causes(3, "all"), ErrorResult) and isinstance(tools.rank_root_causes(3, 1), ErrorResult)
    assert default.model_dump() == blank.model_dump() and default.branches == summary.branches  # the plans of compact are those of summary
    assert len(summary.branches) == len(full.branches) == len(returned_sets(summary))
    for compact, detailed in zip(summary.branches, full.branches):
        assert compact.confirm_test.gap_by_group is None and compact.predicted_outcome.predicted is None
        assert detailed.confirm_test.gap_by_group is not None and len(detailed.predicted_outcome.predicted) == len(fields_of())
        rest = detailed.model_dump()
        rest["confirm_test"]["gap_by_group"], rest["predicted_outcome"]["predicted"] = None, None
        assert compact.model_dump() == rest
    without_branches = {k: v for k, v in summary.model_dump().items() if k != "branches"}
    assert without_branches == {k: v for k, v in full.model_dump().items() if k != "branches"}
    assert len(json.dumps(summary.model_dump())) < len(json.dumps(full.model_dump()))
    assert "branch_detail summary omits" in " ".join(summary.assumptions)


def test_a_replayed_ranking_returns_the_same_branches_and_stores_nothing_twice(episode):
    tools = CausalTools(episode_models(episode.start), episode.released_path)
    episode.read("tier1")
    episode.read("tier2")
    with hidden_files_sealed():
        first = tools.rank_root_causes()
        second = tools.rank_root_causes()
        fresh = CausalTools(episode_models(episode.start), episode.released_path).rank_root_causes()
    assert first.model_dump() == second.model_dump() == fresh.model_dump() and len(first.branches) == len(returned_sets(first))
    records = [p for p in stored_records(episode) if "branch" in p]
    assert len({(p["action"], p["events"], p["observations"]) for p in records}) == len(records)
    assert {p["id"] for p in records} == {s.prediction_id for b in first.branches for s in b.repair_plan.steps} | {b.repair_plan.plan_prediction_id for b in first.branches}


def test_the_branches_change_no_other_field(episode, monkeypatch):
    tools = CausalTools(episode_models(episode.start), episode.released_path)
    episode.read("tier2")
    with hidden_files_sealed():
        ranking = tools.rank_root_causes(branch_detail="summary")
    assert ranking.branches and ranking.branches[0].rank == 1
    monkeypatch.setattr(adapter.causal_tools, "branch_previews", lambda *args, **kwargs: [])
    with hidden_files_sealed():
        bare = tools.rank_root_causes(branch_detail="summary")
    with_branches, without = ranking.model_dump(), bare.model_dump()
    assert without["branches"] == [] and with_branches["branches"]
    assert with_branches.pop("assumptions") == without.pop("assumptions")
    with_branches.pop("branches"), without.pop("branches")
    assert with_branches == without


@pytest.mark.parametrize("ranking_model", RANKING_MODELS)
def test_every_ranking_model_returns_coherent_branches(episode, ranking_model):
    tools = CausalTools(episode_models(episode.start), episode.released_path, ranking_model=ranking_model)
    episode.read("tier3")
    with hidden_files_sealed():
        ranking = tools.rank_root_causes()
    sets = returned_sets(ranking)
    assert [b.ranks for b in ranking.branches] == list(sets.values())
    assert all(abs(b.probability - set_mass(ranking, b.faulty_set)) < 1e-12 for b in ranking.branches)
    for branch in ranking.branches:
        assert 0 <= branch.confirm_test.mass_if_false <= branch.confirm_test.mass_if_true <= 1
        assert sorted(s.component for s in branch.repair_plan.steps) == sorted(branch.faulty_set)
        assert 0 <= branch.predicted_outcome.recovery_if_acted <= 1
    records = [p for p in stored_records(episode) if "branch" in p]
    assert all(p["ranking_model"] == ranking_model for p in records) and records
    if ranking_model == "discriminant_urn":
        assert all(p["recovery_interval"] is None for p in records)


def test_a_two_order_plan_through_the_environment_is_verified_by_one_reading_after_the_last_closure(tmp_path, world):
    """The rule the agent follows: every work order of a plan cites repair_plan.plan_prediction_id, the agent reads once
    after the last order closed, and both verify_outcome and the verifier score the plan's prediction against that state.
    The orders of a plan are created in consecutive calls of duration zero, so they form one visit whose technician repairs
    them back to back, and the agent waits with wait_for_notification until the last close the envelope shows before it
    reads."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig

    cycles, blocks = world
    state_path = tmp_path / "episode.json"
    rig.start(state_path, 0, EpisodeConfig(start={"combination": list(START), "entry_offset": 0}), default_facts_path())
    rig.hand_over(state_path)
    tools = CausalTools(episode_models(rig.replay(rig.load_state(state_path)).initial), rig.released_path(state_path))

    def act(tool, arguments):
        return json.loads(rig.call(state_path, tool, arguments, {"type": tool, **arguments}))

    act("read_sensors", {"sensor_group": "tier3", "cycles": 1})
    with hidden_files_sealed():
        ranking = tools.rank_root_causes()
    truth = dict(zip(COMPONENTS, START))
    branch = next(b for b in ranking.branches if b.faulty_set == faulty_now(truth))  # the test knows the truth, keyed by its faulty set
    plan = branch.repair_plan
    assert len(plan.steps) == 2 and plan.plan_prediction_id not in {s.prediction_id for s in plan.steps}
    act("submit_diagnosis", {"diagnosis": truth, "evidence_ids": ranking.evidence_ids[:2]})
    created = [act("create_work_order", {"component": s.component, "prediction_id": plan.plan_prediction_id, "evidence_ids": []}) for s in plan.steps]
    assert all("refused" not in c for c in created)
    closes = {o["id"]: o["expected_close_t"] for o in created[-1]["open_work_orders"]}
    assert len({c["result"]["work_order"]["visit"] for c in created}) == 1 and len(set(closes.values())) == len(plan.steps), "one visit, back to back"
    envelope, last_close = created[-1], max(closes.values())
    for _wait in range(2 * len(plan.steps) + 2):  # each wait ends at the next public event, such as a close, or at its timeout
        if envelope["time"] >= last_close:
            break
        envelope = act("wait_for_notification", {"timeout": last_close - envelope["time"]})
    assert envelope["time"] == last_close
    act("read_sensors", {"sensor_group": "tier3", "cycles": 1})
    with hidden_files_sealed():
        verified = tools.verify_outcome(plan.plan_prediction_id)
        per_step = [tools.verify_outcome(s.prediction_id) for s in plan.steps]
    assert verified.status == "verified" and verified.n_channels == len(fields_of()) and verified.consistent is True
    assert verified.recovery_predicted == branch.predicted_outcome.recovery_if_acted
    with hidden_files_sealed():
        after = tools.rank_root_causes(N_CANDIDATES)
    assert verified.recovery_posterior == next(c.probability for c in after.candidates if c.current_combination == HEALTHY)
    assert verified.time_window.start == max(closes.values())
    # a step's own prediction describes one replacement, which no reading after both closures can show
    assert all(v.status.startswith("not comparable") for v in per_step)
    act("finish", {"summary": "both work orders closed and the reading matched the plan's prediction"})
    env = rig.replay(rig.load_state(state_path))
    released = json.loads(rig.released_path(state_path).read_text(encoding="utf-8"))
    verdict = verifier.verify(env.log, released, env.oracle(), None, predictions=json.loads((tmp_path / "predictions.json").read_text(encoding="utf-8")))
    assert verdict["verified"] and verdict["pass"] and verdict["recovered"] and verdict["checks"]["provenance"]["ok"]
    rows = verdict["prediction_verdict"]
    assert [r["prediction_id"] for r in rows] == [plan.plan_prediction_id] * 2 and all(r["found"] for r in rows)
    assert all(r["predicted_recovery"] == branch.predicted_outcome.recovery_if_acted for r in rows)
    assert all(abs(r["recovery_error"] - (1 - r["predicted_recovery"])) < 1e-12 for r in rows)


def test_a_plan_pays_one_visit_charge_whatever_its_number_of_steps(world, tmp_path):
    cycles, blocks = world
    rules = default_rules(cycles, blocks)
    rules["prices"] = {**rules["prices"], "visit": 15}
    episode = FixtureEpisode(tmp_path, cycles, blocks, START, seed=0, rules=rules)
    episode.read("tier3")
    with hidden_files_sealed():
        ranking = CausalTools(episode_models(episode.start), episode.released_path).rank_root_causes(N_CANDIDATES)
    for branch in ranking.branches:
        plan = branch.repair_plan
        assert plan.price == sum(s.part_price + s.repair_charge for s in plan.steps) + (15 if plan.steps else 0)
        assert plan.visit_charge == (15 if plan.steps else 0) and plan.visits == (1 if plan.steps else 0)
    assert any(len(b.repair_plan.steps) > 1 for b in ranking.branches)


def test_a_pump_plan_holds_the_valve_and_a_pump_order_alone_leaves_the_pump_degraded():
    """The pump's repair takes only when the valve and the accumulator are healthy at its close, so the plan of a
    combination with the pump and the valve degraded repairs the valve first and lists it as the pump's dependency in the
    plan, the plan restores the combination, and the predict step of a pump order alone predicts no change."""
    view = profile_view()
    combination = (100, 80, 1, 130)
    faulty = view.faulty(combination)
    plan = repair_plan(view, combination, faulty, costs(view, 0), 1.0, 1, lambda components, rank: "pred/b1")
    assert [s.component for s in plan.steps] == ["valve", "pump"]
    pump = plan.steps[1]
    assert pump.depends_on == ["valve", "accumulator"] and pump.dependencies_in_plan == ["valve"] and pump.close_offset > plan.steps[0].close_offset
    assert restored(view, combination, faulty) and not restored(view, combination, ["pump"])
    flt = Filter(view, episode_models(START).effect, Episode(0, [], [], None), resampled=False, prior=None, support=[combination])
    alone = visit_schedule(view, ["pump"])
    after_alone = flt.ahead(alone[-1].close_offset + 1, [("pump", "standard", alone[-1].close_offset)]).posterior().current_posterior()
    both = visit_schedule(view, faulty)
    after_both = flt.ahead(both[-1].close_offset + 1, [(s.component, "standard", s.close_offset) for s in both]).posterior().current_posterior()
    assert after_alone == {combination: 1.0} and after_both == {view.healthy_combination(): 1.0}


def test_the_cycles_to_restored_follow_schedule_visit_for_every_faulty_set():
    view, config = profile_view(), EpisodeConfig()
    for size in range(1, len(view.components) + 1):
        for components in itertools.combinations(view.components, size):
            visit = visit_schedule(view, list(components))
            clock = schedule_visit(config, 0, [(c, "standard") for c in components])
            assert [(s.component, s.close_offset) for s in visit] == [(components[i], close) for i, _execute, close in clock]
            assert plan_terms(view, list(components))[1] == clock[-1][2] + 1
    assert plan_terms(view, []) == (0.0, 0)


def test_a_plan_costs_one_call_per_order_on_the_free_base_row():
    rules = dict(profile_view().rules)
    prices = rules["prices"]
    free = {**prices, "read": {g: 0 for g in prices["read"]}, "parts": {c: 0 for c in prices["parts"]}, "repair_charge": 0, "visit": 0}
    view = profile_view({**rules, "prices": free})
    combination = (100, 80, 1, 130)
    faulty = view.faulty(combination)
    plan = repair_plan(view, combination, faulty, costs(view, 0), 0.0, 1, lambda components, rank: "pred/b1")
    assert plan_terms(view, faulty)[0] == plan.price == len(faulty) and plan.downtime_cost == 0 and plan.visit_charge == 0
    assert plan_terms(view, faulty)[1] == plan_terms(profile_view(), faulty)[1]


def test_a_group_with_nothing_left_confirms_nothing():
    """A sensor group that the row can serve nothing more shows no separation and a pair is offered only when the row can
    serve both of its reads, while confirm_tests without a count gives the result of before under the same seed."""
    import numpy as np

    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.troubleshooting_plans import confirm_tests
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.decision_rule import decide
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import Posterior
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import Left
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.test_belief_at_one_cycle import random_model

    view = profile_view()
    post = Posterior.uniform(random_model(noise=0.01))
    cost = costs(view, 0)
    decision = decide(post, cost, samples=32, seed=0)
    k = post.order()[0]
    groups = sorted(cost.test_price)

    def serving_of(cycles):
        return {g: Left(cycles.get(g, 0), cycles.get(g, 0) * len(view.group_fields(g))) for g in groups}

    def run(**serving):
        return confirm_tests(post, [k], cost, decision, 32, np.random.default_rng(1), "full", **serving)

    assert run() == run(serving=None)
    confirm, two_step = run(serving=serving_of({groups[0]: 1}))
    assert all(confirm.gap_by_group[g] == 0.0 for g in groups[1:])
    assert confirm.sensor_group == groups[0] and confirm.gap > 0 and two_step is None
    _confirm, two_step = run(serving=serving_of({groups[0]: 2}))
    assert two_step is None or two_step.sensor_groups == [groups[0], groups[0]]


def test_the_cost_of_testing_first_is_that_of_the_plan_that_is_first_after_the_reading():
    """A belief of two combinations whose plans have two prices, and a confirm test whose drawn readings are replaced by fixed
    ones, three that settle the first combination and two that settle the second: test_first of every branch is the cost of
    the test plus the mass weighted price and downtime of the two plans, since a reading that settles a combination leaves
    nothing beyond the outlay of its plan."""
    from types import SimpleNamespace

    import numpy as np

    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice import decision_rule as de
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.troubleshooting_plans import branch_previews, plan_money
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import Posterior
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.test_belief_at_one_cycle import random_model

    view = profile_view()
    first, second = (100, 100, 1, 130), (100, 100, 0, 90)  # a pump plan and an accumulator plan, of two prices
    assert view.faulty(first) != view.faulty(second)
    post = Posterior.uniform(random_model(noise=0.01), support=[first, second])
    index = {h: post.candidates.index(h) for h in (first, second)}
    post.override([0.6 if h == first else 0.4 if h == second else 0.0 for h in post.candidates])
    cost = de.costs(view, 0)
    settled = np.zeros((5, len(post.candidates)))
    settled[:3, index[first]], settled[3:, index[second]] = 1.0, 1.0
    groups = sorted(cost.test_price)
    decision = de.Decision(True, groups[0], 0.0, 1.0, {g: 1.0 for g in groups}, {g: 0.0 for g in groups}, {g: 2.5 for g in groups}, 5, 0,
                           {g: settled for g in groups}, {g: np.zeros((5, len(post.space.groups[g]))) for g in groups})
    rate = de.expected_downtime_rate(post.candidates, post.probabilities, cost)
    money = {h: sum(plan_money(view, view.faulty(h), cost, rate)) for h in (first, second)}
    assert money[first] != money[second]
    branches = branch_previews(view, post, post, cost, decision, 16, 0, lambda components, rank: "pred/b1",
                               lambda components, rank: SimpleNamespace(summaries=[], recovery=0.5), level_reachable=True,
                               returned=post.order()[:2])
    assert len(branches) == 2
    for branch in branches:
        assert abs(branch.cost_to_restored.test_first - (2.5 + 0.6 * money[first] + 0.4 * money[second])) < 1e-9


@pytest.mark.parametrize("preset", sorted(p.stem for p in preset_files(CONFIGS).values()))
def test_the_visit_rule_of_the_layer_and_of_the_server_agree_on_every_preset(world, preset):
    """The visit rule is written twice, schedule_visit in the repair rules layer and visit_schedule in the server, which reads
    the released rules: on every preset file of the environment, for every set of components and every mode the preset
    offers, both give the same repairs in the same order with the same close cycles, and the server's lead times and
    repair downtimes are the configuration's."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import compose

    cycles, blocks = world
    path = preset_path(preset)
    config = compose(json.loads(path.read_text(encoding="utf-8")))
    view = profile_view(HydraulicRigEnv(cycles, blocks, config).reset(0)["result"])
    for mode in ("standard", *(["emergency"] if config.emergency_on() else [])):
        for size in range(1, len(view.components) + 1):
            for components in itertools.combinations(view.components, size):
                visit = visit_schedule(view, list(components), mode)
                clock = schedule_visit(config, 0, [(c, mode) for c in components])
                assert [(s.component, s.close_offset) for s in visit] == [(components[i], close) for i, _execute, close in clock]
                assert [(s.lead_time, s.repair_downtime) for s in visit] == \
                    [(config.lead_time(s.component, mode), config.durations["repair_downtime"][s.component]) for s in visit]


def test_two_candidates_of_one_faulty_set_share_one_branch_one_walk_and_one_prediction(episode, monkeypatch):
    """The ranking walks one plan per faulty set of the returned candidates. Two
    candidates of one set share one branch, the plan is walked once (one prediction per set of components, E14 unchanged:
    the belief after the repair under the whole posterior), and they share one plan_prediction_id and one predicted value."""
    calls = []
    walk = CausalTools._predicted_action

    def counted(self, view, flt, post, components):
        calls.append(tuple(components))
        return walk(self, view, flt, post, components)

    monkeypatch.setattr(CausalTools, "_predicted_action", counted)
    tools = CausalTools(episode_models(episode.start), episode.released_path)
    with hidden_files_sealed():
        ranking = tools.rank_root_causes(10 ** 6)  # every candidate, as before the default of five
    shared = [b for b in ranking.branches if len(b.ranks) > 1]
    assert shared, "the first ranking holds a faulty set of several candidates"
    assert len(calls) == len(set(calls)), "each set of components is walked once"
    assert len({b.repair_plan.plan_prediction_id for b in ranking.branches}) == len(ranking.branches)
    branch = shared[0]
    for r in branch.ranks:  # every candidate of the set reads the one plan and its one prediction
        assert faulty_now(ranking.candidates[r - 1].current_combination) == branch.faulty_set
    records = {p["id"]: p for p in stored_records(episode)}
    assert records[branch.repair_plan.plan_prediction_id]["recovery"] == branch.predicted_outcome.recovery_if_acted
    assert load_contract()["tools"]["rank_root_causes"]["parameters"] == {"top_k": "integer", "branch_detail": "string", "response_format": "string"}


def test_the_ranking_returns_the_first_top_k_by_mass_and_counts_the_rest(episode):
    """top_k is the caller's number, 5 when left out, with no maximum. The candidates are the first top_k with
    positive mass, by mass and then by the combination in component order; more_candidates counts the positive-mass rest and
    sums its mass; the branches cover the faulty sets of the returned candidates alone, and the advice reads every candidate."""
    tools = CausalTools(episode_models(episode.start), episode.released_path)
    with hidden_files_sealed():
        default, one, wide = tools.rank_root_causes(), tools.rank_root_causes(1), tools.rank_root_causes(10 ** 6)
        refused = [tools.rank_root_causes(k) for k in (0, -1, True, 2.5, "5")]
    assert all(isinstance(r, ErrorResult) for r in refused)
    positive = len(wide.candidates)
    assert wide.more_candidates.count == 0 and wide.more_candidates.mass == 0.0
    for ranking, k in ((default, adapter.DEFAULT_TOP_K), (one, 1)):
        assert [c.model_dump() for c in ranking.candidates] == [c.model_dump() for c in wide.candidates[:k]]
        assert ranking.more_candidates.count == positive - k
        assert abs(ranking.more_candidates.mass - sum(c.probability for c in wide.candidates[k:])) < 1e-12
        assert {tuple(b.faulty_set) for b in ranking.branches} == set(returned_sets(ranking))
        assert all(r <= k for b in ranking.branches for r in b.ranks)
        assert ranking.advice == wide.advice and ranking.read_first == wide.read_first
    order = [(-c.probability, [c.combination[x] for x in COMPONENTS]) for c in wide.candidates]
    assert order == sorted(order)  # by mass, ties by the combination in component order


def test_the_set_masses_sum_every_combination_of_a_set_whatever_top_k_is(episode):
    """set_masses holds one row per faulty set with positive mass, its sorted component names and the mass summed
    over every combination of the set in the belief, the same for top_k 1 as for every candidate, ordered by mass and then by
    the names, and the rows add up to the mass of every positive candidate."""
    tools = CausalTools(episode_models(episode.start), episode.released_path)
    with hidden_files_sealed():
        one, wide = tools.rank_root_causes(1), tools.rank_root_causes(10 ** 6)
    expected = {}
    for c in wide.candidates:
        key = tuple(sorted(f.component for f in c.faults))
        expected[key] = expected.get(key, 0.0) + c.probability
    rows = [(tuple(r.faulty_set), r.mass) for r in wide.set_masses]
    assert [r.model_dump() for r in one.set_masses] == [r.model_dump() for r in wide.set_masses]
    assert {s for s, _ in rows} == set(expected) and all(abs(m - expected[s]) < 1e-12 for s, m in rows)
    assert all(list(s) == sorted(s) for s, _ in rows)
    assert [(-m, s) for s, m in rows] == sorted((-m, s) for s, m in rows)
    assert abs(sum(m for _, m in rows) - sum(c.probability for c in wide.candidates)) < 1e-9


def test_a_branch_does_not_move_with_top_k(episode):
    """The branch of the top set holds the same probability, confirm test and losses whether the ranking returns one
    candidate or all of them, since its members are every combination of the set; only ranks lists the returned ones."""
    tools = CausalTools(episode_models(episode.start), episode.released_path)
    episode.read("tier1")
    with hidden_files_sealed():
        one, wide = tools.rank_root_causes(1), tools.rank_root_causes(10 ** 6)
    top_one = one.branches[0]
    top_wide = next(b for b in wide.branches if b.faulty_set == top_one.faulty_set)
    assert top_one.ranks == [1] and top_wide.ranks[0] == 1
    same = ("probability", "confirm_test", "expected_loss_if_acted", "cost_to_restored")
    assert {f: getattr(top_one, f) for f in same} == {f: getattr(top_wide, f) for f in same}
    assert abs(top_one.probability - set_mass(one, top_one.faulty_set)) < 1e-12
