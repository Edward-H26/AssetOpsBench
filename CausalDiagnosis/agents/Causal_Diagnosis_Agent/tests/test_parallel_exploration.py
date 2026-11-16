"""The schedule of candidate plans of the CausalDiagnosis agent (CausalDiagnosis/agents/Causal_Diagnosis_Agent/candidate_plan_schedule.py): the schedule of the candidate plans on
the board, the batch of consecutive read steps with one ranking after the last, and the switch that turns both off.

The tests on boards build their profiles here, with nested, partly nested and disjoint sensor groups and made-up components, so
that they hold no name and no number of an asset; two of them read every registered asset and the tiny asset of the asset tests
from their files. The tests through the controller play the rig's own action server in process, and one plays an AssetOpsBench
row over stdio, through the helpers of test_causal_diagnosis_agent.py and test_walkthrough.py, whose tests this file does not import. Every test asserts on a
schedule, a board, a plan, a prompt or a trajectory, never on a recorded value."""
import json
import re
from pathlib import Path
from unittest import mock

import pytest

from CausalDiagnosis.experiments.episode_runners import run_episodes  # noqa: E402  the driver, whose option turns the switch off
from CausalDiagnosis.episode_viewer import walkthrough  # noqa: E402  a reader of the batches that plans.json records
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests import test_causal_diagnosis_agent as causal
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import (ORIENTATION_HEADING, ScriptedPlanner, _aob_run, _assert_aob_surface, _assert_episode_files, _play,
                                    act_now, act_now_on_assetopsbench, act_now_steps, decision_plans, plan_json, read_json)
from CausalDiagnosis.episode_viewer.tests.test_walkthrough import STEPS, parts_of, step_rows

from CausalDiagnosis import datasets as assets_package  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import Board, ReadWalk  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import WHY_SCHEDULE, Plan, Step, advice, batch_reads, check_plan, exploration, repair_options, reserve_completion, rule_plan  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import PARALLEL_SERVER, PARALLEL_WITHOUT_SERVER, planning_turn, system_prompt  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.controller import UNJUDGED, CausalLoopRunner
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.agent_options import CausalLoopOptions  # noqa: E402
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.candidate_plan_schedule import EVERY_COMPONENT, REASONS, lines  # noqa: E402

FLAG = "read_first"  # the key of the ranking that says to read first before a diagnosis
COMPONENTS = ("c1", "c2", "c3", "c4", "c5")
ASSETS_DIR = Path(assets_package.__file__).resolve().parent
TINY_ASSET = ASSETS_DIR / "tests" / "tiny_asset.json"
READ_PRICES = re.compile(r"Read prices per cycle: ([^;]+);")
SENSOR_GROUPS = re.compile(r"Sensor groups and read price per cycle: (.+?)\.\n")


# ----- boards without servers -----

def made_up_profile(groups, components=COMPONENTS, read_price=1, part=10, charge=5, visit=3, lead=2, downtime=1, budget=500, horizon=100,
                    max_steps=100, priced=True):
    """A profile in the shape the agent reads, for any sensor groups and components. Without priced it prices nothing and sets no
    limit, as a row that prices nothing does."""
    profile = {"asset_id": "made_up", "components": list(components), "levels": {c: [0, 1] for c in components}, "healthy": {c: 0 for c in components},
               "sensor_groups": {g: list(fields) for g, fields in groups.items()}, "normal_ranges": {}}
    if priced:
        prices = read_price if isinstance(read_price, dict) else {g: read_price for g in groups}
        profile.update(prices={"read": prices, "parts": {c: part for c in components}, "repair_charge": charge, "visit": visit},
                       durations={"read_per_cycle": 1, "advance_per_cycle": 1, "lead_time": {c: lead for c in components},
                                  "repair_downtime": {c: downtime for c in components}},
                       budget=budget, horizon=horizon, max_steps=max_steps)
    return profile


def ranked_board(profile, plans, named=(), flag=True, status=None, advice=None, **board_fields):
    """A board after a ranking whose troubleshooting plans are plans, each (rank, mass, faulty components, test group, test
    cycles), whose read batch is named, whose flag says to read first or not and which carries the advice of the server when one
    is given, as the default reply does; the rank 1 plan is the first candidate."""
    board = Board(**board_fields)
    board.set_profile(profile)
    board.status = dict(status) if status is not None else {"time": 0, "budget_left": profile.get("budget"), "steps_left": profile.get("max_steps"),
                                                            "open_work_orders": []}
    healthy, candidates, branches = profile["healthy"], [], []
    for rank, mass, faulty, group, cycles in plans:
        combination = {c: (next(v for v in profile["levels"][c] if v != healthy[c]) if c in faulty else healthy[c]) for c in profile["components"]}
        candidates.append({"rank": rank, "combination": combination, "current_combination": dict(combination), "probability": mass, "faults": [],
                           "evidence_ids": []})
        steps = [{"component": c, "part_price": 0, "repair_charge": 0, "lead_time": 0, "repair_downtime": 0, "prediction_id": f"pred/b{rank}"}
                 for c in faulty]
        branches.append({"rank": rank, "combination": combination, "probability": mass, "faulty_set": list(faulty),
                         "confirm_test": {"sensor_group": group, "cycles": cycles, "price": 0} if group else None,
                         "repair_plan": {"steps": steps, "plan_prediction_id": f"pred/b{rank}"}, "cost_to_restored": {"act_now": float(rank), "test_first": 9.0}})
    ranking = {"candidates": candidates, FLAG: flag, "read_batch": [{"sensor_group": g, "cycles": c} for g, c in named], "branches": branches}
    if advice is not None:
        ranking["advice"] = advice
    board.set_ranking(ranking)
    return board


def read_advice(named):
    """The advice of the server while its rule names the reads named: the first of them, with every one in its text."""
    group, cycles = named[0]
    return {"action": "read", "sensor_group": group, "cycles": cycles, "rank": None, "declare_above": {},
            "text": "read " + ", then ".join(f"{g} for {c} cycles" for g, c in named) + " before a diagnosis"}


def reads_of(scheduled):
    return [(r.group, r.cycles, r.serves, r.named) for r in scheduled.reads]


def nested_and_disjoint():
    """Five sensor groups: two nested pairs and one group that shares no field with another."""
    return {"a": ["f1"], "b": ["f1", "f2"], "c": ["f3"], "d": ["f4"], "e": ["f4", "f5"]}


def test_a_read_that_several_candidate_plans_share_is_scheduled_once_with_the_plans_it_serves():
    """Two plans share one test, one test lies in a group that the named read covers, and one test lies in a group that the
    test of another plan covers: the named read comes first with the plan it serves, then one read for the shared test and one
    covering read for the nested pair, each with the ranks it serves, and the board shows them as advice."""
    plans = [(1, 0.30, ["c1"], "c", 1), (2, 0.25, ["c2"], "c", 1), (3, 0.20, ["c3"], "d", 1), (4, 0.15, ["c4"], "a", 1), (5, 0.10, ["c5"], "b", 1)]
    board = ranked_board(made_up_profile(nested_and_disjoint()), plans, named=[("e", 1)])
    scheduled = exploration(board)
    assert reads_of(scheduled) == [("e", 1, [3], True), ("c", 1, [1, 2], False), ("b", 1, [4, 5], False)]
    assert scheduled.cut == [] and scheduled.left_out == [] and scheduled.differs
    text = board.render()
    assert "## Schedule of the candidate plans" in text
    assert "one batch: e x1 (the read the server names, serves ranks 3), c x1 (serves ranks 1, 2), b x1 (serves ranks 4, 5)." in text


def test_a_test_that_one_plan_alone_names_is_not_in_the_batch():
    """A test that no other plan shares and no read covers stays the step of its plan: no read is scheduled for it, and its plan
    is not left out."""
    groups = {"a": ["f1"], "c": ["f3"], "g": ["f6"]}
    board = ranked_board(made_up_profile(groups), [(1, 0.5, ["c1"], "c", 1), (2, 0.3, ["c2"], "c", 1), (3, 0.2, ["c3"], "g", 1)])
    scheduled = exploration(board)
    assert reads_of(scheduled) == [("c", 1, [1, 2], False)]
    assert all(r.group != "g" for r in scheduled.reads) and scheduled.left_out == [] and scheduled.cut == []


@pytest.mark.parametrize("limit", ["budget", "step cap", "horizon"])
def test_the_reads_of_a_batch_are_cut_where_they_no_longer_fit_together(limit):
    """Three named reads that each fit alone, while their sum breaks the budget, the step cap or the horizon: the schedule holds
    the prefix that fits and names the read that ends it with its limit, while the reads that the board advises without the
    schedule, each checked alone, hold all three."""
    groups = {"g1": ["f1"], "g2": ["f2"], "g3": ["f3"]}
    plan = [(1, 0.9, ["c1"], "g1", 1)]
    named = [("g1", 1), ("g2", 1), ("g3", 1)]
    # the reserve is the completion of the repair of the first candidate (reserve_completion: the diagnosis, its one order, the
    # waits of its visit, the verifying read and finish); each limit leaves room for the reserve and two reads (budget, 15 each)
    # or one read (step cap, horizon, each read one call and one cycle)
    reserve = reserve_completion(ranked_board(made_up_profile(groups, read_price=15), plan, named=named))
    profile = {"budget": made_up_profile(groups, read_price=15, budget=reserve.price + 2 * 15 + 5),
               "step cap": made_up_profile(groups, max_steps=reserve.calls + 1),
               "horizon": made_up_profile(groups, horizon=reserve.cycles + 2)}[limit]
    board = ranked_board(profile, plan, named=named)
    scheduled = exploration(board)
    kept = {"budget": 2, "step cap": 1, "horizon": 1}[limit]
    assert [(r.group, r.cycles) for r in scheduled.reads] == named[:kept]
    assert scheduled.cut == [[named[kept][0], 1, limit]] and scheduled.differs
    assert batch_reads(board) == named
    assert f"Not in the batch: {named[kept][0]} x1, since {REASONS[limit]}." in board.render()


def test_a_plan_that_could_not_be_completed_after_the_named_reads_is_left_out_and_named():
    """A plan whose repair no longer fits the budget once the named reads are paid is left out and named with its limit, while the
    plan of the first candidate, which the reserve keeps, stays served; while the ranking names no read, the line names the limit
    alone."""
    groups = {"g1": ["f1"], "g2": ["f2"]}
    profile = made_up_profile(groups, read_price=10, part=30, budget=100)
    board = ranked_board(profile, [(1, 0.6, ["c1"], "g1", 1), (2, 0.4, ["c2", "c3", "c4"], "g2", 1)], named=[("g1", 2)])
    scheduled = exploration(board)
    assert reads_of(scheduled) == [("g1", 2, [1], True)]
    assert scheduled.left_out == [[2, "budget"]] and scheduled.differs
    assert "Left out: rank 2, whose repair no longer fits the budget after these reads." in board.render()
    closed = ranked_board(profile, [(1, 0.6, ["c1"], "g1", 1), (2, 0.4, ["c2", "c3", "c4"], "g2", 1)], flag=False)
    assert exploration(closed).reads == [] and "Left out: rank 2, whose repair no longer fits the budget." in closed.render()


@pytest.mark.parametrize("budget, added", [(50, False), (90, True), (None, True)])  # every component: 4 x 15 + 3, a read 10, the verifying read 10
def test_a_read_the_server_does_not_name_is_added_only_beside_the_repair_of_every_component(budget, added):
    """A read that two plans share and that the server does not name joins only when the budget pays it beside one visit that
    repairs every component: a budget that pays it beside the most expensive plan and not beside every component gives no read
    and leaves the board as it is without the switch, a larger one gives it, and a profile without prices always does."""
    groups = {"g1": ["f1"], "g2": ["f2"]}
    components = ("c1", "c2", "c3", "c4")
    profile = made_up_profile(groups, components=components, read_price=10, budget=budget, priced=budget is not None)
    board = ranked_board(profile, [(1, 0.5, ["c1"], "g2", 1), (2, 0.4, ["c2"], "g2", 1)])
    scheduled = exploration(board)
    if added:
        assert reads_of(scheduled) == [("g2", 1, [1, 2], False)] and scheduled.cut == []
    else:  # nothing is added, merged or left out, so the board shows no schedule
        assert scheduled.reads == [] and scheduled.cut == [["g2", 1, EVERY_COMPONENT]] and not scheduled.differs
        assert f"Not in the batch: g2 x1, since {REASONS[EVERY_COMPONENT]}." in lines(scheduled, True)
        assert "## Schedule" not in board.render()


def asset_files():
    """Every registered asset file and the tiny asset of the asset tests."""
    return sorted(ASSETS_DIR.glob("*/asset.json")) + [TINY_ASSET]


def asset_profile(path):
    """The profile the agent reads of an asset file, without prices: its components, levels and sensor groups."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = {g: list(kind["channels"]) for g, kind in data["observation_kinds"].items()}
    return data, {"asset_id": data["asset_id"], "components": list(data["components"]), "levels": dict(data["levels"]), "healthy": dict(data["healthy"]),
                  "sensor_groups": groups, "normal_ranges": {}}


@pytest.mark.parametrize("path", asset_files(), ids=lambda p: Path(p).parent.name if Path(p).name == "asset.json" else Path(p).stem)
@pytest.mark.parametrize("recorded", [True, False], ids=["serves_recorded_cycles", "plays_its_cycles"])
@pytest.mark.parametrize("advised", [True, False], ids=["with_the_advice", "without_advice"])
def test_under_one_read_through_a_group_that_covers_every_test_the_board_is_the_board_without_the_switch(path, recorded, advised):
    """On every asset whose sensor groups hold one that covers the others, a board whose named read is that group for every unread
    cycle, while every plan's test lies in some group, with the advice of the server naming that read as the default reply does
    and without it: the read serves every test, so the schedule adds no read, leaves no plan out and does not differ, and the
    board renders byte for byte as with the switch off."""
    _data, profile = asset_profile(path)
    groups = profile["sensor_groups"]
    top = next((g for g in groups if all(set(fields) <= set(groups[g]) for fields in groups.values())), None)
    if top is None:
        pytest.skip("no sensor group of this asset covers the others")
    names = list(groups)
    plans = [(rank, 1.0 / (rank + 1), [c], names[(rank - 1) % len(names)], 1) for rank, c in enumerate(profile["components"], 1)]
    status = {"time": 10, "recorded_from": 0, "recorded_until": 9, "open_work_orders": []} if recorded else {"time": 0, "open_work_orders": []}
    advice = read_advice([(top, 10)]) if advised else None
    boards = [ranked_board(profile, plans, named=[(top, 10)], status=status, advice=advice, parallel=parallel) for parallel in (True, False)]
    scheduled = exploration(boards[0])
    assert [(r.group, r.cycles) for r in scheduled.reads] == [(top, 10)] and scheduled.left_out == [] and not scheduled.differs
    assert boards[0].render() == boards[1].render() and "## Schedule" not in boards[0].render()


def test_a_paid_cycle_is_priced_once_inside_one_plan():
    """On a row that serves recorded cycles with one cycle left to read, a plan that reads a group and then a group it covers pays
    the cycle once: it passes the check with a budget that pays the cycle once and not twice. A walk that reads the covered group
    first pays both, since the smaller group does not hold the fields of the larger."""
    groups = {"small": ["f1"], "big": ["f1", "f2"]}
    profile = made_up_profile(groups, read_price={"small": 4, "big": 10}, budget=10)
    board = ranked_board(profile, [(1, 1.0, [], "small", 1)], flag=False,
                         status={"time": 5, "budget_left": 10, "steps_left": 20, "recorded_from": 4, "recorded_until": 4, "open_work_orders": []})
    plan = Plan(branch=1, mode="test_first", why="x", steps=[Step("read", {"sensor_group": "big", "cycles": 1}, {}, []),
                                                             Step("read", {"sensor_group": "small", "cycles": 1}, {}, []), Step("finish", {}, {}, [])])
    assert check_plan(plan, board) == []
    walk = ReadWalk(board)
    assert walk.take("big", 1)[0] == 10 and walk.take("small", 1)[0] == 0 and walk.price == 10 and walk.calls == 2
    walk = ReadWalk(board)
    assert walk.take("small", 1)[0] == 4 and walk.take("big", 1)[0] == 10 and walk.delivered("small") == 1


def after_refuted(profile, plans, refuted, closed, opened, named, observations_since=0):
    """A board after a refuted repair plan on a row that serves recorded cycles: the diagnosis is made, the order of every closed
    component closed, the order of every opened component open, the refuted set recorded, and the ranking after the verdict
    says to read first and names the reads named. The newest recorded cycle was read through the richest group after the
    close, so a read asks for one cycle that the board does not hold."""
    groups = profile["sensor_groups"]
    richest = max(groups, key=lambda g: len(groups[g]))
    board = Board()
    board.set_profile(profile)
    board.status = {"time": 8, "budget_left": profile.get("budget"), "steps_left": profile.get("max_steps"), "recorded_from": 6, "recorded_until": 7,
                    "open_work_orders": [{"id": f"wo-{c}"} for c in opened]}
    board.add_reading({"sensor_group": richest, "served": [{"n": 1, "t": 7, "values": {f: 1.0 for f in groups[richest]}}]})
    board.diagnosis = {"diagnosis": dict(profile["healthy"]), "evidence_ids": []}
    board.orders = {f"wo-{c}": {"id": f"wo-{c}", "component": c, "created_t": 1, "expected_close_t": 4, "status": "closed" if c in closed else "open",
                                "prediction_id": "pred/b1", "closed_t": 4 if c in closed else None} for c in [*closed, *opened]}
    board.record_refuted(list(refuted), "the reading after the last close is inconsistent with the plan's prediction")
    board.set_ranking(ranked_board(profile, plans, named=named).ranking)
    board.observations += observations_since
    return board


def refuted_board(observations_since=0):
    """After a refuted plan on c1, with the order on c3 open: rank 1 is the refuted set, rank 2 still lists the component of the
    open order, and rank 3 lists the component whose order closed."""
    profile = made_up_profile({"g1": ["f1"], "g2": ["f1", "f2"]}, read_price=2, components=("c1", "c2", "c3", "c4"), budget=400, max_steps=80)
    plans = [(1, 0.5, ["c1"], "g1", 1), (2, 0.3, ["c2", "c3"], "g1", 1), (3, 0.2, ["c1", "c4"], "g2", 1)]
    return after_refuted(profile, plans, ["c1"], closed=["c1"], opened=["c3"], named=[("g2", 2)], observations_since=observations_since)


def test_after_a_refuted_plan_the_schedule_lists_what_each_remaining_plan_needs_and_repeats_no_executed_step():
    """After a refuted plan the refuted set is no job, the component of an open order is not ordered again, a component whose
    order closed is listed only for a plan that still lists it, the read the ranking names is advised while no reading came
    since the verdict and not after one, and a cycle the board paid is not priced again."""
    board = refuted_board()
    scheduled = exploration(board)
    assert [(r["rank"], r["components"]) for r in scheduled.remaining] == [(2, ["c2"]), (3, ["c1", "c4"])]
    assert all(r["verification_group"] in board.profile["sensor_groups"] for r in scheduled.remaining) and scheduled.differs
    assert [(r.group, r.cycles, r.named) for r in scheduled.reads] == [("g2", 2, True)]
    assert ReadWalk(board).take("g2", 2)[0] == 2 * 1  # the board paid cycle 7, so the read pays cycle 6 alone
    text = board.render()
    assert "After the refuted plan, what each remaining plan still needs: rank 2: order c2," in text and "rank 3: order c1, c4," in text
    later = refuted_board(observations_since=1)
    after = exploration(later)
    assert after.reads == [] and [r["rank"] for r in after.remaining] == [2, 3] and after.differs


def test_without_the_server_candidates_that_share_an_indicated_group_get_one_read_and_none_without_channels():
    """Without the server the model's own candidates are the plans: two candidates whose faulty components the same group
    indicates get one read of it, a candidate whose failure modes name no channel gets none, and once a reading of that group
    came after the model listed its candidates the read is served and no schedule is shown."""
    groups = {"g1": ["f1"], "g2": ["f2"]}
    profile = made_up_profile(groups, components=("c1", "c2", "c3"))
    board = Board(server=False)
    board.set_profile(profile)
    board.status = {"time": 0, "budget_left": 500, "steps_left": 100, "open_work_orders": []}
    board.failure_modes = [{"component": "c1", "code": "m1", "channels": ["f1"]}, {"component": "c2", "code": "m2", "channels": ["f1"]},
                           {"component": "c3", "code": "m3", "channels": []}]
    board.set_candidates([{"levels": {"c1": 1}, "failure_mode_ids": []}, {"levels": {"c2": 1}, "failure_mode_ids": []},
                          {"levels": {"c3": 1}, "failure_mode_ids": []}])
    scheduled = exploration(board)
    assert reads_of(scheduled) == [("g1", 1, [1, 2], False)] and scheduled.differs
    assert "## Schedule of your candidates" in board.render() and "g1 x1 (serves your candidates 1, 2)" in board.render()
    board.add_reading({"sensor_group": "g1", "served": [{"n": 1, "t": 0, "values": {"f1": 1.0}}]})
    assert exploration(board).reads == [] and "## Schedule" not in board.render()


def test_the_rule_plans_the_scheduled_reads_as_one_run_and_follows_the_board_otherwise():
    """The rule's own plan takes the reads of a schedule that differs as consecutive read steps, and with the switch off, or on a
    board whose schedule does not differ, it plans as the board advises without the schedule."""
    plans = [(1, 0.30, ["c1"], "c", 1), (2, 0.25, ["c2"], "c", 1), (3, 0.20, ["c3"], "d", 1), (4, 0.15, ["c4"], "a", 1), (5, 0.10, ["c5"], "b", 1)]
    on = ranked_board(made_up_profile(nested_and_disjoint()), plans, named=[("e", 1)])
    off = ranked_board(made_up_profile(nested_and_disjoint()), plans, named=[("e", 1)], parallel=False)
    plan = rule_plan(on, 6)
    assert plan.why == WHY_SCHEDULE and [(s.kind, s.args["sensor_group"]) for s in plan.steps] == [("read", "e"), ("read", "c"), ("read", "b")]
    assert [(s.kind, s.args.get("sensor_group")) for s in rule_plan(off, 6).steps] == [("read", "e")] and rule_plan(off, 6).why != WHY_SCHEDULE
    same = [ranked_board(made_up_profile(nested_and_disjoint()), [(1, 0.6, ["c1"], "d", 1), (2, 0.4, ["c2"], "d", 1)], named=[("e", 1)], parallel=p)
            for p in (True, False)]
    assert not exploration(same[0]).differs and rule_plan(same[0], 6).to_json() == rule_plan(same[1], 6).to_json()


def test_under_the_advice_of_the_server_the_reads_of_its_rule_run_as_one_batch_where_it_names_more_than_one():
    """Under the main settings the board advises the read of the server's advice, the first read that its rule names. Where the
    sensor groups are not nested the rule names more than one read: the schedule holds them all, differs from the board without
    it and is shown, and the rule plans them as one batch, while with the switch off it plans the advised read alone. Where the
    rule names one read, the schedule is that read and does not differ."""
    plans = [(1, 0.6, ["c1"], "c", 1), (2, 0.4, ["c2"], "d", 1)]
    named = [("e", 1), ("c", 1)]
    on, off = (ranked_board(made_up_profile(nested_and_disjoint()), plans, named=named, advice=read_advice(named), parallel=p) for p in (True, False))
    scheduled = exploration(on)
    assert [(r.group, r.cycles, r.named) for r in scheduled.reads] == [("e", 1, True), ("c", 1, True)] and scheduled.differs
    assert "Advised reads, the read of the server: e x1." in on.render() and "## Schedule of the candidate plans" in on.render()
    assert rule_plan(on, 6).why == WHY_SCHEDULE and [s.args["sensor_group"] for s in rule_plan(on, 6).steps] == ["e", "c"]
    assert [s.args["sensor_group"] for s in rule_plan(off, 6).steps] == ["e"] and "## Schedule" not in off.render()
    one = ranked_board(made_up_profile(nested_and_disjoint()), plans, named=named[:1], advice=read_advice(named[:1]))
    assert [(r.group, r.cycles) for r in exploration(one).reads] == [("e", 1)] and not exploration(one).differs
    assert "## Schedule" not in one.render()


def whole_words(text):
    return set(re.findall(r"[A-Za-z0-9_]+", text))


def test_the_texts_of_the_schedule_on_the_tiny_asset_hold_no_word_of_another_asset():
    """Every text of the mechanism, the two sentences of the system prompt and every line of the schedule on boards of the tiny
    asset (a named read, a shared read, a read cut by the repair of every component, a plan left out and the plans after a
    refuted one), holds no whole word of another registered asset: its components, sensor groups, channel tags, short noun and
    asset id, the words computed from the asset files less the words the tiny asset holds itself."""
    tiny, profile = asset_profile(TINY_ASSET)
    banned = set()
    for path in sorted(ASSETS_DIR.glob("*/asset.json")):
        data, other = asset_profile(path)
        banned |= set(other["components"]) | set(other["sensor_groups"]) | set(data["channels"]) | {data["asset_id"], data["text"]["short_noun"]}
    banned -= set(profile["components"]) | set(profile["sensor_groups"]) | set(tiny["channels"]) | {tiny["asset_id"], tiny["text"]["short_noun"]}
    groups = list(profile["sensor_groups"])
    small, big = sorted(groups, key=lambda g: len(profile["sensor_groups"][g]))[0], sorted(groups, key=lambda g: len(profile["sensor_groups"][g]))[-1]
    priced = {**profile, "prices": {"read": {g: 10 for g in groups}, "parts": {c: 30 for c in profile["components"]}, "repair_charge": 5, "visit": 3},
              "durations": {"read_per_cycle": 1, "advance_per_cycle": 1, "lead_time": {c: 2 for c in profile["components"]},
                            "repair_downtime": {c: 1 for c in profile["components"]}}, "budget": 120, "horizon": 100, "max_steps": 100}
    one, two, three = profile["components"][:3]
    texts = [PARALLEL_SERVER, PARALLEL_WITHOUT_SERVER, *REASONS.values()]
    shared = ranked_board({**priced, "budget": 115}, [(1, 0.5, [one], small, 1), (2, 0.3, [two], small, 1), (3, 0.2, [one, two, three], big, 1)],
                          named=[(big, 1)])
    texts += lines(exploration(shared), True)
    tight = ranked_board({**priced, "budget": 80}, [(1, 0.5, [one], small, 1), (2, 0.4, [two], small, 1)])
    texts += lines(exploration(tight), True)
    refuted = after_refuted(priced, [(1, 0.5, [one], small, 1), (2, 0.3, [two, three], small, 1)], [one], closed=[one], opened=[], named=[(big, 1)])
    texts += lines(exploration(refuted), True)
    assert any(t.startswith("After the refuted plan") for t in texts)
    texts += [line for line in system_prompt(6).split("\n") if "Consecutive read steps" in line]
    assert any("Left out:" in t for t in texts) and any("Not in the batch:" in t for t in texts)
    assert not banned & whole_words("\n".join(texts)), sorted(banned & whole_words("\n".join(texts)))


# ----- the controller, in process on the rig's own action server -----

def groups_of(prompt):
    """The sensor groups a board prices, in the order of the profile."""
    match = READ_PRICES.search(prompt)
    assert match, "the board shows no read prices"
    return [item.rsplit(" ", 1)[0] for item in match.group(1).split(", ")]


def covering_pair(prompts):
    """A sensor group and a group that covers it, (smaller, larger), from the profile that the orientation shows."""
    text = next(p for p in prompts if ORIENTATION_HEADING in p)
    match = SENSOR_GROUPS.search(text)
    assert match, "the orientation shows no sensor groups"
    fields = {}
    for item in match.group(1).split("; "):
        name, rest = item.split(" (", 1)
        fields[name] = set(rest.split("): ", 1)[1].split(", "))
    pairs = [(a, b) for a in fields for b in fields if a != b and fields[a] < fields[b]]
    assert pairs, "no sensor group of this asset covers another"
    return min(pairs, key=lambda pair: (len(fields[pair[0]]), -len(fields[pair[1]])))


def reads_first(count, expect=None):
    """A planner whose first decision plan reads count sensor groups of the board in a row, each with expect or with no
    expectation, and which acts on the top candidate after it."""
    def plans(prompt, call):
        if call == 1:
            reads = [{"kind": "read", "args": {"sensor_group": g, "cycles": 1}, "expect": dict((expect or {}).get(i, {})), "evidence": []}
                     for i, g in enumerate(groups_of(prompt)[:count])]
            return plan_json(reads, mode="test_first")
        return act_now(prompt, call)
    return ScriptedPlanner(plans)


def tools_after(trajectory, start):
    return [t["tool"] for t in trajectory[start:]]


def first_batch(ep_dir):
    """The trajectory and the index of the first read call of the first decision plan."""
    trajectory = read_json(ep_dir / "trajectory.json")
    first = next(i for i, t in enumerate(trajectory) if t["tool"] == "read_sensors")
    return trajectory, first


@pytest.mark.parametrize("parallel", [True, False], ids=["on", "off"])
def test_consecutive_read_steps_run_with_one_ranking_after_the_last(tmp_path, parallel):
    """A plan of three read steps: with the switch on the trajectory holds the three reads and then one ranking, every read step
    carries the top of that ranking, and the board's batches hold one entry of three reads with one ranking; with it off each
    read is followed by its own ranking."""
    ep_dir, row, _backend = _play(tmp_path, reads_first(3), {} if parallel else {"parallel": False})
    _assert_episode_files(ep_dir, row)
    trajectory, first = first_batch(ep_dir)
    head = tools_after(trajectory, first)[:6]
    if parallel:
        assert head[:4] == ["read_sensors", "read_sensors", "read_sensors", "rank_root_causes"]
    else:
        assert head == ["read_sensors", "rank_root_causes"] * 3
    plans = read_json(ep_dir / "plans.json")
    batch_plan = decision_plans(plans)[0]
    reads = [s for s in batch_plan["steps"] if s["kind"] == "read"]
    assert len(reads) == 3 and all(s["status"] == "done" for s in reads)
    if parallel:
        assert len({json.dumps(s["observed"]["top"]) for s in reads}) == 1
        assert {"plan": batch_plan["index"], "first_step": 1, "reads": [[s["args"]["sensor_group"], 1] for s in reads], "rankings": 1} in plans["board"]["batches"]
    else:
        assert [b["reads"] for b in plans["board"]["batches"] if b["plan"] == batch_plan["index"]] == [[[s["args"]["sensor_group"], 1]] for s in reads]


@pytest.mark.parametrize("parallel", [True, False], ids=["on", "off"])
def test_the_walkthrough_places_every_call_of_a_batch_on_its_plan_step(tmp_path, parallel):
    """The walkthrough of an episode whose first decision plan reads three groups: every call lands on a plan step, the one
    ranking after the batch on its last read, and the process says that the batch took the effective hypotheses to the value
    after it; with the switch off each read has its own ranking and the process speaks of each read alone."""
    ep_dir, _row, _backend = _play(tmp_path, reads_first(3), {} if parallel else {"parallel": False})
    text = walkthrough.render(ep_dir)
    rows = step_rows(parts_of(text)[STEPS])
    assert len(rows) == len(read_json(ep_dir / "trajectory.json")) and all(r["Plan step"] for r in rows)
    index = decision_plans(read_json(ep_dir / "plans.json"))[0]["index"]
    labels = [r["Plan step"].split(",")[0] for r in rows if r["Plan step"].startswith(f"plan {index} step ")]
    if parallel:
        assert labels[:4] == [f"plan {index} step {k}" for k in (1, 2, 3, 3)]
        assert "and the batch of 3 reads it ran in, from step 1," in text
    else:
        assert labels[:6] == [f"plan {index} step {k}" for k in (1, 1, 2, 2, 3, 3)]
        assert "the batch of" not in text


class RefusingSecondRead(causal.InProcessServers):
    """The stand-in servers of test_causal_diagnosis_agent.py, whose second read of the episode is refused."""

    reads = 0

    async def call(self, path, tool, args):
        if tool == "read_sensors":
            self.reads += 1
            if self.reads == 2:
                return json.dumps({"error": "the read is refused for this test"})
        return await super().call(path, tool, args)


def test_a_read_that_is_refused_ends_the_batch_and_the_reads_before_it_are_ranked_once(tmp_path):
    """A batch of three reads whose second is refused: the third is not called, the reading made is ranked once, and the plan
    stops on the refused read, whose record comes last."""
    with mock.patch.object(causal, "InProcessServers", RefusingSecondRead):
        ep_dir, row, _backend = _play(tmp_path, reads_first(3), {})
    trajectory, first = first_batch(ep_dir)
    assert tools_after(trajectory, first)[:3] == ["read_sensors", "read_sensors", "rank_root_causes"]
    plan = decision_plans(read_json(ep_dir / "plans.json"))[0]
    assert [s["status"] for s in plan["steps"]] == ["done", "surprise"] and plan["surprise"]["step"] == 2
    assert plan["surprise"]["reason"].startswith("read_sensors refused:")


def test_the_surprise_of_a_batch_is_the_first_expectation_that_fails(tmp_path):
    """A batch of three reads whose second expects fewer effective hypotheses than any belief holds: the first read is done, the
    second is the surprise, and the third, made before the ranking, is done with its expectation not judged."""
    ep_dir, row, _backend = _play(tmp_path, reads_first(3, expect={1: {"effective_at_most": 0.5}}), {})
    plan = decision_plans(read_json(ep_dir / "plans.json"))[0]
    assert [s["status"] for s in plan["steps"]] == ["done", "surprise", "done"] and plan["surprise"]["step"] == 2
    assert plan["steps"][2]["note"] == UNJUDGED and "eliminated nothing" in plan["surprise"]["reason"]


def acts_then_reads_twice(prompt):
    """Diagnose and order the top candidate's repair, wait past its close, then read twice in a row, the first read with an
    expectation that no belief meets, without a finish, so that the verification after the batch decides."""
    steps = act_now_steps(prompt)[:-2]  # the diagnosis, the orders and the wait, without the verifying read and the finish
    steps[-1] = {**steps[-1], "args": {"cycles": 2 * steps[-1]["args"]["cycles"]}}
    group = groups_of(prompt)[-1]
    steps += [{"kind": "read", "args": {"sensor_group": group, "cycles": 1}, "expect": {"effective_at_most": 0.5}, "evidence": []},
              {"kind": "read", "args": {"sensor_group": group, "cycles": 1}, "expect": {}, "evidence": []}]
    assert len(steps) <= 6, "the plan length of the controller"
    return plan_json(steps)


def advice_then_reads_twice():
    """A planner that reads what the board advises while nothing is committed, then acts once with acts_then_reads_twice, and
    acts on the top candidate after that."""
    acted = []

    def plans(prompt, call):
        advised = causal.advised_read(prompt)
        if advised:
            return advised
        if not acted and "Commitment: none" in prompt:
            acted.append(call)
            return acts_then_reads_twice(prompt)
        return act_now(prompt, call)
    return ScriptedPlanner(plans)


def test_a_refuted_plan_is_the_surprise_of_the_last_read_made_before_any_expectation(tmp_path):
    """Two reads after the close of a repair whose verdict the stand-in turns inconsistent: the refuted plan is the surprise of the
    last read made, and the first read, whose expectation would fail, is done with its expectation not judged."""
    ep_dir, row, _backend = _play(tmp_path, advice_then_reads_twice(), {}, refute_first=True)
    plans = read_json(ep_dir / "plans.json")
    refuted = next(p for p in plans["plans"] if p["surprise"] and "was refuted" in p["surprise"]["reason"])
    steps = refuted["steps"]
    assert steps[-1]["kind"] == "read" and steps[-1]["status"] == "surprise" and steps[-2]["kind"] == "read"
    assert steps[-2]["status"] == "done" and steps[-2]["note"] == UNJUDGED and refuted["surprise"]["step"] == len(steps)


def rankings_after_reads(trajectory, first, reads):
    """The ranking that follows the reads-th read call from first on, as the server answered it."""
    seen = 0
    for t in trajectory[first:]:
        seen += t["tool"] == "read_sensors"
        if seen == reads and t["tool"] == "rank_root_causes":
            return json.loads(t["response"])
    raise AssertionError("no ranking follows the reads")


def test_the_ranking_after_a_batch_is_the_ranking_after_the_same_readings_one_by_one(tmp_path):
    """The same two reads in one batch and one by one, on a row whose clock moves only by the declared durations, so that the
    rankings between them take no time: the ranking after the batch names the candidates with the masses of the ranking after
    the second read taken alone. A controller that drops a reading of the batch fails it."""
    rankings = []
    for parallel in (True, False):
        ep_dir, _row, _backend = _play(tmp_path / str(parallel), reads_first(2), {} if parallel else {"parallel": False})
        trajectory, first = first_batch(ep_dir)
        ranking = rankings_after_reads(trajectory, first, 2)
        rankings.append([(json.dumps(c["combination"], sort_keys=True), round(c["probability"], 9)) for c in ranking["candidates"]])
    assert rankings[0] == rankings[1]


def test_with_the_switch_off_the_agent_is_the_agent_without_the_mechanism(tmp_path):
    """With the switch off no prompt holds a sentence of the mechanism, no board shows a schedule, every read is followed by its
    own ranking, no plan records a schedule, the label ends in seq, and the switch is accepted without the server. The option
    of the driver sets it off and gives that label, and without the option the switch stays on."""
    parser = run_episodes.build_parser()
    sequential = run_episodes.causal_options(parser.parse_args(["--agent", "causal", "--causal-sequential"]))
    assert sequential["parallel"] is False and run_episodes.agent_label_of("causal", "m", sequential) == "causal_seq_m"
    assert run_episodes.causal_options(parser.parse_args(["--agent", "causal"]))["parallel"] is True
    assert run_episodes.agent_label_of("causal", "m", run_episodes.causal_options(parser.parse_args(["--agent", "causal"]))) == "causal_m"
    ep_dir, row, backend = _play(tmp_path, reads_first(3), {"parallel": False})
    prompts = [c["prompt"] for c in backend.calls]
    assert prompts and not any(PARALLEL_SERVER.strip() in p or "## Schedule" in p for p in prompts)
    assert row["agent"] == "causal_seq_scripted-causal"
    trajectory = read_json(ep_dir / "trajectory.json")
    tools = [t["tool"] for t in trajectory]
    for i, tool in enumerate(tools):
        if tool == "read_sensors" and i + 1 < len(tools):
            assert tools[i + 1] == "rank_root_causes", tools[i:i + 3]
    assert all(p["schedule"] is None for p in read_json(ep_dir / "plans.json")["plans"])
    plans = [(1, 0.30, ["c1"], "c", 1), (2, 0.25, ["c2"], "c", 1)]
    off = ranked_board(made_up_profile(nested_and_disjoint()), plans, named=[("e", 1)], parallel=False)
    assert exploration(off) is None and "## Schedule" not in off.render()
    assert CausalLoopRunner(llm=None, server_paths={"hydraulic_rig": "x"}, options=CausalLoopOptions(parallel=False)).options.parallel is False


class ReadFirstAfterTheVerdict(causal.InProcessServers):
    """The stand-in servers of test_causal_diagnosis_agent.py that refute the first verified verdict, and whose ranking after that verdict
    says to read first, once, naming the group of the first elimination row. A refuted verdict says that the repair did not
    restore the asset, so every ranking after it leaves out the branches under which no component is faulty, whatever the
    recorded cycles after the repair show: the plans left to repair are the faulty ones. The candidates stay as ranked, since a
    plan names its branch by a rank among them."""

    def __init__(self, ep_dir, **fake):
        super().__init__(ep_dir, **{**fake, "refute_first": True})
        self.read_first_after = False

    async def call(self, path, tool, args):
        text = await super().call(path, tool, args)
        if tool == "rank_root_causes" and self.refuted:
            ranking = json.loads(text)
            if "error" in ranking:
                return text
            ranking["branches"] = [b for b in ranking.get("branches") or [] if b.get("faulty_set")]
            rows = ranking.get("elimination") or []
            if rows and not self.read_first_after:
                ranking[FLAG] = True
                ranking["read_batch"] = [{"sensor_group": rows[0]["sensor_group"], "cycles": 1}]
                self.read_first_after = True
            text = json.dumps(ranking)
        return text


@pytest.mark.parametrize("parallel", [True, False], ids=["on", "off"])
def test_after_a_refuted_plan_the_agent_reads_what_the_remaining_plans_share_before_the_next_repair(tmp_path, parallel):
    """The rule plays an episode whose first verified verdict is refuted, and the ranking after the verdict says to read first:
    with the switch on the plan after the verdict holds the scheduled read before any order, no order repeats the refuted set,
    and the episode ends with finish; with it off the plan after the verdict orders at once."""
    planner = ScriptedPlanner(lambda prompt, call: pytest.fail("the rule only controller must not ask for a plan"))
    with mock.patch.object(causal, "InProcessServers", ReadFirstAfterTheVerdict):
        ep_dir, row, _backend = _play(tmp_path, planner, {"rule_only": True, **({} if parallel else {"parallel": False})})
    verdict = _assert_episode_files(ep_dir, row)
    plans = read_json(ep_dir / "plans.json")
    refuted = plans["board"]["refuted"]
    assert len(refuted) == 1
    surprise = next(p for p in plans["plans"] if p["surprise"] and "was refuted" in p["surprise"]["reason"])
    after = next(p for p in plans["plans"] if p["index"] > surprise["index"])
    if parallel:
        assert after["plan"]["why"] == WHY_SCHEDULE and [s["kind"] for s in after["plan"]["steps"]] == ["read"] * len(after["plan"]["steps"])
        assert after["schedule"]["reads"] and after["schedule"]["remaining"]
        later = [s["args"]["component"] for p in plans["plans"] if p["index"] > after["index"] for s in p["plan"]["steps"] if s["kind"] == "order"]
        assert later and sorted(set(later)) != refuted[0]["components"]
    else:
        assert after["plan"]["steps"][0]["kind"] == "order" and after["schedule"] is None
    assert verdict["end_reason"] == "finish"


# ----- an AssetOpsBench row over stdio -----

@pytest.mark.parametrize("parallel", [True, False], ids=["on", "off"])
def test_a_batch_on_a_row_that_serves_recorded_cycles_makes_one_ranking_and_verifies_its_repair(tmp_path, parallel):
    """On the real-time row with the pinned clock a plan reads a sensor group and then a group that covers it: with the switch on
    one ranking follows the two reads, with it off two, and either way the episode ends with its repair verified and accepted by
    the verifier."""
    holder = {}

    def plans(prompt, call):
        if call == 1:
            small, big = covering_pair(holder["planner"].prompts)
            reads = [{"kind": "read", "args": {"sensor_group": g, "cycles": 1}, "expect": {}, "evidence": []} for g in (small, big)]
            return plan_json(reads, mode="test_first")
        return act_now_on_assetopsbench(prompt, call)

    planner = holder["planner"] = ScriptedPlanner(plans)
    ep_dir, row, backend = _aob_run(tmp_path, "real_time", planner, {} if parallel else {"parallel": False})
    trajectory, _verdict = _assert_aob_surface(ep_dir, row, clock=True)
    first = next(i for i, t in enumerate(trajectory) if t["tool"] == "history")
    head = [t["tool"] for t in trajectory[first:first + 4]]
    if parallel:
        assert head[:3] == ["history", "history", "rank_root_causes"]
    else:
        assert head[:4] == ["history", "rank_root_causes", "history", "rank_root_causes"]


# ----- the advice when no candidate plan is left -----

# the mass of every faulty set as the server sums it over every combination of the set: c2 and c3, faulty at two levels each, hold
# the most in four combinations that the first two returned candidates leave out; rank 1 is c1 and rank 2 the healthy asset
SET_MASSES = [{"faulty_set": ["c2", "c3"], "mass": 0.44}, {"faulty_set": ["c1"], "mass": 0.3}, {"faulty_set": [], "mass": 0.26}]
PLANS_LEFT = [(1, 0.3, ["c1"], "g1", 1), (2, 0.26, [], None, 1)]  # the previews of the two returned candidates
# the two forms of the line, each before the components it names
REFUTED_FORM = ("The last repair plan was refuted and no other plan is left, so the advice is to order the repair of the components "
                "that the ranking still holds as faulty, those of the faulty set with the most mass that hold no open work order and "
                "lie outside the refuted set: ")
GENERAL_FORM = ("No candidate plan is left, so the advice is to order the repair of the components that the ranking still holds "
                "as faulty, those of the faulty set with the most mass that hold no open work order: ")


def no_plan_profile():
    """Four components, of which c2 and c3 hold two faulty levels each, so that their faulty set spreads over four combinations."""
    profile = made_up_profile({"g1": ["f1"], "g2": ["f1", "f2"]}, components=("c1", "c2", "c3", "c4"))
    profile["levels"].update(c2=[0, 1, 2], c3=[0, 1, 2])
    return profile


def refuted_without_other_plan(plans=PLANS_LEFT, set_masses=SET_MASSES, opened=()):
    """The board after a refuted repair of c1 whose ranking previews no plan that repairs a component outside the refuted set."""
    board = after_refuted(no_plan_profile(), plans, ["c1"], closed=["c1"], opened=list(opened), named=[])
    board.ranking["set_masses"] = set_masses
    return board


def still_faulty_lines(board):
    return [text for text in advice(board) if "still holds as faulty" in text]


def test_after_a_refuted_plan_with_no_other_plan_left_the_advice_names_what_the_ranking_still_holds_as_faulty():
    """The previews are the refuted set c1 and the healthy asset, so no other plan is left, and one line of advice names the
    components of the faulty set with the most mass, c2 and c3, which no returned candidate holds, and not c1 of rank 1: the line
    reads the mass the server sums over every combination of a set, which holds for any asset and does not move with top_k. It
    stands in the planning prompt, leaves out a component whose work order is open, and is absent when the set with the most mass
    is the healthy asset."""
    board = refuted_without_other_plan()
    assert repair_options(board) == [] and board.faulty_set(board.ranked_top()["combination"]) == ["c1"]
    line = REFUTED_FORM + "c2, c3."
    assert still_faulty_lines(board) == [line] and line in planning_turn(board, None)
    open_c3 = refuted_without_other_plan(opened=["c3"])
    assert still_faulty_lines(open_c3) == [REFUTED_FORM + "c2."]
    healthy = refuted_without_other_plan(set_masses=[{"faulty_set": [], "mass": 0.5}, {"faulty_set": ["c1"], "mass": 0.3},
                                                     {"faulty_set": ["c2"], "mass": 0.2}])
    assert still_faulty_lines(healthy) == []


def test_the_line_names_a_repair_only_when_the_budget_left_pays_it():
    """The line advises no repair that the plan check would refuse for its price: with the budget left one below the completion
    of the repair of c2 and c3, priced as repair_options prices a plan, no line names them, and with the budget left at that
    price the line stands, the control."""
    board = refuted_without_other_plan()
    price = board.completion(["c2", "c3"]).price
    board.status["budget_left"] = price - 1
    assert repair_options(board) == [] and still_faulty_lines(board) == []
    board.status["budget_left"] = price
    assert still_faulty_lines(board) == [REFUTED_FORM + "c2, c3."]


def test_after_a_commitment_with_no_plan_offered_the_advice_names_what_the_ranking_still_holds_as_faulty():
    """With the troubleshooting plans hidden no plan is offered: before the commitment the advice acts on rank 1 and names no
    component, and once the diagnosis is made one line names the components of the faulty set with the most mass."""
    board = ranked_board(no_plan_profile(), PLANS_LEFT, flag=False, show_branches=False)
    board.ranking["set_masses"] = SET_MASSES
    assert still_faulty_lines(board) == [] and "Advised repair plan, with no troubleshooting plan shown: act on rank 1, the top candidate." in advice(board)
    board.diagnosis = {"diagnosis": dict(board.candidate(1)["combination"]), "evidence_ids": []}
    assert repair_options(board) == [] and still_faulty_lines(board) == [GENERAL_FORM + "c2, c3."]


def test_while_a_candidate_plan_is_left_the_advice_names_the_plan_and_no_component():
    """Control: with a previewed plan of c2 and c3 left after the refuted plan, the advice names that plan as the cheapest other
    plan and no line names the components; after a commitment with plans offered and nothing refuted, no line names them either."""
    with_plan = [*PLANS_LEFT, (3, 0.11, ["c2", "c3"], "g2", 1)]
    board = refuted_without_other_plan(plans=with_plan)
    assert repair_options(board) == [(3.0, 3)] and still_faulty_lines(board) == []
    assert any(text.startswith("The last repair plan was refuted, so the advice is to repair rank 3, the cheapest other plan") for text in advice(board))
    offered = ranked_board(no_plan_profile(), with_plan, flag=False)
    offered.ranking["set_masses"] = SET_MASSES
    offered.diagnosis = {"diagnosis": dict(offered.candidate(1)["combination"]), "evidence_ids": []}
    assert repair_options(offered) == [(1.0, 1), (3.0, 3)] and still_faulty_lines(offered) == []


def test_after_a_refuted_plan_the_line_leaves_out_the_refuted_set_that_the_board_and_the_prompt_keep_out():
    """The board, the system prompt and the wait step say that the advice keeps the last refuted set out until a later ranking
    names it first, so the line leaves out its components: with the refuted c1 alone on top no line stands, with c1 and c2 on top
    the line names c2 alone, and with c2 and c3 on top it names both, the control."""
    on_top = refuted_without_other_plan(set_masses=[{"faulty_set": ["c1"], "mass": 0.6}, {"faulty_set": [], "mass": 0.4}])
    assert repair_options(on_top) == [] and on_top.top_set() == (["c1"], 0.6) and on_top.refuted_last == ["c1"]
    assert still_faulty_lines(on_top) == []
    shared = refuted_without_other_plan(set_masses=[{"faulty_set": ["c1", "c2"], "mass": 0.55}, {"faulty_set": [], "mass": 0.45}])
    assert still_faulty_lines(shared) == [REFUTED_FORM + "c2."]
    control = refuted_without_other_plan()
    assert still_faulty_lines(control) == [REFUTED_FORM + "c2, c3."]
