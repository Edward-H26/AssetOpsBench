"""The advice of rank_root_causes, rule by rule against the verifier's own verdict on constructed episodes, played in
process on the engine of the AssetOpsBench rows from a development combination: on the base preset, whose rules price
nothing, and on the preset with prices, where the advice is one read of the record that the budget pays beside the repair
of every component, then the diagnosis. A ranking is asked for in the compact form unless a test names the long form, which
adds the block e10_rule of the earlier rule."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.gymnasium_wrapper import oracle_cost
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.harness import preset_config
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import tools as adapter
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice import decision_rule as de
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import CausalTools
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import profile_view
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.troubleshooting_plans import plan_terms
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import Left
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests import advice_walk as aw
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.fixture_episode import assert_compact_shortens, episode_models
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as server

pytestmark = pytest.mark.skipif(not Path(default_facts_path()).exists(), reason="hydraulic rig facts file not present; set HYD_FACTS_PATH")


@pytest.fixture(scope="module")
def start():
    """A development combination with three faults, the first in grid order."""
    return next(s for s in aw.development_combinations() if sum(1 for c, level in zip(profile_view().components, s) if level != profile_view().healthy[c]) == 3)


class Episode:
    """One episode on the engine of the AssetOpsBench rows from the start, with the released evidence written beside it."""

    def __init__(self, start, preset="base", entry=0, seed=0, **overrides):
        cycles, blocks = aw.world()
        config = aw.started(preset, start, entry)
        if overrides:
            config = preset_config(preset, **{**{k: getattr(config, k) for k in ("fault_rule", "start")}, **overrides})
        self.start, self.config, self.seed, self.world = tuple(start), config, seed, (cycles, blocks)
        self.env = RecordedRigEnv(cycles, blocks, config)
        self.env.reset(seed)
        self.folder = tempfile.TemporaryDirectory()
        self.shown = {g: set() for g in config.sensor_groups}

    def rank(self, top_k=3, detail="compact", **options):
        state = Path(self.folder.name) / "state.json"
        server.write_released(state, self.env)
        return CausalTools(episode_models(self.start), server.released_path(state), **options).rank_root_causes(top_k, detail)

    def read(self, group, cycles):
        return aw.read(self.env, group, cycles, self.shown)

    def view(self):
        """The view the server builds from the released rules of the episode."""
        state = Path(self.folder.name) / "state.json"
        server.write_released(state, self.env)
        raw = json.loads(server.released_path(state).read_text(encoding="utf-8"))
        return profile_view(dict(raw["rules"], asset_id=raw["asset_id"]))

    def cited_reading(self):
        """An identifier of the first summary the last read released, as an agent cites a reading."""
        served = next(e for e in reversed(self.env.log) if e["event"] == "read")["served"]
        return f"obs/{served[0]['n']}/{sorted(served[0]['values'])[0]}"

    def declare(self):
        """A declaration that cites a reading, after one read of the richest group."""
        group = richest(profile_view())
        self.read(group, 5)
        self.env.step({"type": "declare_insufficient_evidence", "separating_test": {"sensor_group": group, "cycles": 1},
                       "evidence_ids": [self.cited_reading()]})

    def act(self, combination):
        """The diagnosis of the combination, one work order for each of its faulty components, and the finish."""
        diagnosis = dict(zip(self.config.components, combination))
        self.env.step({"type": "submit_diagnosis", "diagnosis": diagnosis, "evidence_ids": []})
        for c in self.config.components:
            if diagnosis[c] != self.config.healthy[c]:
                self.env.step({"type": "create_work_order", "component": c, "prediction_id": "", "evidence_ids": []})
        self.env.step({"type": "finish", "summary": "done"})

    def verdict(self, grade=None):
        cycles, blocks = self.world
        return verifier.verify(self.env.log, verifier.released_from_env(self.env), self.env.oracle(),
                               oracle_cost(RecordedRigEnv(cycles, blocks, self.config), self.seed), grade=grade, scope=self.config.verifier_scope)


def richest(view):
    return max(view.sensor_groups, key=lambda g: (len(view.sensor_groups[g]), g))


def faults(combination):
    view = profile_view()
    return [c for c, level in zip(view.components, combination) if level != view.healthy[c]]


def test_a_wrong_level_an_unnecessary_order_and_a_missed_fault_each_fail_the_episode_that_the_loss_prices(start):
    """Four episodes from one start: the truth, one level off, one order too many, one fault left. The verifier passes the
    first alone, and the loss of a row that prices nothing charges the other three one failed episode each."""
    view = profile_view()
    levels = dict(zip(view.components, start))
    healthy_one = next(c for c in view.components if levels[c] == view.healthy[c])
    faulty_one = faults(start)[0]
    off = dict(levels, **{faulty_one: next(lv for lv in view.levels[faulty_one] if lv not in (levels[faulty_one], view.healthy[faulty_one]))})
    extra = dict(levels, **{healthy_one: next(lv for lv in view.levels[healthy_one] if lv != view.healthy[healthy_one])})
    left = dict(levels, **{faulty_one: view.healthy[faulty_one]})
    acts = [tuple(start)] + [tuple(a[c] for c in view.components) for a in (off, extra, left)]
    passes = []
    for act in acts:
        episode = Episode(start)
        episode.act(act)
        passes.append(episode.verdict()["pass"])
    assert passes == [True, False, False, False]
    cost = de.costs(Episode(start).view(), time=0, steps_left=30)
    assert cost.unit == "calls"
    assert [float(de.decision_loss(acts, cost)[a, 0]) for a in range(len(acts))] == [0.0, 30.0, 30.0, 30.0]


def test_a_reading_lowers_neither_measure_on_a_row_that_prices_nothing(start):
    plain, read = Episode(start), Episode(start)
    read.read(richest(profile_view()), 5)
    for episode in (plain, read):
        episode.act(tuple(start))
    assert [(v["diagnosis_all_correct"], v["pass"]) for v in (plain.verdict(), read.verdict())] == [(True, True), (True, True)]


def test_the_advice_reads_every_recorded_cycle_before_the_first_diagnosis(start):
    episode = Episode(start)
    first = episode.rank()
    recorded = len(episode.env.recorded_cycles())
    assert first.read_first and first.advice.action == "read"
    assert first.advice.sensor_group == first.separating_test.sensor_group == richest(profile_view())
    assert first.advice.cycles == first.separating_test.cycles == recorded
    assert [(b.sensor_group, b.cycles) for b in first.read_batch] == [(first.advice.sensor_group, recorded)]
    episode.read(first.advice.sensor_group, first.advice.cycles)
    after = episode.rank()
    assert not after.read_first and after.separating_test is None and after.read_batch == []
    assert after.advice.action == "diagnose" and after.advice.rank == 1


def test_the_server_offers_no_test_the_row_cannot_serve(start):
    view = profile_view()
    smallest = min(view.sensor_groups, key=lambda g: (len(view.sensor_groups[g]), g))
    episode = Episode(start)
    episode.read(smallest, 5)
    reply = episode.rank(detail="summary")  # the edge cutting score stands in the long forms alone
    row = next(e for e in reply.elimination if e.sensor_group == smallest)
    assert row.information_gain == 0.0 and row.ec2 == 0.0
    assert reply.separating_test is not None and reply.separating_test.sensor_group != smallest
    assert smallest not in [b.sensor_group for b in reply.read_batch]


def test_the_advice_never_declares_under_the_decision_rule_and_the_verifier_gives_a_declaration_zero(start):
    _log, _verdict, replies = aw.walk(start)
    assert all(r.advice.action in ("read", "diagnose") for r in replies)
    assert all(r.advice.declare_above["top1"] is None and r.advice.declare_above["pass"] is None for r in replies)
    episode = Episode(start)
    episode.declare()
    episode.env.step({"type": "finish", "summary": "declared"})
    plain = episode.verdict()
    assert plain["abstention_cites_reading"] and not plain["diagnosis_all_correct"] and not plain["pass"]
    inseparable = {"separable": False, "share": 0.0, "rollouts": 1, "source": "test", "rules_key": "test", "e18": {"separable": False}}
    graded = episode.verdict(grade=inseparable)
    assert graded["success"] and graded["abstention_correct"]
    failed = dict(graded, abstention_correct=False, success=False)
    assert abs(verifier.reward(graded, episode.config.horizon) - verifier.reward(failed, episode.config.horizon) - de.DECLARATION_WORTH["reward"]) < 1e-12


def test_under_the_fixed_cut_the_advice_keeps_the_cut(start):
    episode = Episode(start)
    first = episode.rank(abstention_rule="fixed_cut", abstention_cut=0.95)
    assert first.advice.action == "read" and first.advice.cycles == len(episode.env.recorded_cycles())
    episode.read(first.advice.sensor_group, first.advice.cycles)
    mass = episode.rank().candidates[0].probability
    above, below = min(1.0, mass + (1 - mass) / 2), mass / 2
    assert episode.rank(abstention_rule="fixed_cut", abstention_cut=above).advice.action == "declare"
    assert episode.rank(abstention_rule="fixed_cut", abstention_cut=below).advice.action == "diagnose"


def test_the_advice_leaves_the_calls_of_the_diagnosis_the_orders_and_the_finish(start):
    """With one call less than a read needs beside the act the action is diagnose although a group is open, and the flag,
    the separating test and the batch say the same."""
    orders = len(faults(Episode(start).rank().candidates[0].combination.values()))
    tight = Episode(start, max_steps=1 + orders + 2 - 1)
    reply = tight.rank()
    assert reply.advice.action == "diagnose" and not reply.read_first and reply.separating_test is None and reply.read_batch == []
    assert "a reading is left" in reply.message
    room = Episode(start, max_steps=1 + orders + 2).rank()
    assert room.advice.action == "read" and room.read_first


def test_decide_on_a_row_with_prices_is_given_the_serving_and_draws_as_before_without_it(start, monkeypatch):
    """The adapter hands decide the count of what the row can still serve on every row, a row with prices too, and decide
    given none draws as before: every group in name order from one seeded generator."""
    calls = []
    original = adapter.causal_tools.decide

    def spy(*args, **kwargs):
        calls.append(kwargs.get("serving", "missing"))
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter.causal_tools, "decide", spy)
    episode = Episode(start, preset="priced")
    episode.rank()
    assert len(calls) == 1 and set(calls[0]) == set(episode.view().sensor_groups)
    view = episode.view()
    model = __import__("CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.test_belief_at_one_cycle", fromlist=["random_model"]).random_model(noise=0.01)
    post = __import__("CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle", fromlist=["Posterior"]).Posterior.uniform(model)
    cost = de.costs(view, 0)
    decision = original(post, cost, samples=16, seed=3, serving=None)
    rng = np.random.default_rng(3)
    for group in sorted(cost.test_price):
        assert np.array_equal(decision.draws[group], post.sample_next(post.space.groups[group], 16, rng))


def test_no_advice_after_a_diagnosis_or_a_declaration(start):
    diagnosed = Episode(start)
    diagnosed.env.step({"type": "submit_diagnosis", "diagnosis": dict(zip(diagnosed.config.components, start)), "evidence_ids": []})
    assert diagnosed.rank().advice is None
    declared = Episode(start)
    declared.declare()
    assert any(e["event"] == "declare_insufficient_evidence" for e in declared.env.log)
    assert declared.rank().advice is None


def test_the_belief_does_not_read_the_serving(start, monkeypatch):
    episode = Episode(start)
    plain = episode.rank()
    view = profile_view()
    monkeypatch.setattr(adapter.causal_tools, "left_in_episode", lambda view_, episode_, window: {g: Left(0, 0) for g in view.sensor_groups})
    closed = episode.rank()
    assert [(c.combination, c.probability) for c in plain.candidates] == [(c.combination, c.probability) for c in closed.candidates]
    assert plain.advice.action == "read" and closed.advice.action == "diagnose"


def test_on_a_start_with_no_fault_the_advice_is_the_diagnosis_with_no_order():
    """The healthy combination as the start: after the record is read the advice is the diagnosis of rank 1 with no work
    order, and the episode that follows it passes the verifier. It needs a fault rule of the starts that admits no fault."""
    view = profile_view()
    healthy = tuple(view.healthy[c] for c in view.components)
    log, verdict, replies = aw.walk(healthy)
    last = replies[-1]
    assert last.advice.action == "diagnose" and "create no work order" in last.advice.text
    assert tuple(last.candidates[0].combination[c] for c in view.components) == healthy
    assert not any(e["event"] == "work_order_created" for e in log)
    assert verdict["diagnosis_all_correct"] and verdict["recovered"] and verdict["unnecessary"] == 0 and verdict["pass"]


def priced_episode(start, money):
    """An episode of the preset with prices whose budget leaves the given money beside the repair of every component."""
    view = Episode(start, preset="priced").view()
    return Episode(start, preset="priced", budget=plan_terms(view, list(view.components))[0] + money)


def test_on_a_row_with_prices_the_advice_reads_the_record_once_then_diagnoses(start):
    """On the preset with prices the first ranking advises the group with every channel for every recorded cycle, which the
    budget of the preset pays beside the repair of every component, and after that one read the advice is the diagnosis of
    rank 1."""
    episode = Episode(start, preset="priced")
    first = episode.rank()
    assert first.advice.action == "read" and first.advice.sensor_group == richest(profile_view())
    assert first.advice.cycles == len(episode.env.recorded_cycles())
    episode.read(first.advice.sensor_group, first.advice.cycles)
    after = episode.rank()
    assert after.advice.action == "diagnose" and after.advice.rank == 1


def test_the_read_of_a_row_with_prices_is_the_richest_group_the_money_pays_beside_the_repair_of_every_component(start):
    """Money for one cycle of the second richest group beside the repair of every component, less than one cycle of the
    richest, names one cycle of the second richest, and less than one cycle of the cheapest group gives the diagnosis."""
    view = Episode(start, preset="priced").view()
    price = de.costs(view, 0).test_price
    richest_first = sorted(view.sensor_groups, key=lambda g: (-len(view.sensor_groups[g]), g))
    second, cheapest = richest_first[1], min(view.sensor_groups, key=lambda g: (price[g], g))
    assert price[second] < price[richest_first[0]]
    advice = priced_episode(start, price[second]).rank().advice
    assert (advice.action, advice.sensor_group, advice.cycles) == ("read", second, 1)
    assert priced_episode(start, price[cheapest] / 2).rank().advice.action == "diagnose"


def test_the_cycles_of_the_read_are_what_the_money_left_pays(start):
    """Money for two cycles and a half of the richest group beside the repair of every component names two cycles of it,
    while the record holds more."""
    episode = Episode(start, preset="priced")
    group = richest(profile_view())
    assert len(episode.env.recorded_cycles()) > 2
    advice = priced_episode(start, 2.5 * de.costs(episode.view(), 0).test_price[group]).rank().advice
    assert (advice.action, advice.sensor_group, advice.cycles) == ("read", group, 2)


def test_a_row_with_prices_never_advises_a_declaration(start):
    """Every ranking of a walk that follows the advice on the preset with prices advises a read or the diagnosis."""
    _log, _verdict, replies = aw.walk(start, preset="priced")
    assert replies and all(r.advice is not None and r.advice.action in ("read", "diagnose") for r in replies)


def test_the_misdiagnosis_price_moves_no_advice(start):
    """The price of a wrong diagnosis at the value of the table and at the budget gives the same advice, before the read of
    the record and after it."""
    advices = []
    for at_budget in (False, True):
        episode = Episode(start, preset="priced")
        if at_budget:
            episode = Episode(start, preset="priced", prices={**episode.config.prices, "misdiagnosis": episode.config.budget})
        first = episode.rank()
        episode.read(first.advice.sensor_group, first.advice.cycles)
        advices.append((first.advice.model_dump(), episode.rank().advice.model_dump()))
    assert advices[0] == advices[1]


TOP_LEVEL_OF_THE_RULE = ("read_first", "separating_test", "read_batch", "advice")


def test_the_forms_differ_in_e10_rule_and_in_what_compact_leaves_out(start):
    """On the preset with prices at the first ranking the summary form adds e10_rule, E10 and E22 of the base, the compact form
    leaves out and shortens what assert_compact_shortens names, and every other field is the same in both forms: the top level
    states the one read of the rule in each."""
    episode = Episode(start, preset="priced")
    short, long = episode.rank(detail="compact"), episode.rank(detail="summary")
    assert short.e10_rule is None and long.e10_rule is not None
    assert short.read_first and (short.separating_test.sensor_group, short.separating_test.cycles) == (short.advice.sensor_group, short.advice.cycles)
    assert [(b.sensor_group, b.cycles) for b in short.read_batch] == [(short.advice.sensor_group, short.advice.cycles)]
    rule = long.e10_rule
    assert rule.read_first == (min(long.expected_loss_test.values()) < long.expected_loss_act)
    assert rule.separating_test is None or rule.separating_test.cycles == 1
    assert rule.batch_cap is None and all(b.cycles == 1 for b in rule.read_batch)
    assert "at most" not in rule.text and "at most 1 reads" in episode.rank(detail="summary", batch_cap=1).e10_rule.text
    assert_compact_shortens(short, long)


def test_the_long_form_holds_the_earlier_rule_with_the_fields_of_the_card(start):
    """The summary form holds e10_rule and the edge cutting scores of E21, and the compact form neither. A margin above any
    saving turns read_first of the block off, batch_cap 1 leaves at most one read in its batch and reserve_plans 2 its reserve,
    and none of the three moves a top-level field, the advice or the compact form."""
    episode = Episode(start, preset="priced")
    plain = episode.rank(detail="summary")
    compact = episode.rank(detail="compact")
    assert compact.e10_rule is None and all(e.ec2 is None and e.ec2_per_cost is None for e in compact.elimination)
    assert all(e.ec2 is not None and e.ec2_per_cost is not None for e in plain.elimination)
    for card in ({"e10_margin": 1e9}, {"batch_cap": 1}, {"reserve_plans": 2}):
        carded = episode.rank(detail="summary", **card)
        assert [getattr(carded, name) for name in TOP_LEVEL_OF_THE_RULE] == [getattr(plain, name) for name in TOP_LEVEL_OF_THE_RULE]
        assert episode.rank(detail="compact", **card) == episode.rank(detail="compact")
        assert carded.e10_rule.margin == card.get("e10_margin", 0.0) and carded.e10_rule.batch_cap == card.get("batch_cap")
        assert carded.e10_rule.reserve_plans == card.get("reserve_plans", 1)
    off = episode.rank(detail="summary", e10_margin=1e9).e10_rule
    assert not off.read_first and off.separating_test is None and off.read_batch == []
    assert len(episode.rank(detail="summary", batch_cap=1).e10_rule.read_batch) <= 1


def test_the_batch_leaves_the_repair_and_the_read_that_verifies_it_as_the_agent_keeps_them(start):
    """Told equals done on a row with prices: with a budget one read short of the repair of the top candidate, its
    cheapest verifying read and the cheapest further read, the batch of the long form buys no read the agent's plan check would
    refuse, since its reserve holds the repair and the read that verifies it, as the agent's reserve does."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import verifying_read_price
    probe = Episode(start, preset="priced")
    view = probe.view()
    top = [c for c, level in probe.rank(detail="summary").candidates[0].combination.items() if level != view.healthy[c]]
    verify = verifying_read_price(view, top, verifier.released_from_env(probe.env)["events"])
    cheapest = min(view.rules["prices"]["read"].values())
    assert top and verify > 0 and cheapest > 0
    for money in (cheapest + verify - 1e-6, cheapest + verify + cheapest):
        budget = plan_terms(view, top)[0] + money
        rule = Episode(start, preset="priced", budget=budget).rank(detail="summary").e10_rule
        spent = sum(view.rules["prices"]["read"][b.sensor_group] for b in rule.read_batch)
        assert spent + plan_terms(view, top)[0] + verify <= budget + 1e-9


def test_the_compact_form_runs_no_search_of_e22(start, monkeypatch):
    """The compact form calls no read_batch, and the summary form calls it once while read_first of e10_rule is on."""
    calls = []
    original = adapter.causal_tools.read_batch

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter.causal_tools, "read_batch", counted)
    episode = Episode(start, preset="priced")
    episode.rank(detail="compact")
    assert calls == []
    long = episode.rank(detail="summary")
    assert calls == ([1] if long.e10_rule.read_first else [])


# ----- what a row with a clock can still serve -----

def released_episode(episode):
    """The released evidence of the episode as the server parses it."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import parse_released

    state = Path(episode.folder.name) / "state.json"
    server.write_released(state, episode.env)
    return parse_released(json.loads(server.released_path(state).read_text(encoding="utf-8")))


def clock_episode(start, cycles):
    """An episode of the preset with prices and the real-time clock whose budget pays the repair of every component and the
    given number of cycles of the richest group beside it."""
    view = Episode(start, preset="real_time").view()
    price = de.costs(view, 0).test_price[richest(view)]
    return Episode(start, preset="real_time", budget=plan_terms(view, list(view.components))[0] + price * cycles)


def test_on_a_row_with_a_clock_the_first_advice_reads_every_recorded_cycle_and_the_next_is_the_diagnosis(start):
    """k cycles played before the first ranking: the advice reads the window recorded before the episode and the k played
    cycles through the richest group the budget pays, and after that read the advice is the diagnosis of rank 1 although the
    clock has recorded further cycles, since every group has been shown a window of recordings."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import LIVE_WINDOW

    k = 3
    episode = clock_episode(start, LIVE_WINDOW + k + 2)
    episode.env.step({"type": "get_current_time"}, think_cycles=k)
    first = episode.rank()
    assert len(episode.env.recorded_cycles()) == LIVE_WINDOW + k
    assert (first.advice.action, first.advice.sensor_group, first.advice.cycles) == ("read", richest(episode.view()), LIVE_WINDOW + k)
    episode.read(first.advice.sensor_group, first.advice.cycles)
    episode.env.step({"type": "get_current_time"}, think_cycles=2)
    after = episode.rank()
    assert len(episode.env.recorded_cycles()) == LIVE_WINDOW + k + 2
    assert after.advice.action == "diagnose" and not after.read_first and after.read_batch == []


def test_after_a_repair_the_flag_names_the_cycles_recorded_since_the_closure(start):
    """After the read of the record, the diagnosis and the repair of every faulty component of the start, the cycles that the
    clock records after the closure are what the row can still serve: the flag is on, its read takes those cycles, and
    left_to_read names them for every group, the reading that verify_outcome needs."""
    episode = clock_episode(start, 1000)
    first = episode.rank()
    episode.read(first.advice.sensor_group, first.advice.cycles)
    episode.env.step({"type": "submit_diagnosis", "diagnosis": dict(zip(episode.config.components, start)), "evidence_ids": []})
    for component in faults(start):
        episode.env.step({"type": "create_work_order", "component": component, "prediction_id": "", "evidence_ids": []})
    for _ in range(20):
        if not episode.env.public_work_orders():
            break
        episode.env.step({"type": "wait_for_notification", "timeout": 5})
    episode.env.step({"type": "get_current_time"}, think_cycles=2)
    closure = released_episode(episode).closures[-1]
    since = [t for t in episode.env.recorded_cycles() if t >= closure.t]
    ranking = episode.rank()
    assert since and ranking.read_first and ranking.separating_test.cycles == len(since)
    assert ranking.left_to_read == {g: len(since) for g in episode.config.sensor_groups}


def test_the_reply_says_what_every_group_can_still_be_read(start):
    """left_to_read holds, for every sensor group, the cycles of the count of what the row can still serve, and it is zero for
    every group once nothing is left."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import LIVE_WINDOW
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import left_in_episode

    episode = clock_episode(start, LIVE_WINDOW + 4)
    episode.env.step({"type": "get_current_time"}, think_cycles=2)
    ranking = episode.rank()
    left = left_in_episode(episode.view(), released_episode(episode), LIVE_WINDOW)
    assert ranking.left_to_read == {g: left[g].cycles for g in left} and all(ranking.left_to_read.values())
    episode.read(ranking.advice.sensor_group, ranking.advice.cycles)
    assert set(episode.rank().left_to_read.values()) == {0}
