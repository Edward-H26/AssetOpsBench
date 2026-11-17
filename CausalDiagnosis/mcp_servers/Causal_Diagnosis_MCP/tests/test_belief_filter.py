"""The filter of version 7: on the starts of the old fault rule without a transition it equals the joint_block posterior of
version 6, the schedule kernel walks the paths of draw_schedule, the hazard kernel reproduces the drop frequencies of the
clock's rule, a repair takes only when its dependencies are healthy, an emergency hold expires, the fix history weighs the
prior as the history rule says, a replay is deterministic, the predict and update steps of EffectDiagnostics replay the
same belief, and no module but asset_model reads an asset table."""
from CausalDiagnosis.utils.paths import CONFIGS, ROOT, preset_path
import ast
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from CausalDiagnosis.environment.Causal_Diagnosis_Environment import layers
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_5_deterioration_and_failure import degradation_order, draw_hazard, draw_schedule, hazard_rate, worst_level
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import build_blocks, default_facts_path, load_cycles
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief import belief_filter as fl
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import CausalTools
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import PRIVATE_KEYS, profile_view
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.diagnostics_model import PROTOCOL_MEMBERS, DiagnosticsModel, EffectDiagnostics, diagnostics_backend
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import normalized
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.fixture_episode import FixtureEpisode, episode_models
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as rig
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP.policy_through_tools import tool_call

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)

START = (100, 80, 0, 90)
PACKAGE = Path(__file__).resolve().parents[1]
PRESETS = CONFIGS


@pytest.fixture(scope="module")
def world():
    cycles = load_cycles(default_facts_path())
    return cycles, build_blocks(cycles)


def view_with(**processes):
    config = EpisodeConfig().to_json()
    return profile_view({"processes": {key: {**config[key], **value} for key, value in processes.items()}})


def one_state(view, combination, sched=None):
    """States with one row at the combination, weight one under one fit, no hold, entered before any reading."""
    rules = fl.Rules(view, episode_models(START).space)
    sched = np.zeros((1, 0, 3), dtype=int) if sched is None else np.array([sched], dtype=int)
    return fl.States(rules, np.array([rules.positions(combination)]), np.zeros(1, dtype=int), sched, np.full((1, rules.J), -1),
                     np.full((1, rules.J), -1), np.zeros((1, 1)), {"offset": np.zeros((1, 1, 1))})


def combination_of(states, row=0):
    rules = states.rules
    return tuple(rules.space.levels[j][rules.level_index_of_position[j][p]] for j, p in enumerate(states.pos[row]))


def test_the_filter_equals_the_joint_block_posterior_of_version_6_on_the_starts_without_a_transition(world, tmp_path):
    """With the support cut to the 47 starts of one or two faults, a uniform prior and no transition, the filter's masses
    equal the marginal likelihood of the stacked readings under one block offset shared by the block, each start with the
    covariances of the fit that scores it, normalized over the starts, which is the joint_block posterior of
    version 6."""
    cycles, blocks = world
    episode = FixtureEpisode(tmp_path, cycles, blocks, START, seed=0)
    episode.read("tier1")
    episode.read("tier3", cycles=2)
    episode.read("tier2")
    released = json.loads(episode.released_path.read_text(encoding="utf-8"))
    view, effect = profile_view(released["rules"]), episode_models(START).effect
    starts = [h for h in view.combinations if 1 <= view.n_faulty(h) <= 2]
    assert len(starts) == 47
    parsed = fl.parse_released(released)
    post = fl.Filter(view, effect, parsed, resampled=False, prior=None, support=starts).posterior()
    observations = fl.merge_observations(parsed.readings, effect.space)
    index = np.concatenate([o.index for o in observations])
    y = np.concatenate([o.x for o in observations])
    def cholesky_of(h):
        own = effect.space.combo_index[h]
        covariance = effect.sigma_block[own][np.ix_(index, index)].copy()
        start = 0
        for o in observations:
            covariance[start:start + len(o.index), start:start + len(o.index)] += effect.sigma_within[own][np.ix_(o.index, o.index)]
            start += len(o.index)
        return np.linalg.cholesky(covariance)
    log_likelihood = np.array([fl.log_density((y - effect.mu(h)[index])[None], cholesky_of(h))[0] for h in starts])
    reference = normalized(log_likelihood)
    masses = np.array([post.probabilities[effect.space.combo_index[h]] for h in starts])
    assert np.allclose(masses, reference, rtol=1e-6, atol=1e-12) and abs(masses.sum() - 1) < 1e-12
    assert post.n_states == len(starts)


class Picks:
    """A stand in for the generator of draw_schedule that picks the given component and first drop cycle."""

    def __init__(self, component, first):
        self.component, self.first = component, first

    def sample(self, population, k):
        return [self.component][:k] if self.component in population else []

    def randint(self, low, high):
        return self.first


@pytest.mark.parametrize("start", [(100, 100, 0, 130), (3, 100, 0, 90), (3, 73, 2, 90)])
def test_every_schedule_draw_walks_the_path_of_draw_schedule(start):
    config = EpisodeConfig(schedule={"mode": "seeded", "components": 1, "first_drop": [5, 9], "interval": 10})
    view = view_with(schedule=config.schedule)
    rules = fl.Rules(view, episode_models(START).space)
    draws = rules.schedule_draws(start)
    assert abs(sum(math.exp(log_p) for log_p, _ in draws) - 1) < 1e-12
    eligible = [c for c in view.components if start[view.components.index(c)] == view.healthy[c]] or list(view.components)
    assert len(draws) == len(eligible) * 5
    for _log_p, draw in draws:
        (j, tau, l0), = draw
        component = view.components[j]
        expected = [(d["t"], d["component"], d["to_level"]) for d in draw_schedule(Picks(component, tau), config, list(start))]
        states, path = one_state(view, start, [[j, tau, l0]]), []
        for t in range(config.horizon):
            before = combination_of(states)
            fl.schedule_kernel(states, t)
            after = combination_of(states)
            path += [(t, c, after[k]) for k, c in enumerate(view.components) if after[k] != before[k]]
        assert path == expected, (component, tau)


def test_the_hazard_kernel_reproduces_the_drop_frequencies_of_the_clock_rule():
    """The exact distribution of the kernel after a few cycles against the frequencies of seeded runs of the clock's own
    rule, a drop when the uniform of draw_hazard falls below hazard_rate, within four binomial standard errors."""
    hazard = {"mode": "stochastic", "base_rate": 0.05, "depth_factor": 1.0, "collateral": 0.0}
    config = EpisodeConfig(hazard=hazard)
    view = view_with(hazard=hazard)
    start, cycles, runs = (100, 90, 0, 115), 6, 4000
    states = one_state(view, start)
    for _t in range(cycles):
        fl.hazard_kernel(states)
        states.changed[:] = False
        states.merge()
    exact = {}
    for row in range(len(states)):
        exact[combination_of(states, row)] = exact.get(combination_of(states, row), 0.0) + float(np.exp(states.logw[0, row]))
    assert abs(sum(exact.values()) - 1) < 1e-12
    counts = {}
    for run in range(runs):
        table = draw_hazard(random.Random(f"hazard-test:{run}"), config)
        levels = dict(zip(config.components, start))
        for t in range(cycles):
            for i, c in enumerate(config.components):
                if levels[c] != worst_level(config, c) and table["drop"][t][i] < hazard_rate(config, c, levels[c]):
                    order = degradation_order(config, c)
                    levels[c] = order[order.index(levels[c]) + 1]
        final = tuple(levels[c] for c in config.components)
        counts[final] = counts.get(final, 0) + 1
    assert set(counts) <= set(exact)
    for combination, p in exact.items():
        observed = counts.get(combination, 0) / runs
        assert abs(observed - p) <= 4 * math.sqrt(p * (1 - p) / runs) + 1 / runs, combination


def test_a_repair_takes_only_when_the_components_it_depends_on_are_healthy():
    view = view_with()
    pump = view.components.index("pump")
    assert view.dependencies("pump") == ("valve", "accumulator")
    blocked, free = one_state(view, (100, 80, 1, 130)), one_state(view, (100, 100, 1, 130))
    for states in (blocked, free):
        fl.close_kernel(states, pump, "standard", 5)
    assert combination_of(blocked) == (100, 80, 1, 130) and not blocked.changed[0]
    assert combination_of(free) == (100, 100, 0, 130) and free.changed[0]
    healthy_pump = one_state(view, (100, 100, 0, 130))
    fl.close_kernel(healthy_pump, pump, "standard", 5)
    assert combination_of(healthy_pump) == (100, 100, 0, 130) and len(healthy_pump) == 1  # no collateral damage at rate zero


def test_an_emergency_repair_holds_until_its_expiry_and_a_standard_close_ends_the_hold():
    view = view_with()
    hold = view.processes["emergency"]["hold"]
    valve = view.components.index("valve")
    states = one_state(view, (100, 80, 0, 130))
    fl.close_kernel(states, valve, "emergency", 5)
    assert combination_of(states) == (100, 100, 0, 130) and states.hold_exp[0, valve] == 5 + hold
    fl.expiry_kernel(states, 5 + hold - 1)
    assert combination_of(states) == (100, 100, 0, 130)
    fl.expiry_kernel(states, 5 + hold)
    assert combination_of(states) == (100, 80, 0, 130) and states.hold_exp[0, valve] == -1
    ended = one_state(view, (100, 80, 0, 130))
    fl.close_kernel(ended, valve, "emergency", 5)
    fl.close_kernel(ended, valve, "standard", 7)
    fl.expiry_kernel(ended, 5 + hold)
    assert combination_of(ended) == (100, 100, 0, 130) and ended.hold_exp[0, valve] == -1


def test_the_fix_history_weighs_the_prior_by_the_history_rule():
    view, effect = view_with(), episode_models(START).effect
    rule = view.processes["fix_history"]
    history = [{"id": "hist/1", "component": "valve", "closed_t": -10, "mode": "standard", "outcome": "closed"}]
    post = fl.Filter(view, effect, fl.Episode(0, [], [], history), resampled=False).posterior()
    index = effect.space.combo_index
    degraded, healthy = post.probabilities[index[(100, 80, 0, 130)]], post.probabilities[index[(100, 100, 0, 130)]]
    width = {kind: rule[f"{kind}_ago"][1] - rule[f"{kind}_ago"][0] + 1 for kind in ("healthy", "degraded")}
    expected = (rule["degraded_probability"] / width["degraded"]) / (rule["healthy_probability"] / width["healthy"])
    assert abs(degraded / healthy - expected) < 1e-9
    uniform = fl.Filter(view, effect, fl.Episode(0, [], [], None), resampled=False).posterior()
    assert np.allclose(uniform.probabilities, 1 / len(view.combinations))


def test_a_replay_is_deterministic_and_the_view_holds_no_private_key(world, tmp_path):
    cycles, blocks = world
    episode = FixtureEpisode(tmp_path, cycles, blocks, START, seed=1)
    episode.read("tier2", cycles=2)
    episode.advance(3)
    episode.read("tier3")
    released = json.loads(episode.released_path.read_text(encoding="utf-8"))
    released["rules"].update({"allowed": [[1, 2, 3, 4]], "start": {"combination": list(START)}, "seed": 5})
    view = profile_view(released["rules"])
    assert not any(key in view.rules for key in PRIVATE_KEYS)
    first, second = (fl.Filter(view, episode_models(START).effect, fl.parse_released(released)).posterior() for _ in range(2))
    assert np.array_equal(first.probabilities, second.probabilities) and np.array_equal(first.mean, second.mean)
    assert np.array_equal(first.resampled_probabilities, second.resampled_probabilities)


def test_only_asset_model_reads_the_tables_of_an_asset():
    """Every module of the package but asset_model imports from the rig packages at most the generic arithmetic of the
    classifiers, the environment's rules of the cost unit of a row and of a row without a clock, the key of the stopped rate, the
    key of a combination in the label table, and the public constant of
    the reward for a correct declaration with the format of the released evidence and the events that commit an episode, so
    the asset layer replaces one file."""
    allowed = {"CausalDiagnosis.environment.reference_policies.tier_classifiers": {"ALPHA_GRID", "dm_log_likelihood", "lda_fit", "lda_predict", "scatter"},
               "CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config": {"STOPPED", "cost_unit", "has_clock"},
               "CausalDiagnosis.environment.Causal_Diagnosis_Environment.start_state_seeds": {"combo_key"},
               "CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier": {"ABSTENTION_CREDIT", "COMMIT_EVENTS", "RELEASED_FORMAT"}}
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name == "asset_model.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(("envs.", "CausalDiagnosis.mcp_servers.Environment_Actions_MCP")):
                names = {alias.name for alias in node.names}
                assert names <= allowed.get(node.module, set()), (path.name, node.module, names)
            if isinstance(node, ast.Import):
                assert not any(alias.name.startswith(("envs.", "CausalDiagnosis.mcp_servers.Environment_Actions_MCP")) for alias in node.names), path.name


def test_the_steps_of_effect_diagnostics_replay_the_belief_of_the_filter(world, tmp_path):
    """EffectDiagnostics is the DiagnosticsModel of the rig: its prior, then one predict step per cycle and one update per
    recorded cycle, then the closures that ended the last cycle, give the belief of its replay, and its other members
    answer within their ranges."""
    cycles, blocks = world
    episode = FixtureEpisode(tmp_path, cycles, blocks, START, seed=2)
    episode.read("tier1", cycles=2)
    episode.submit_diagnosis(dict(zip(("cooler", "valve", "pump", "accumulator"), START)))
    episode.create_work_order("accumulator")
    episode.advance(5)
    episode.read("tier3")
    released = json.loads(episode.released_path.read_text(encoding="utf-8"))
    view, models = profile_view(released["rules"]), episode_models(episode.start)
    model = EffectDiagnostics(models)
    assert isinstance(model, DiagnosticsModel) and model.calibrated and model.hypotheses(view) == list(view.combinations)
    parsed = fl.parse_released(released)
    observations = fl.merge_observations(parsed.readings, models.space)
    belief = model.prior(view, parsed)
    for observation in (o for o in observations if o.t < 0):
        belief = model.update(belief, observation)
    for t in range(parsed.time):
        belief = model.predict(belief, t, parsed.closures)
        for observation in (o for o in observations if o.t == t):
            belief = model.update(belief, observation)
    belief = belief.settle(parsed.closures)
    for observation in (o for o in observations if o.t == parsed.time):
        belief = model.update(belief, observation)
    stepped = belief.posterior()
    replayed = model.replay(view, parsed).posterior()
    assert np.allclose(stepped.probabilities, replayed.probabilities, rtol=1e-12, atol=1e-15) and np.allclose(stepped.mean, replayed.mean)
    assert set(model.tests(view)) == set(view.sensor_groups)
    now = model.replay(view, parsed)
    ee = model.expected_elimination(now, "tier3", 64, 0)
    assert ee <= now.posterior().effective_hypotheses() - 1 + 1e-9  # every belief after a reading keeps at least one effective hypothesis
    assert 0 <= model.information_gain(now, "tier3", 64, 0) <= math.log(now.next_reading().posterior().effective_hypotheses()) + 1e-9
    after = model.after_repair(now, ["valve"], view).posterior()
    assert after.current_marginals()["valve"] <= now.posterior().current_marginals()["valve"] + 1e-12
    assert [s.component for s in model.repair_plan((100, 80, 1, 130), view)] == ["valve", "pump"]


def test_a_backend_without_the_information_gain_is_refused_by_name():
    """The member information_gain is required of DiagnosticsModel beside expected_elimination, since the read
    selection by information gain needs the gain from every backend, so a backend that lacks it is refused where the server binds it,
    with the missing member named, and the rig's backend passes."""
    members = {name: (lambda self, *args: None) for name in PROTOCOL_MEMBERS if name not in ("calibrated", "information_gain")}
    lacking = type("WithoutGain", (), {**members, "calibrated": True})()
    assert "information_gain" in PROTOCOL_MEMBERS and "expected_elimination" in PROTOCOL_MEMBERS
    assert not isinstance(lacking, DiagnosticsModel)
    with pytest.raises(TypeError, match="WithoutGain is not a DiagnosticsModel: it lacks information_gain$"):
        diagnostics_backend(lacking)
    rig = EffectDiagnostics(models=None)
    assert diagnostics_backend(rig) is rig and isinstance(rig, DiagnosticsModel)


def counted_observations(monkeypatch):
    """[(filter, observation numbers)] of every update step the filters take while the test runs."""
    calls, original = [], fl.Filter.observe

    def observe(self, observation):
        calls.append((id(self), observation.numbers))
        return original(self, observation)

    monkeypatch.setattr(fl.Filter, "observe", observe)
    return calls


def test_a_read_of_zero_duration_enters_the_belief_on_the_priced_row(tmp_path, monkeypatch):
    """On the rows whose reads take no time, the priced row among them, a read stamps the cycle at which the clock stands,
    so the replay must observe the readings of the current cycle: one read of five cycles of tier3, then a ranking, and
    every filter of the ranking observes each released cycle once."""
    config = _composed("priced")
    assert config.durations["read_per_cycle"] == 0
    state_path = tmp_path / "episode.json"
    rig.start(state_path, 7, config, default_facts_path())
    rig.hand_over(state_path)
    rig.call(state_path, "read_sensors", {"sensor_group": "tier3", "cycles": 5}, {"type": "read_sensors", "sensor_group": "tier3", "cycles": 5})
    parsed = fl.parse_released(json.loads(rig.released_path(state_path).read_text(encoding="utf-8")))
    models = episode_models(rig.replay(rig.load_state(state_path)).initial)
    released = sorted(o.numbers for o in fl.merge_observations(parsed.readings, models.space))
    assert released and all(r.t == parsed.time for r in parsed.readings)
    calls = counted_observations(monkeypatch)
    CausalTools(models, rig.released_path(state_path)).rank_root_causes(1)
    per_filter = {}
    for filter_id, numbers in calls:
        per_filter.setdefault(filter_id, []).append(numbers)
    assert per_filter and all(sorted(seen) == released for seen in per_filter.values())


def test_a_prehistory_reading_at_a_negative_cycle_enters_once_before_the_first_cycle(monkeypatch):
    """A reading recorded before the episode began, as the AssetOpsBench rows release, carries a negative cycle: it updates
    the prior once, before the predict step of the first cycle, and moves the belief toward the combination it was drawn
    at."""
    models, view = episode_models(START), profile_view()
    space = models.space
    truth = space.combinations.index(START)
    values = {f: float(models.effect.mu_table[truth, space.field_index[f]]) for f in space.groups["tier1"]}
    calls = counted_observations(monkeypatch)
    belief = fl.Filter(view, models.effect, fl.Episode(2, [fl.Reading(0, -3, "tier1", values)], [], None))
    assert [numbers for _filter, numbers in calls] == [(0,)] and belief.time == 2
    assert belief.posterior().probabilities[truth] > 1 / len(space.combinations)


# ----- the order in which the rig served the evidence -----

COMPONENT_ORDER = ("cooler", "valve", "pump", "accumulator")
VALVE_START = (100, 80, 0, 130)
EVIDENCE = ["obs/1/PS1_Pressure_bar_100Hz_mean"]


def _composed(name):
    """A preset's configuration, the composition of its big layers as the drivers load it."""
    return layers.compose(json.loads(preset_path(name).read_text(encoding="utf-8")))


def preset_config(name, start, **replaced):
    """A preset's configuration with a fixed start at block offset 0 and the given keys replaced."""
    data = {**_composed(name).to_json(), **replaced}
    return EpisodeConfig.from_json({**data, "start": {"combination": list(start), "entry_offset": 0}})


class RigEpisode:
    """An episode on the action server in which every call thinks a given number of cycles of the episode's clock."""

    def __init__(self, folder, config, seed=7):
        self.path = Path(folder) / "episode.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rig.start(self.path, seed, config, default_facts_path())
        # the task is handed over at the readings of the start, from which every call below stamps its own
        state = rig.load_state(self.path)
        rig.hand_over(self.path, wall=state["last_call_wall_time"], mono=state["last_call_monotonic_time"])

    @property
    def released_path(self):
        return rig.released_path(self.path)

    def act(self, action, think=0):
        state = rig.load_state(self.path)
        seconds = EpisodeConfig.from_json(state["config"]).think_seconds_per_cycle
        tool, arguments = tool_call(action)
        # the thinking clock charges monotonic time, so the call stamps the wall and the monotonic clock alike: a millisecond
        # past the whole cycles of thinking, so that rounding never drops a cycle while the seconds the server carries from call
        # to call stay far below one cycle
        later = think * seconds + 0.001
        rig.call(self.path, tool, arguments, action, now=state["last_call_wall_time"] + later, monotonic=state["last_call_monotonic_time"] + later)

    def read(self, group, cycles=1, think=0):
        self.act({"type": "read_sensors", "sensor_group": group, "cycles": cycles}, think)

    def repair(self, diagnosis, *components, prediction_id=""):
        self.act({"type": "submit_diagnosis", "diagnosis": dict(zip(COMPONENT_ORDER, diagnosis)), "evidence_ids": EVIDENCE})
        for component in components:
            self.act({"type": "create_work_order", "component": component, "prediction_id": prediction_id, "evidence_ids": EVIDENCE})

    def released(self):
        return json.loads(self.released_path.read_text(encoding="utf-8"))

    def filter(self):
        released = self.released()
        return EffectDiagnostics(episode_models(self.start)).replay(profile_view(released["rules"]), fl.parse_released(released))

    def hidden(self):
        return tuple(rig.replay(rig.load_state(self.path)).hidden)

    @property
    def start(self):
        """The combination the episode started in, which the driver names at launch."""
        return tuple(rig.replay(rig.load_state(self.path)).initial)


def mass_of(post, combination):
    return float(post.probabilities[post.candidates.index(combination)])


def test_a_repair_settled_on_a_row_without_a_clock_enters_between_its_readings(tmp_path):
    """On the priced row no cycle runs, so a work order closes inside its call at the cycle where the clock stands.
    The readings released before it show the faulty valve and the readings after it the healthy one,
    and the replay applies the closure between them, as the rig served them, so every hypothesis left holds the repaired valve
    and the repaired unit comes first."""
    episode = RigEpisode(tmp_path, preset_config("priced", VALVE_START))
    episode.read("tier3", cycles=3)
    episode.repair(VALVE_START, "valve")
    episode.read("tier3", cycles=3)
    healthy = (100, 100, 0, 130)
    assert episode.hidden() == healthy
    flt = episode.filter()
    post = flt.posterior()
    assert post.current_marginals()["valve"] == 0 and post.candidates[post.order()[0]] == healthy
    (closure,) = flt.episode.closures
    assert closure.t == 0 and closure.after == 3 and all(r.standing and r.t == 0 for r in flt.episode.readings)


def test_readings_a_read_of_zero_duration_served_precede_the_kernels_of_their_cycle(tmp_path):
    """On the real-time clock row a read takes no time, so the three readings of cycle 0 were served while the clock stood
    at 0, before the thinking time ran cycle 0, whose kernels closed the valve order due at 0. The replay observes them
    before those kernels: its belief equals the three readings stepped through cycles 0 and 1 with the closure after them,
    and differs from the same steps with the closure before them."""
    episode = RigEpisode(tmp_path, preset_config("real_time", VALVE_START))
    for _ in range(3):
        episode.read("tier3")
    episode.repair(VALVE_START, "valve")
    episode.act({"type": "get_current_time"}, think=2)
    flt = episode.filter()
    (closure,) = flt.episode.closures
    assert flt.episode.time == 2 and closure.t == 0 and [r.t for r in flt.episode.readings] == [0, 0, 0]
    observations = fl.merge_observations(flt.episode.readings, flt.space)

    def stepped(observed_first):
        steps = fl.Filter(flt.view, flt.effect, fl.Episode(0, [], [], None))
        for observation in observations if observed_first else ():
            steps.observe(observation)
        steps.predict(0, [closure])
        for observation in () if observed_first else observations:
            steps.observe(observation)
        return steps.predict(1).settle().posterior().probabilities

    replayed = flt.posterior().probabilities
    assert np.allclose(replayed, stepped(True), rtol=1e-10, atol=1e-12)
    assert not np.allclose(replayed, stepped(False), rtol=1e-6, atol=1e-9)


def test_the_next_reading_meets_the_belief_now_where_a_read_takes_no_time(tmp_path):
    """A read of zero duration is served at once, so E8, E9 and E21 draw the next reading from the belief now
    and not from the belief after the kernels of the current cycle, which here would close the valve order due now. Where
    a read takes a cycle, as on the advanced real-time clock row, the next reading is recorded after those kernels."""
    episode = RigEpisode(tmp_path / "real_time", preset_config("real_time", VALVE_START))
    episode.read("tier3")
    episode.read("tier2")
    episode.repair(VALVE_START, "valve")
    ranking = CausalTools(episode_models(episode.start), episode.released_path).rank_root_causes(3)
    assert ranking.elimination and all(abs(row.effective_before - ranking.effective_hypotheses) < 1e-9 for row in ranking.elimination)
    flt = episode.filter()
    assert np.array_equal(flt.next_reading().posterior().probabilities, flt.posterior().probabilities)
    assert not np.allclose(flt.ahead(1).posterior().probabilities, flt.posterior().probabilities)
    timed = RigEpisode(tmp_path / "advanced_real_time", preset_config("advanced_real_time", VALVE_START))
    timed.read("tier3")
    flt = timed.filter()
    assert np.array_equal(flt.next_reading().posterior().probabilities, flt.ahead(1).posterior().probabilities)


def test_the_visit_closes_in_priority_order_before_the_next_reading_on_a_row_without_a_clock(tmp_path):
    """On a row without a clock whose repair card is on, the orders of a visit stay open until the next call that is not
    an order, and the visit closes them in priority order, so the valve closes before the pump that
    depends on it although the pump order came first. The predict step closes the open orders in that order, and the next
    reading, served after the visit settles, meets the repaired unit."""
    repair_rules, priced = (_composed(name).to_json() for name in ("repair_rules", "priced"))
    start, repaired = (100, 80, 1, 130), (100, 100, 0, 130)
    episode = RigEpisode(tmp_path, preset_config("priced", start, dependencies=repair_rules["dependencies"], priority=repair_rules["priority"],
                                                  prices={**priced["prices"], "visit": repair_rules["prices"]["visit"]}))
    episode.read("tier3", cycles=2)
    episode.repair(start, "pump", "valve")
    flt = episode.filter()
    assert [c.component for c in flt.episode.pending] == ["pump", "valve"]
    # the pump's repair takes only after the valve's, so in the other order no mass would reach the repaired unit
    for post in (flt.ahead(1).posterior(), flt.next_reading().posterior()):
        assert post.candidates[post.order()[0]] == repaired and mass_of(post, repaired) > 0.5
    episode.act({"type": "get_current_time"})
    assert episode.hidden() == repaired


def test_verify_outcome_reads_the_first_reading_served_after_the_closure(tmp_path):
    """On the real-time clock row the valve order due at cycle 0 closes when the thinking time runs cycle 0, so a read of
    zero duration made at cycle 0 after the order was created still shows the faulty valve. verify_outcome takes the first
    reading served after the closure, not the first one stamped at or after the closure's cycle."""
    episode = RigEpisode(tmp_path, preset_config("real_time", VALVE_START))
    episode.read("tier3")
    tools = CausalTools(episode_models(episode.start), episode.released_path)
    prediction = tools.estimate_intervention_effect("replace:valve").prediction_id
    episode.repair(VALVE_START, "valve", prediction_id=prediction)
    episode.read("tier3")
    episode.act({"type": "get_current_time"}, think=1)
    episode.read("tier3")
    assert [(o["n"], o["t"]) for o in episode.released()["observations"]] == [(1, 0), (2, 0), (3, 1)]
    verification = tools.verify_outcome(prediction)
    assert (verification.observation, verification.observed_t) == (3, 1)


def test_the_moments_of_a_combination_do_not_move_with_its_mass_down_to_the_smallest_a_float_holds():
    """A combination whose states entered their blocks under two labels carries the mixture of its states, each by its share
    of the combination's own mass. A filter at cycle 0 with states set by hand: the two states of the combination hold
    different offsets and entry labels whose covariances differ, and their log weights fall by one depth at a time against a
    second combination, from a mass a float holds exactly to one far below the smallest normal float. At every mass the mean
    and the covariance of the combination are the moments of the shares, the covariance is symmetric and has no negative
    eigenvalue to rounding, and the predictive of every sensor group factors."""
    view, effect = profile_view(), episode_models(START).effect
    space, index = effect.space, effect.space.combo_index
    other, combination = view.healthy_combination(), START
    flt = fl.Filter(view, effect, fl.Episode(0, [], [], None), resampled=False, prior=None, support=[other, combination], replay=False)
    states = flt.states
    rows = flt.rules.combo_index(states.pos)
    assert len(states) == 2 and sorted(rows.tolist()) == sorted([index[other], index[combination]])
    first, kept = int(np.nonzero(rows == index[combination])[0][0]), int(np.nonzero(rows == index[other])[0][0])
    states.extend(states.take(np.array([first])))
    second = len(states) - 1
    states.entry[second] = 1
    rng, d = np.random.default_rng(0), len(space.fields)
    spread = rng.normal(size=(d, d))
    flt.covariance[1] = flt.covariance[0] + (spread @ spread.T / d)[None]
    offsets = rng.normal(size=(2, d))
    states.payload["offset"][0, first], states.payload["offset"][0, second] = offsets
    shares = np.array([0.3, 0.7])
    mean = shares @ offsets
    h = index[combination]
    covariance = (shares[0] * flt.covariance[0][h] + shares[1] * flt.covariance[1][h]
                  + sum(s * np.outer(o - mean, o - mean) for s, o in zip(shares, offsets)))
    masses = []
    for depth in (0.0, 300.0, 700.0, 712.0, 725.0, 735.0, 740.0):
        states.logw[0, kept] = 0.0
        states.logw[0, first], states.logw[0, second] = np.log(shares) - depth
        post = flt.posterior()
        masses.append(post.probabilities[h])
        np.testing.assert_allclose(post.mean[h], mean, rtol=1e-9, atol=1e-12)
        got = post.covariance[h]
        np.testing.assert_allclose(got, covariance, rtol=1e-9, atol=1e-12)
        scale = np.abs(got).max()
        assert np.abs(got - got.T).max() <= 1e-12 * scale and np.linalg.eigvalsh((got + got.T) / 2).min() >= -1e-12 * scale
        for fields in space.groups.values():
            post.predictive(fields)
    assert masses[0] == pytest.approx(0.5) and 0 < masses[-1] < np.finfo(float).tiny


def test_every_attribute_of_the_rule_cards_is_read_in_a_module_of_the_package():
    """Every attribute that Rules sets on itself is read somewhere in the modules of the server's package, so the rule cards
    carry no parameter that no step of the filter uses."""
    init = next(node for node in ast.walk(ast.parse((ROOT / "CausalDiagnosis" / "mcp_servers" / "Causal_Diagnosis_MCP" / "belief" / "belief_filter" / "rule_cards.py").read_text(encoding="utf-8")))
                if isinstance(node, ast.ClassDef) and node.name == "Rules")
    init = next(node for node in init.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assigned = {node.attr for node in ast.walk(init) if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
                and isinstance(node.value, ast.Name) and node.value.id == "self"}
    read = {node.attr for path in PACKAGE.rglob("*.py") if "tests" not in path.relative_to(PACKAGE).parts
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)}
    assert "hold" in assigned and not assigned - read, f"Rules sets attributes that no module of the package reads: {sorted(assigned - read)}"


def test_a_backend_without_the_replay_is_refused_by_name():
    """The adapter replays the released evidence through the backend at every call, so replay is a member of
    DiagnosticsModel, a backend that lacks it is refused where the server binds it with the member named, and the backend of
    the server passes."""
    members = {name: (lambda self, *args: None) for name in PROTOCOL_MEMBERS if name not in ("calibrated", "replay")}
    lacking = type("WithoutReplay", (), {**members, "calibrated": True})()
    assert "replay" in PROTOCOL_MEMBERS and not isinstance(lacking, DiagnosticsModel)
    with pytest.raises(TypeError, match="WithoutReplay is not a DiagnosticsModel: it lacks replay$"):
        diagnostics_backend(lacking)
    backend = EffectDiagnostics(models=None)
    assert diagnostics_backend(backend) is backend and isinstance(backend, DiagnosticsModel)


def test_a_reading_of_a_recorded_cycle_meets_the_belief_now(tmp_path):
    """The released evidence of an episode whose valve order closes in the kernels of the current cycle, read under rules
    with a read duration of one cycle: on a row that reads a record, whose released tool table lists the read of recorded
    cycles, the next reading meets the belief now, since such a read moves no clock; on a row that plays a recording per
    read the same rules give the belief after the kernels of the current cycle, which differs from the belief now."""
    episode = RigEpisode(tmp_path, preset_config("real_time", VALVE_START))
    episode.read("tier3")
    episode.read("tier2")
    episode.repair(VALVE_START, "valve")
    released = episode.released()
    rules = dict(released["rules"], durations={**released["rules"]["durations"], "read_per_cycle": 1})
    record = profile_view(dict(rules, tools=[{"name": "history", "server": "iot"}]))
    per_read = profile_view(rules)
    assert record.reads_record and not per_read.reads_record
    model = EffectDiagnostics(episode_models(episode.start))
    on_record, played = (model.replay(view, fl.parse_released(released)) for view in (record, per_read))
    assert np.array_equal(on_record.next_reading().posterior().probabilities, on_record.posterior().probabilities)
    assert not np.allclose(on_record.ahead(1).posterior().probabilities, on_record.posterior().probabilities)
    assert np.array_equal(played.next_reading().posterior().probabilities, played.ahead(1).posterior().probabilities)


def test_the_readings_of_e4_are_those_of_the_block_of_the_first_combination():
    """States set by hand at cycle 0 over two combinations: the heaviest state belongs to a combination whose mass is not
    the largest, while the other combination holds the largest mass over two states. E4 reads the block of the first
    combination, so the entry of the posterior is that of the heaviest state of the first combination, and the latest values
    are the readings after that entry."""
    view, effect = profile_view(), episode_models(START).effect
    space, index = effect.space, effect.space.combo_index
    alone, split = view.healthy_combination(), START
    group = min(view.sensor_groups, key=lambda g: (len(view.group_fields(g)), g))
    fields = view.group_fields(group)
    readings = [fl.Reading(n, n - 4, group, {f: float(10 * n + i) for i, f in enumerate(fields)}) for n in (1, 2, 3)]
    flt = fl.Filter(view, effect, fl.Episode(0, readings, [], None), resampled=False, prior=None, support=[alone, split], replay=False)
    states = flt.states
    rows = flt.rules.combo_index(states.pos)
    assert len(states) == 2 and len(flt.observations) == 3
    first, kept = int(np.nonzero(rows == index[split])[0][0]), int(np.nonzero(rows == index[alone])[0][0])
    states.extend(states.take(np.array([first])))
    second = len(states) - 1
    flt.covariance[1], flt.covariance[2] = flt.covariance[0].copy(), flt.covariance[0].copy()
    states.entry[kept], states.entry[first], states.entry[second] = 0, 1, 2
    states.logw[0, kept], states.logw[0, first], states.logw[0, second] = np.log([0.4, 0.35, 0.25])
    post = flt.posterior()
    assert int(np.argmax(flt.weights()[0])) == kept and post.candidates[post.order()[0]] == split
    assert post.entry_of_top == 1
    expected = {f: float(np.mean([r.values[f] for r in readings[1:]])) for f in fields}
    assert post.latest_values() == pytest.approx(expected, rel=1e-12)


def test_a_reading_that_no_state_can_produce_is_refused_by_name():
    """A reading whose likelihood is zero under every state of a fit, here an infinite value that a port could release,
    leaves no weight to normalize, so the update names it and writes no nan into the belief."""
    models, view = episode_models(START), profile_view()
    values = {f: math.inf for f in models.space.groups["tier1"]}
    with pytest.raises(ValueError, match="zero likelihood under every state of a fit"):
        fl.Filter(view, models.effect, fl.Episode(1, [fl.Reading(1, 0, "tier1", values)], [], None))


def test_a_support_without_a_combination_of_the_asset_is_refused_by_name():
    """A support that holds no combination of the profile gives no state to start from with or without the fix history, so
    the filter names it where it would otherwise call itself without end."""
    models, view = episode_models(START), profile_view()
    outside = tuple(-1 for _ in view.components)
    with pytest.raises(ValueError, match="holds no combination of the asset"):
        fl.Filter(view, models.effect, fl.Episode(0, [], [], None), resampled=False, prior=None, support=[outside])


def test_a_schedule_card_that_draws_more_components_than_the_asset_has_is_refused_by_name():
    """A schedule card that asks for more components than the asset has leaves no subset to draw, so its draws are refused
    by name before the logarithm of their count."""
    view = view_with(schedule={"mode": "seeded", "components": len(profile_view().components) + 1, "first_drop": [5, 9], "interval": 10})
    rules = fl.Rules(view, episode_models(START).space)
    with pytest.raises(ValueError, match="the schedule card draws"):
        rules.schedule_draws(START)
