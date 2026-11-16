"""A no_call step counts against the step limit alone: on a
configuration that prices nothing, three tool calls and the same three calls with two no_call steps between them give the same
cost on the calls unit and the same tool calls and refusals of the row, and five steps against three."""
import dataclasses

import pytest

from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_1_priced import cost_unit
from CausalDiagnosis.experiments.episode_runners.run_episodes import tool_counts
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

PROFILE, NO_CALL = {"type": "get_asset_profile"}, {"type": "no_call"}


def play(folder, actions):
    folder.mkdir()
    path = folder / "episode.json"
    base = EpisodeConfig()
    prices = {**base.prices, "read": {g: 0 for g in base.prices["read"]}, "parts": {c: 0 for c in base.prices["parts"]},
              "repair_charge": 0, "visit": 0}
    config = dataclasses.replace(base, prices=prices, budget=None)
    with episode.recorded_clock(995.0, 995.0):
        episode.start(path, 3, config, default_facts_path())
    episode.hand_over(path, wall=1000.0, mono=1000.0)
    for k, action in enumerate(actions):
        e = 1010.0 + 10 * k
        with episode.recorded_clock(e, e, stamps=[(e, e, 0.0), (e + 1, e + 1, 0.5)]):
            episode.call(path, action["type"], {}, action)
    state = episode.load_state(path)
    env = episode.replay(state)
    costs = verifier._costs(env.log, {}, None, unit=cost_unit(config.prices))
    return costs, env.steps, tool_counts(state)


def test_a_no_call_enters_the_step_limit_and_no_count_of_tool_calls(tmp_path):
    if not default_facts_path().exists():
        pytest.skip("hydraulic rig facts file not present")
    plain = play(tmp_path / "plain", [PROFILE] * 3)
    idle = play(tmp_path / "idle", [PROFILE, NO_CALL, PROFILE, NO_CALL, PROFILE])
    assert plain[0]["cost_unit"] == idle[0]["cost_unit"] == "calls"
    assert plain[0]["cost"] == idle[0]["cost"] == 3
    assert (plain[1], idle[1]) == (3, 5)
    assert plain[2] == idle[2] == {"tool_calls": 3, "refusals": 0}
