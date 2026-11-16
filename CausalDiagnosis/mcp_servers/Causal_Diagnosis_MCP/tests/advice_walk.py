"""A walk that follows the advice of rank_root_causes with no model, a development check and no row of any
table, policy or grid line. walk() plays one episode in process, and the module run from a tree with HYD_FACTS_PATH set,

    python -m CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.advice_walk [preset]

walks every development combination at every entry offset on the preset (base by default) and prints counts to the
console: episodes, declarations, reads before the diagnosis, and how often the pass differs from "rank 1 was the start"."""
import json
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

from CausalDiagnosis.datasets.dataset_loader.evidence_source import BLOCK_CAP
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.datasets.splits.development_split import development_starts
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import build_blocks, default_facts_path, load_cycles
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.gymnasium_wrapper import oracle_cost
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.tests.harness import preset_config
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import CausalTools
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.fixture_episode import episode_models
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as server

MAX_RANKINGS = 20  # rankings of one walk at most, far above the one read and the one diagnosis that the advice asks for


@lru_cache(maxsize=1)
def world():
    cycles = load_cycles(default_facts_path())
    return cycles, build_blocks(cycles)


def development_combinations():
    """The development starts of the default asset, in grid order."""
    return development_starts(get_profile(DEFAULT_ASSET), sorted(world()[1]))


def started(preset, start, entry):
    """The preset with the start fixed at the entry offset and the fault rule of the starts widened to admit it."""
    config = preset_config(preset)
    faults = sum(1 for c, level in zip(config.components, start) if level != config.healthy[c])
    return preset_config(preset, fault_rule={"min_faulty": min(1, faults), "max_faulty": len(config.components)},
                         start={"combination": list(start), "entry_offset": entry})


def rank(env, folder, start, top_k=3):
    """The compact reply of rank_root_causes on what the environment has released so far, with the fits of an episode of the
    start; its advice is the advice of every form."""
    state = Path(folder) / "state.json"
    server.write_released(state, env)
    return CausalTools(episode_models(tuple(start)), server.released_path(state)).rank_root_causes(top_k, "compact")


def read(env, group, cycles, shown):
    """One history read of the newest recorded cycles that no reading of the group has shown, as the surface of the agent asks
    for them. shown holds the cycles read so far per group and is updated."""
    fresh = [t for t in sorted(env.recorded_cycles(), reverse=True) if t not in shown[group]][:cycles]
    if not fresh:
        return None
    obs, _cost, _done, _info = env.step({"type": "read_recorded", "tool": "history", "sensor_group": group,
                                         "fields": sorted(env._group_fields(group)), "start": min(fresh), "end": max(fresh) + 1,
                                         "offset": 0, "limit": 1000})
    for g, sensors in env.config.sensor_groups.items():
        if set(sensors) <= set(env.config.sensor_groups[group]):
            shown[g].update(fresh)
    return obs


def walk(start, entry=0, preset="base", seed=0):
    """One episode on the engine of the AssetOpsBench rows, in process: at each turn it ranks and does what the advice says,
    the read with its cycles, or the diagnosis of the advised rank with one work order per faulty component of that
    candidate, citing the plan's prediction, and the finish. Returns (log, verdict, replies), the verdict of verify."""
    cycles, blocks = world()
    config = started(preset, start, entry)
    env = RecordedRigEnv(cycles, blocks, config)
    env.reset(seed)
    shown = {g: set() for g in config.sensor_groups}
    replies = []
    with tempfile.TemporaryDirectory() as folder:
        for _ in range(MAX_RANKINGS):
            reply = rank(env, folder, start)
            replies.append(reply)
            if reply.advice is None or reply.advice.action != "read" or read(env, reply.advice.sensor_group, reply.advice.cycles, shown) is None:
                break
        advice = replies[-1].advice
        if advice is not None and advice.action == "diagnose":
            candidate, branch = replies[-1].candidates[advice.rank - 1], replies[-1].branches[advice.rank - 1]
            env.step({"type": "submit_diagnosis", "diagnosis": dict(candidate.combination), "evidence_ids": []})
            for step in branch.repair_plan.steps:
                env.step({"type": "create_work_order", "component": step.component, "prediction_id": branch.repair_plan.plan_prediction_id or "",
                          "evidence_ids": []})
        if not env.done:
            env.step({"type": "finish", "summary": "The advice of the diagnostics server was followed."})
        stored = Path(folder) / "predictions.json"
        predictions = json.loads(stored.read_text(encoding="utf-8")) if stored.exists() else None
        verdict = verifier.verify(env.log, verifier.released_from_env(env), env.oracle(), oracle_cost(RecordedRigEnv(cycles, blocks, config), seed),
                                  grade=None, scope=config.verifier_scope, predictions=predictions)
    return env.log, verdict, replies


def main(argv):
    preset = argv[0] if argv else "base"
    episodes = declarations = reads = differ = 0
    for start in development_combinations():
        for entry in range(BLOCK_CAP):
            log, verdict, replies = walk(start, entry, preset)
            episodes += 1
            declarations += sum(1 for r in replies if r.advice is not None and r.advice.action == "declare")
            reads += sum(1 for e in log if e["event"] == "read")
            first = replies[-1].candidates[0].combination
            differ += verdict["pass"] != (tuple(first[c] for c in started(preset, start, entry).components) == tuple(start))
    print(f"{preset}: {episodes} episodes, {declarations} declarations advised, {reads} reads before the diagnosis, "
          f"{differ} episodes whose pass differs from rank 1 being the start")
    return 1 if declarations or differ else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
