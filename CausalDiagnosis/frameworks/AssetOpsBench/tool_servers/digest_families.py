"""The digest families of the tool surface of AssetOpsBench (the declaration's DIGEST_FAMILIES, entry ROWS), which the digest
program of the core (CausalDiagnosis/experiments/run_verification/invariant_digests.py) runs beside its own: one episode of the
AssetOpsBench rows per digest preset and seed played in process through the metering layer on the memory store (the keys
scripted.rows.*), the texts an agent of the rows reads before its first call (texts.rows.*), and the texts of the
CausalDiagnosis agent on this surface (texts.agent.*.aob). The functions below moved here unchanged from the digest program.
"""
import asyncio
import dataclasses
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from CausalDiagnosis.experiments.run_verification.invariant_digests import PLAN_LENGTH, SCRIPTED_SEEDS, VOLATILE_KEYS, digest, preset, presets

ROWS_FILES = ("episode.json", "hidden/log.json", "released.json", "aob_documents.json", "replies")
MEMORY_STORE = {"kind": "memory"}  # the store of the AssetOpsBench servers that run_episodes --store memory names
CLOCK_ORIGIN = 1_000_000.0  # the wall clock and the monotonic clock at the start of an episode played in process, in seconds
WAIT_CYCLES = 5  # the timeout of the wait that the episode of the rows adds, that of the wait of the rig's wiring plan


TOKEN_CLOCK_MODE = "generated_tokens"  # the clock mode of a preset that charges the tokens the harness of the agent reports


def aob_agent_texts(question, profile, envelope, facts, detail):
    """The texts of the CausalDiagnosis agent on the surface of the AssetOpsBench rows (the keys texts.agent.*.aob), which the
    keys texts.agent.* do not see, since they are built on the action server of the asset: the system prompt of the surface as
    the driver builds it for a row with a store, before the profile is read and after the opening of an episode on every preset
    of aob_presets, and on each such preset the orientation, the planning turn and the summary on the board of texts.agent.*,
    and the tool arguments the surface writes for every kind it binds, which reach the judge through the trajectory."""
    from CausalDiagnosis.experiments.episode_runners import run_episodes
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import Board
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import orientation_turn, planning_turn, summary_turn, system_prompt

    def prompts_of(surface):
        return [system_prompt(PLAN_LENGTH, surface, True), system_prompt(PLAN_LENGTH, surface, False)]

    systems = [prompts_of(run_episodes.causal_surface(preset("static").asset_id, MEMORY_STORE))]
    parts = {"orientation": {}, "planning": {}, "summary": {}, "bindings": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for name in aob_presets():
            config = preset(name)
            surface = run_episodes.causal_surface(config.asset_id, MEMORY_STORE)
            board = Board(evidence_form=surface.evidence_form, reads_take_time=surface.reads_take_time, noun=surface.noun, question=question)
            board.set_profile(profile)
            board.set_failure_modes(envelope["result"])
            board.update_status(envelope)
            aob_opening(config, facts, Path(tmp) / name, surface, board)
            systems.append(prompts_of(surface))
            parts["orientation"][name] = orientation_turn(question, profile, board.failure_modes, PLAN_LENGTH, surface=surface)
            parts["planning"][name] = planning_turn(board, PLAN_LENGTH, surface=surface)
            parts["summary"][name] = summary_turn(board, PLAN_LENGTH, surface=surface)
            parts["bindings"][name] = aob_bindings(config, surface, board)
    out = {"texts.agent.system.aob": digest(systems)}
    detail["texts.agent.system.aob"] = [digest(pair) for pair in systems]
    for part, values in parts.items():
        out[f"texts.agent.{part}.aob"] = digest([values[name] for name in values])
        detail[f"texts.agent.{part}.aob"] = {name: digest(value) for name, value in values.items()}
    return out


def aob_presets():
    """The digest presets of the keys texts.agent.*.aob: every one but those of the clock of generated tokens, whose episode
    charges the tokens the agent's harness reports, which an opening without an agent does not have. The surface reads the
    same kinds of answer on the twin preset of their clock."""
    return [name for name in presets() if preset(name).clock_mode != TOKEN_CLOCK_MODE]


def aob_opening(config, facts, folder, surface, board):
    """The opening of the agent on the AssetOpsBench rows of a preset, seed 0: the profile, then the opening kinds the surface
    names, each bound by the surface, called through the metering layer at the clocks of play_rows, parsed by the surface and
    put on the board, as the runner does. A kind whose tool the row does not list is left out, as play_rows leaves out such a
    call."""
    from CausalDiagnosis.experiments.episode_runners.run_folder_layout import STATE_FILE
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import episode_start_and_replay as rows
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.server import EpisodeStore
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / STATE_FILE
    rows.start(state_path, 0, config, str(facts))
    state = episode.load_state(state_path)
    state.update(last_call_wall_time=CLOCK_ORIGIN, episode_started_at=CLOCK_ORIGIN, last_call_monotonic_time=CLOCK_ORIGIN)
    episode.save_state(state_path, state)
    episode.hand_over(state_path, wall=CLOCK_ORIGIN, mono=CLOCK_ORIGIN)  # the play is in process, so it hands the task over itself
    EpisodeStore(state_path, MEMORY_STORE).create()
    calls = 0

    def play(kind):
        nonlocal calls
        server, tool, arguments = surface.bind(kind, board)
        at = CLOCK_ORIGIN + (calls + 1) * config.think_seconds_per_cycle
        meter = EpisodeStore(state_path, MEMORY_STORE).meter(wall_clock=lambda: at, monotonic_clock=lambda: at)
        if tool not in meter.tools_of(server):
            return
        reply = asyncio.run(meter.call(server, tool, arguments))
        calls += 1
        board.update_status(surface.parse(kind, tool, reply.text, board))

    play("profile")  # the opening kinds depend on the clock the profile names
    for kind in surface.opening_kinds(board):
        play(kind)
    EpisodeStore(state_path, MEMORY_STORE).drop()


def aob_bindings(config, surface, board):
    """(kind, server, tool, arguments) of every kind the surface binds after the opening, on fixed fields: a read of every
    sensor group, an order on every component, the diagnosis of the healthy levels, a declaration, a wait, the finish, the
    opening reads and the three diagnostics calls."""
    evidence = ("obs/2000-01-01T00:00:00/x", "fmea/source/code")
    groups = sorted(surface.sensor_groups)
    calls = [("profile", {}), ("time", {}), ("extent", {}), ("fix_history", {}),
             *(("failure_modes", {"component": component}) for component in surface.components),
             *(("read", {"sensor_group": group, "cycles": 3}) for group in groups),
             ("diagnose", {"levels": {component: config.healthy[component] for component in config.components}, "evidence": evidence}),
             *(("declare", {"sensor_group": group, "cycles": 2, "evidence": evidence}) for group in groups[:1]),
             *(("order", {"component": component, "prediction_id": "p1", "evidence": evidence}) for component in surface.components),
             ("wait", {"timeout": 3}), ("finish", {"summary": "the summary"}), ("changes", {"since": 0}),
             ("rank", {"top_k": 3}), ("estimate", {"action": "replace the cooler"}), ("verify", {"prediction_id": "p1"})]
    return [[kind, surface.bind(kind, board, **fields)] for kind, fields in calls]


def rows_plan(config, hidden):
    """The one scripted plan of the AssetOpsBench rows as (server, tool, arguments): the driver's plan
    (scripted_wiring_check.ScriptedBackend over the metered servers, so without the diagnostics server) with the checks of the rows
    family beside it. The site listing comes before the asset listing, the diagnosis names the hidden start, and in place
    of the driver's one work order stands one per faulty component, on the child asset of the component. hidden is the
    start, one level per component in the order of config.components, and a component is faulty when its level is off its
    healthy level. A wait of the task server and a second read of the first history read come before the finish."""
    from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import METERED_SERVERS, TASK_SERVER
    from CausalDiagnosis.frameworks.AssetOpsBench.agents import scripted_wiring_check
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract

    steps = [(server, tool, json.loads(arguments)) for _task, server, tool, arguments in scripted_wiring_check.ScriptedBackend(METERED_SERVERS).steps]
    *body, finish = steps
    levels = dict(zip(config.components, hidden))
    faulty = [component for component in config.components if levels[component] != config.healthy[component]]
    plan = []
    for server, tool, arguments in body:
        if tool == "assets":
            plan.append((server, "sites", {}))
        if tool == "submit_diagnosis":
            arguments = {**levels, "evidence_ids": arguments.get("evidence_ids", "")}
        if tool == "generate_work_order":
            plan += [(server, tool, {**arguments, "description": f"replace the {component}",
                                     "asset_num": layer_contract.child_asset(config.asset_id, component)}) for component in faulty]
            continue
        plan.append((server, tool, arguments))
    again = [step for step in plan if step[1] == "history"][:1]
    return [*plan, (TASK_SERVER, "wait_for_notification", {"timeout": WAIT_CYCLES}), *again, finish]


def play_rows(config, seed, facts, folder):
    """One episode of the AssetOpsBench rows, played in process in an empty folder: the start, the databases of the episode
    on the memory store, the calls of rows_plan through the metering layer and the export of the documents. The tool gives
    every clock: CLOCK_ORIGIN at the start and one think_seconds_per_cycle of the preset more at every call, and in the
    generated_tokens mode the ledger of the folder gains the tokens of one think_seconds_per_cycle at seconds_per_token before
    every call, so on a preset that charges thinking time each call is charged one cycle. A call whose tool the row does not list
    is left out. Returns the values of ROWS_FILES: the state file without VOLATILE_KEYS, the log of the engine of the rows
    replayed from it, the released evidence, the exported documents and the replies."""
    from CausalDiagnosis.experiments.episode_runners.run_folder_layout import RELEASED_FILE, STATE_FILE
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import token_clock
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import layer_2_real_time as real_time
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import episode_start_and_replay as rows
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.server import EpisodeStore
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / STATE_FILE
    rows.start(state_path, seed, config, str(facts))
    state = episode.load_state(state_path)
    state.update(last_call_wall_time=CLOCK_ORIGIN, episode_started_at=CLOCK_ORIGIN, last_call_monotonic_time=CLOCK_ORIGIN)
    episode.save_state(state_path, state)
    episode.hand_over(state_path, wall=CLOCK_ORIGIN, mono=CLOCK_ORIGIN)  # the play is in process, so it hands the task over itself
    EpisodeStore(state_path, MEMORY_STORE).create()
    per_call = 0
    if config.clock_mode == real_time.TOKEN_MODE:
        token_clock.open_ledger(folder)
        per_call = round(config.think_seconds_per_cycle / config.seconds_per_token)
    replies = []
    hidden = rows.replay(state).initial
    for server, tool, arguments in rows_plan(config, hidden):
        at = CLOCK_ORIGIN + (len(replies) + 1) * config.think_seconds_per_cycle
        meter = EpisodeStore(state_path, MEMORY_STORE).meter(wall_clock=lambda at=at: at, monotonic_clock=lambda at=at: at)
        if tool not in meter.tools_of(server):
            continue
        token_clock.add(folder, per_call)
        reply = asyncio.run(meter.call(server, tool, arguments))
        replies.append([server, tool, reply.text, reply.is_error])
    store = EpisodeStore(state_path, MEMORY_STORE)
    documents = json.loads(store.export(folder).read_text(encoding="utf-8"))
    store.drop()
    state = episode.load_state(state_path)
    log = rows.replay(state).log
    for key in VOLATILE_KEYS["episode.json"]:
        state.pop(key, None)
    released = json.loads((folder / RELEASED_FILE).read_text(encoding="utf-8"))
    return dict(zip(ROWS_FILES, (state, log, released, documents, replies)))


def rows_digests(facts):
    """The family scripted.rows: play_rows on every digest preset under the seeds of the scripted family."""
    from CausalDiagnosis.experiments.episode_runners.run_folder_layout import episode_name

    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in presets():
            config = preset(name)
            for seed in range(SCRIPTED_SEEDS):
                values = play_rows(config, seed, facts, Path(tmp) / name / episode_name(seed))
                out.update({f"scripted.rows.{name}.{episode_name(seed)}.{rel}": digest(value) for rel, value in values.items()})
    return out


def rows_texts(world, name, config, facts):
    """The texts an agent of the AssetOpsBench rows reads on a preset before its first call: the briefing at every level and
    the profile view of the metering layer, both on the public profile of the engine of the rows, and the tool definitions
    of the four metered servers on an episode of the preset."""
    from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import METERED_SERVERS
    from CausalDiagnosis.experiments.episode_runners.run_folder_layout import STATE_FILE
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import aob_tool_contract as layer_contract, episode_start_and_replay as rows
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.aob_briefing import briefing
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.metering_layer import meter_config
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.server import EPOCH, EpisodeServer
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.cycle_timestamps import Timeline

    cycles, blocks = world
    env = RecordedRigEnv(cycles, blocks, config)
    env.reset(0)
    profile = env.public_profile()
    out = {f"texts.rows.briefing.{name}.{level}": digest(briefing(profile, dataclasses.replace(config, guidance_level=level)))
           for level in (1, 2, 3)}
    view = meter_config(layer_contract.row_of(config), config, Timeline(EPOCH, config.seconds_per_cycle)).profile_view(profile)
    out[f"texts.rows.profile.{name}"] = digest(view)
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / STATE_FILE
        rows.start(state_path, 0, config, str(facts))
        for server in METERED_SERVERS:
            tools = asyncio.run(EpisodeServer(state_path, server, MEMORY_STORE).list_tools())
            out[f"texts.rows.tools.{name}.{server}"] = digest([tool.model_dump(mode="json") for tool in tools])
    return out


# the family as the digest program asks for it: scripted(facts), texts(world, name, config, facts) for a preset, agent_texts(question,
# profile, envelope, facts, detail) on the board of texts.agent.*
ROWS = SimpleNamespace(scripted=rows_digests, texts=rows_texts, agent_texts=aob_agent_texts)
