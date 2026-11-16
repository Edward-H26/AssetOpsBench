"""The replay check of a finished run (replay_check/): the episodes of a scripted run fed again through the servers of this
tree, a line of another server inside a call read in the order of service, the problems that a changed answer, a torn line, a
changed argument, another version of the evidence, another clock, a changed briefing and a recorded command that composes another
configuration give, the runs that a replay does not play, the certificate and its ledger event, the replays that count nowhere, a
run whose only differences are generated tables, which is certified with them named, and the certificate bound to the commit and
the platform of the run. A scripted run is the driver with the scripted plan, which calls every tool of its servers and needs no
model. The tests assert relations between the files of a run and of its replay, never a recorded value, and run folders go under
pytest's folders, outside the repository."""
import asyncio
import dataclasses
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.run_verification import replay_check
from CausalDiagnosis.evaluation.result_tables import pooling_rules
from CausalDiagnosis.experiments.run_verification import verify_results_folder as verify_results
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER, METERED_SERVERS, RIG_SERVER, ROOT
from CausalDiagnosis.experiments.episode_runners.run_configuration import DEFAULT_ASSET
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import HIDDEN, SCORE_COLUMNS, STATE_FILE, episode_dirs, episode_name, read_rows
from CausalDiagnosis.utils.json_files import read_json, write_json
from CausalDiagnosis.utils.csv_rows import write_rows
from CausalDiagnosis.experiments.run_verification.run_identity_file_lists import lists_of
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import misdiagnosis_charge_of
from CausalDiagnosis.mcp_servers.utils import transcript

# the rig's action server with the diagnostics server on a preset without a clock and on the preset whose clock charges the wall
# seconds between two calls, the AssetOpsBench servers behind the metering layer on the memory store, and a trial of the training
# path, which plays the preset without a clock on the development split
RUNS = {"rig": ["--config", "static"], "clock": ["--config", "real_time"], "aob": ["--config", "base", "--servers", "aob"],
        "trial": ["--config", "static", "--training-split", "development"]}
PRESETS = ("static", "real_time", "base")
TRIAL_PRESETS = ("static",)  # the presets that a run of RUNS plays widened to the development starts
EPISODE = "seed000000"
SHIFT = 600.0  # seconds added to every wall clock reading of the calls of a transcript, the start line left as it was
_MADE = {}  # form of RUNS: the folder of its scripted run, made once per session and only read afterward


@pytest.fixture(scope="session")
def scripted(tmp_path_factory, table_guard):
    """scripted(form) is the folder of the scripted run of that form of RUNS, made at its first use in the session after the label
    and grade tables of PRESETS are computed, and those of TRIAL_PRESETS on the development starts, as a grid computes them before
    its first line, so that no run extends a table that its replay then reads in another state. A test that edits a run edits a
    copy of it (_copy)."""
    if not Path(default_facts_path()).exists():
        pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH")
    if importlib.util.find_spec("mcp") is None:
        pytest.skip("needs the mcp package")

    def run_of(form):
        if form not in _MADE:
            from CausalDiagnosis.experiments.cluster_runs import precompute_grid_tables as precompute_tables
            from CausalDiagnosis.experiments.episode_runners import run_episodes

            if not _MADE:
                precompute_tables.main([arg for preset in PRESETS for arg in ("--config", preset)])
                precompute_tables.main(["--starts", "development", *(arg for preset in TRIAL_PRESETS for arg in ("--config", preset))])
            out = tmp_path_factory.mktemp(form)
            run_dir = Path(run_episodes.main([*RUNS[form], "--agent", "scripted", "--seeds", "1", "--out", str(out)]))
            rows = read_rows(run_dir)
            assert rows and all(row["error"] == "" for row in rows), rows
            moved = [event["files"] for event in ledger.load(run_dir)["events"] if event["what"] == pooling_rules.DATA_CHANGED]
            assert not moved, f"a table changed while the run played, so the run recorded it in another state: {moved}"
            _MADE[form] = run_dir
            table_guard.record()  # the run recorded the session copies of the tables, so no later test may change them
        return _MADE[form]

    return run_of


def _copy(run_dir, folder):
    """A copy of a run folder under folder, for a test that edits it."""
    return Path(shutil.copytree(run_dir, Path(folder) / Path(run_dir).name))


def _set_meta(run_dir, **fields):
    """run.json of a copied run with the fields set, None taking a key out."""
    meta = read_json(Path(run_dir) / "run.json")
    for key, value in fields.items():
        if value is None:
            meta.pop(key, None)
        else:
            meta[key] = value
    write_json(Path(run_dir) / "run.json", meta)
    return meta


COMMIT = "c" * 40  # a made-up commit for the tests that read the tree under test as a clone of the run's commit
TEST = "CausalDiagnosis/experiments/run_verification/tests/test_replay_check.py"


def _set_commit(run_dir, commit):
    """run.json of a copied run names commit as the commit it was played at, with the change in its ledger."""
    before = ledger.snapshot(run_dir)
    _set_meta(run_dir, code={**read_json(Path(run_dir) / "run.json")["code"], "commit": commit})
    ledger.record_changes(run_dir, before, TEST, "the copy names the commit it was played at")


def _played_at(run_dir, monkeypatch, commit=COMMIT, here=COMMIT):
    """The run names commit as its commit, and the tree under test reads as a clone at here (None: no repository of its own)."""
    _set_commit(run_dir, commit)
    monkeypatch.setattr(ledger.code_identity, "git_state", lambda paths, root=ROOT: (here, [] if here else None))


def _write_transcript(ep_dir, lines, renumber=True):
    """The transcript of an episode folder written as the servers write it, every line numbered by its place and the entry that a
    line names (enter_n) by the place of that entry, unless renumber is False, as in a torn or edited file."""
    if renumber:
        place_of = {line["n"]: place for place, line in enumerate(lines)}
        lines = [{**line, "n": place, **({"enter_n": place_of[line["enter_n"]]} if "enter_n" in line else {})} for place, line in enumerate(lines)]
    (Path(ep_dir) / transcript.NAME).write_text("".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines), encoding="utf-8")


def _set_command(run_dir, argv):
    """The ledger of a copied run with the command line argv in place of the one it recorded, None taking the command out."""
    path = Path(run_dir) / ledger.LEDGER
    record = read_json(path)
    if argv is None:
        record.pop("command", None)
    else:
        record["command"] = {**record.get("command", {}), "argv": list(argv)}
    write_json(path, record)


# ----- the replay through the servers -----

@pytest.mark.parametrize("form", ["rig", "aob"])
def test_a_scripted_run_replays_through_the_servers_on_both_surfaces(scripted, form):
    """Both surfaces replay with no problem and count nowhere else, every recorded line has a line of the scratch transcript, so a
    replay that compared nothing would fail here, and the run folder keeps its bytes."""
    run = scripted(form)
    before = ledger.snapshot(run)
    result = replay_check.replay_run(run)
    assert (result["state"], result["problems"], result["advisory"]) == ("replayed", [], [])
    recorded = sum(len(transcript.read(ep_dir)) for ep_dir in episode_dirs(run))
    assert result["lines"] == recorded > result["episodes"] >= 1
    assert result["listings_equal"] and result["data_equal"] and result["config_equal"] and result["passed"]
    assert ledger.snapshot(run) == before

def test_replay_reproduces_the_misdiagnosis_charge(scripted):
    """The rig's preset without a clock prices a misdiagnosis, so the hidden log of a scripted episode holds the charge of the
    environment exactly when the episode wrote a diagnosis that its verdict scores as missed, with the price of the rules and
    the field of the verdict that names it, and the replay through the servers reproduces the hidden log and the verdict, the
    charge included, with no problem."""
    run = scripted("rig")
    for ep_dir in episode_dirs(run):
        log = read_json(ep_dir / HIDDEN / "log.json")
        verdict = read_json(ep_dir / HIDDEN / "verdict.json")
        price = log[0]["config"]["prices"]["misdiagnosis"]
        wrote = any(e["event"] == "submit_diagnosis" for e in log)
        charged = [e["cost"] for e in log if e["event"] == "misdiagnosis"]
        assert price > 0 and charged == ([price] if wrote and not verdict["diagnosis_all_correct"] else []), ep_dir.name
        assert misdiagnosis_charge_of(verdict) == sum(charged), ep_dir.name
    assert replay_check.replay_run(run)["problems"] == []


def test_a_replay_starts_the_episode_again_and_names_a_changed_briefing(scripted, monkeypatch):
    """The replay starts the episode through the start function of the driver, so a briefing that gains a word changes the released
    evidence of the start line and of the end, which a replay of the stored actions alone cannot see."""
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

    run = scripted("rig")
    written = episode.write_released

    def with_a_word(state_path, env, compose=episode.briefing):
        return written(state_path, env, compose=lambda profile, config: compose(profile, config) + " Changed.")

    monkeypatch.setattr(episode, "write_released", with_a_word)
    problems = replay_check.replay_run(run)["problems"]
    assert f"{run.name}/{EPISODE}: line 0 differs in released_sha256" in problems
    assert f"{run.name}/{EPISODE}: the released evidence differs from released.json" in problems


def test_the_replay_feeds_the_handlers_that_the_mcp_package_calls_for_a_client(scripted, tmp_path):
    """For the action server, a FastMCP server, and for a metered server, a low level server, the object that server_object builds
    holds the request handlers of a listing and of a call, and a recorded call of a tool that steps the episode, fed through them
    after the handover, writes the lines of its service: its entry, the line of the answered call and its exit, both naming that
    entry."""
    import mcp.types as mcp_types
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.episode_start_and_replay import start as rows_start
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.server import EpisodeStore
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

    for form, store in (("rig", None), ("aob", {"kind": "memory"})):
        ep_dir = scripted(form) / EPISODE
        meta = read_json(ep_dir.parent / "run.json")
        state = read_json(ep_dir / STATE_FILE)
        line = next(line for line in transcript.read(ep_dir) if line["kind"] == "call" and line["stepped"]
                    and (line["server"] in METERED_SERVERS if store else line["server"] == RIG_SERVER))
        folder = tmp_path / form
        folder.mkdir()
        (rows_start if store else episode.start)(folder / STATE_FILE, state["seed"], EpisodeConfig.from_json(state["config"]), state["facts_path"])
        episode_store = EpisodeStore(folder / STATE_FILE, store) if store else None
        if episode_store:
            episode_store.create()
        try:
            episode.hand_over(folder / STATE_FILE)  # a counted call of an episode never handed over is refused by name
            app = replay_check.episode_replay.server_object(line["server"], folder, meta, (), store)
            assert {mcp_types.ListToolsRequest, mcp_types.CallToolRequest} <= set(app.request_handlers), form
            before = len(transcript.read(folder))
            asyncio.run(replay_check.feed(app, line))
            added = transcript.read(folder)[before:]
            assert [new["kind"] for new in added] == ["enter", "call", "exit"], form
            entry, answered, leaving = added
            assert (answered["server"], answered["tool"], answered["is_error"]) == (line["server"], line["tool"], False), form
            assert answered["enter_n"] == leaving["enter_n"] == entry["n"] and leaving["outcome"] == "answered", form
        finally:
            if episode_store:
                episode_store.drop()


def test_a_changed_answer_a_missing_line_and_a_changed_argument_are_each_named_by_line_and_field(scripted, tmp_path):
    """Three copies of a recorded transcript with one edit each give exactly one problem each, which names the line and the field and
    holds no answer text: a changed answer hash, a line taken out, and a changed argument of a call of the diagnostics server, whose
    answer the replay then gives for the argument the record names."""
    run = scripted("rig")
    lines = transcript.read(run / EPISODE)
    answer = next(place for place, line in enumerate(lines) if line["kind"] == "call")
    listing = next(place for place, line in enumerate(lines) if line["kind"] == "list" and place > answer)
    argument = next(place for place, line in enumerate(lines)
                    if line["kind"] == "call" and line["server"] == CAUSAL_SERVER and "sensor_group" in (line.get("arguments") or {}))
    group = lines[argument]["arguments"]["sensor_group"]
    label = f"{run.name}/{EPISODE}"

    def changed(place, **fields):
        return [{**line, **fields} if n == place else line for n, line in enumerate(lines)]

    cases = {
        "answer": (changed(answer, answer_sha256="0" * 64), False, f"{label}: line {answer} differs in answer_sha256"),
        "missing": (lines[:listing] + lines[listing + 1:], False, f"{label}: line {listing} of the transcript is torn or out of place"),
        "argument": (changed(argument, arguments={**lines[argument]["arguments"], "sensor_group": "tier1" if group != "tier1" else "tier3"}),
                     False, f"{label}: line {argument} differs in answer, answer_sha256"),
    }
    for name, (edited, renumber, expected) in cases.items():
        copy = _copy(run, tmp_path / name)
        _write_transcript(copy / EPISODE, edited, renumber)
        problems = replay_check.replay_run(copy)["problems"]
        assert problems == [expected], name
        assert not any(line["answer"] in problem for problem in problems for line in lines if "answer" in line), name


def test_the_recorded_clock_decides_the_thinking_charge_of_a_replay(scripted, tmp_path):
    """On the preset that charges the wall seconds between two calls the replay under the recorded readings gives the recorded
    thinking and the recorded wall and monotonic gaps, the first one from the start included. With every reading of the calls
    shifted, those of their entries and of the ends of their service, the problem names the hidden log, and with the monotonic
    reading of the start shifted, the state file alone."""
    run = scripted("clock")
    assert sum(read_json(run / EPISODE / STATE_FILE)["think_seconds"]) > 0  # the row charged the wall seconds of the run
    assert replay_check.replay_run(run)["problems"] == []
    shifted = _copy(run, tmp_path / "calls")
    lines = transcript.read(shifted / EPISODE)
    readings = ("wall", "mono", "end_wall", "end_mono")
    _write_transcript(shifted / EPISODE, [{**line, **{key: line[key] + SHIFT for key in readings if key in line}}
                                          if line["kind"] in ("enter", "call") and "wall" in line else line for line in lines])
    assert f"{shifted.name}/{EPISODE}: the hidden log differs from {HIDDEN}/log.json" in replay_check.replay_run(shifted)["problems"]
    started = _copy(run, tmp_path / "start")
    lines = transcript.read(started / EPISODE)
    _write_transcript(started / EPISODE, [{**lines[0], "mono": lines[0]["mono"] - SHIFT}, *lines[1:]])
    assert replay_check.replay_run(started)["problems"] == [f"{started.name}/{EPISODE}: the state file differs from episode.json"]


def test_a_call_of_the_diagnostics_server_is_fed_against_the_evidence_it_read(scripted, tmp_path):
    """A call of the diagnostics server whose line stands after the action call that it overlapped, past the exit line of that
    call, with the hash of the evidence before that call, is served that version and replays. A line that names a version the
    replay never produced is named."""
    run = _copy(scripted("rig"), tmp_path)
    lines = transcript.read(run / EPISODE)

    def diagnostic(place):
        return lines[place]["kind"] == "call" and lines[place]["server"] == CAUSAL_SERVER

    first = next(place for place in range(len(lines)) if diagnostic(place))
    a = next(place for place in range(first + 1, len(lines)) if lines[place]["kind"] == "enter")  # the entry of the action call
    k = max(place for place in range(first, a) if diagnostic(place))  # the call that stands last before the action call, so no two diagnostics calls trade places
    end = next(place for place in range(a + 1, len(lines)) if lines[place]["kind"] == "exit" and lines[place]["enter_n"] == a)
    _write_transcript(run / EPISODE, lines[:k] + lines[k + 1:end + 1] + [lines[k]] + lines[end + 1:])
    assert replay_check.replay_run(run)["problems"] == []
    moved = transcript.read(run / EPISODE)
    assert (moved[end]["tool"], moved[end]["released_sha256"]) == (lines[k]["tool"], lines[k]["released_sha256"])
    moved[end]["released_sha256"] = "f" * 64
    _write_transcript(run / EPISODE, moved)
    message = f"{run.name}/{EPISODE}: line {end} names a version of the released evidence that the replay never produced"
    assert message in replay_check.replay_run(run)["problems"]


def _moved(lines, k, after):
    """The lines of a transcript with the line at place k moved to stand right after the line at place after."""
    rest = [line for place, line in enumerate(lines) if place != k]
    at = rest.index(lines[after]) + 1
    return rest[:at] + [lines[k]] + rest[at:]


def test_a_line_of_another_server_inside_a_call_replays_in_the_order_of_service(scripted, tmp_path):
    """The hold is one lock per server process, so a line of the diagnostics server that read the evidence before an action call
    can stand between the entry and the exit of that call, after its entry or after its line, also in the last call, whose entry
    the state file names as the call in service. Each such record replays, read in the order of service. A changed answer inside
    an interleaved bracket, of the line inside it or of the call itself, is named by its recorded number, and two calls that
    trade their entries are named."""
    run = scripted("rig")
    lines = transcript.read(run / EPISODE)
    label = f"{run.name}/{EPISODE}"

    def diagnostic(place):
        return lines[place]["kind"] == "call" and lines[place]["server"] == CAUSAL_SERVER

    def named(entry, kind):
        return next(place for place, line in enumerate(lines) if line.get("enter_n") == entry and line["kind"] == kind)

    first = next(place for place in range(len(lines)) if diagnostic(place))
    a = next(place for place in range(first + 1, len(lines)) if lines[place]["kind"] == "enter")  # the entry of an action call
    k = max(place for place in range(first, a) if diagnostic(place))
    last = max(place for place in range(len(lines)) if lines[place]["kind"] == "enter")  # the entry of the call in service
    j = max(place for place in range(last) if diagnostic(place))
    cases = {"after_entry": _moved(lines, k, a), "after_call": _moved(lines, k, named(a, "call")), "in_service": _moved(lines, j, last)}
    for name, edited in cases.items():
        copy = _copy(run, tmp_path / name)
        _write_transcript(copy / EPISODE, edited)
        # a played run names the call in service by the number that its entry has in the file
        state = read_json(copy / EPISODE / STATE_FILE)
        place = {line["n"]: new for new, line in enumerate(edited)}
        write_json(copy / EPISODE / STATE_FILE, {**state, "open_service": {**state["open_service"], "n": place[state["open_service"]["n"]]}})
        assert replay_check.replay_run(copy)["problems"] == [], name
    for place in (a, a + 1):  # the moved line of the diagnostics server, then the line of the action call after it
        copy = _copy(run, tmp_path / f"changed_{place}")
        _write_transcript(copy / EPISODE, [{**line, "answer_sha256": "0" * 64} if n == place else line for n, line in enumerate(cases["after_entry"])])
        assert replay_check.replay_run(copy)["problems"] == [f"{label}: line {place} differs in answer_sha256"], place
    second = next(place for place in range(named(a, "exit"), len(lines)) if lines[place]["kind"] == "enter")
    traded = {named(a, "call"): second, named(second, "call"): a}
    copy = _copy(run, tmp_path / "traded")
    _write_transcript(copy / EPISODE, [{**line, "enter_n": traded[n]} if n in traded else line for n, line in enumerate(lines)])
    problems = replay_check.replay_run(copy)["problems"]
    assert any(problem.startswith(f"{label}: line {second} differs in") and "tool" in problem for problem in problems), problems


def _played_in_process(run, folder, cancel_at):
    """The episode of a scripted run of the AssetOpsBench rows played again in process through the server objects of this tree
    under the real clocks, with the stepped call number cancel_at (0 for the first) cancelled while the meter serves it, as the cut
    at the wall limit cancels an open call, then scored as the driver scores it. Returns the lines of its transcript."""
    from CausalDiagnosis.experiments.episode_runners import run_episodes
    from CausalDiagnosis.experiments.episode_runners.episode_launchers import DEFAULT_RANKING_MODEL
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import NO_CALL_ACTION
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.recorded_history_env import RecordedRigEnv
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import metering_layer as metering
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.episode_start_and_replay import start
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.server import EpisodeStore
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode
    from CausalDiagnosis.experiments.table_definitions.training_variants import asset_of

    meta, state, store = read_json(run / "run.json"), read_json(run / EPISODE / STATE_FILE), {"kind": "memory"}
    config = EpisodeConfig.from_json(state["config"])
    folder.mkdir(parents=True)
    start(folder / STATE_FILE, state["seed"], config, state["facts_path"])
    left_out = tuple(episode.replay(episode.load_state(folder / STATE_FILE), env_class=RecordedRigEnv).initial)
    episode_store = EpisodeStore(folder / STATE_FILE, store)
    episode_store.create()
    episode.hand_over(folder / STATE_FILE)

    async def cancelled(app, line):
        served = asyncio.Event()

        async def waits(self, *arguments):  # the service of the call goes on until the cut cancels it
            served.set()
            await asyncio.Event().wait()

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(metering.Meter, "_call", waits)
            task = asyncio.ensure_future(replay_check.feed(app, line))
            await served.wait()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def play():
        apps, stepped = {}, 0
        for line in transcript.read(run / EPISODE)[1:]:
            kind = line["kind"]
            if kind in ("enter", "exit", "handover"):
                continue
            if kind == "ready":
                transcript.append(folder, "ready", **{k: v for k, v in line.items() if k not in ("n", "kind", "pid")})
            elif line.get("tool") == NO_CALL_ACTION:
                await replay_check.feed_no_call(folder, store)
            else:
                key = (line["server"], line["pid"])
                if key not in apps:
                    apps[key] = replay_check.episode_replay.server_object(line["server"], folder, meta, left_out, store)
                counted = kind == "call" and line["stepped"]
                await (cancelled if counted and stepped == cancel_at else replay_check.feed)(apps[key], line)
                stepped += counted

    try:
        asyncio.run(play())
        episode_store.export(folder)
    finally:
        episode_store.drop()
    _facts, cycles, blocks = load_world(None, asset_of(meta))
    options = {"judge": None, "grade": meta.get("grade_lookup", True), "ranking_model": meta.get("ranking_model") or DEFAULT_RANKING_MODEL}
    run_episodes.verify_episode(folder, config, cycles, blocks, state["seed"], options, RecordedRigEnv)
    return transcript.read(folder)


def test_a_counted_call_cancelled_in_its_service_replays_with_its_entry_and_exit_in_their_places(scripted, tmp_path):
    """The cut at the wall limit cancels an open counted call of a metered server: the record then holds the call's entry and an
    exit line of the outcome cancelled, no call line, and the call saved nothing. Played so in process, the record replays, its
    two lines written back in their places. The record as the driver played it still replays, and an exit line whose outcome
    was changed is named."""
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
    from CausalDiagnosis.experiments.table_definitions.training_variants import asset_of

    run = scripted("aob")
    folder = tmp_path / "cancelled" / EPISODE
    lines = _played_in_process(run, folder, cancel_at=2)
    entry = next(line["n"] for line in lines if line["kind"] == "enter" and not any(other.get("enter_n") == line["n"] and other["kind"] == "call"
                                                                                for other in lines))
    leave = next(line["n"] for line in lines if line["kind"] == "exit" and line["enter_n"] == entry)
    assert lines[leave]["outcome"] == "cancelled"
    meta, store = read_json(run / "run.json"), {"kind": "memory"}
    world = load_world(None, asset_of(meta))
    assert replay_check.replay_episode(folder, meta, tmp_path / "replayed", store, world)["problems"] == []
    assert replay_check.replay_run(run)["problems"] == []
    _write_transcript(folder, [{**line, "outcome": "answered"} if line["n"] == leave else line for line in lines])
    assert replay_check.replay_episode(folder, meta, tmp_path / "changed", store, world)["problems"] == [
        f"cancelled/{EPISODE}: line {leave} differs in outcome"]


def test_a_loop_episode_and_a_sampled_episode_replay_like_an_episode_of_a_table(scripted, tmp_path):
    """A run in the mode of the online loop of GEPA with the episode folder of a metric call, and a run in the sampling mode of fine
    tuning, replay as an episode of a table does, since the replay reads the episode folder and never the mode."""
    from CausalDiagnosis.training import gepa_prompt_optimization as gepa_online
    from CausalDiagnosis.experiments.table_definitions import training_variants as training

    loop = _copy(scripted("rig"), tmp_path / "loop")
    _set_meta(loop, mode=gepa_online.MODE)
    # the command of a loop on the run's preset, which composes the configuration of its run.json as the loop composes it
    argv = read_json(loop / ledger.LEDGER)["command"]["argv"]
    preset = argv[argv.index("--config") + 1]
    _set_command(loop, ["--config", preset, "--model", "m", "--training-fold", "0", "--reflection-model", "r", "--reflection-base-url", "http://127.0.0.1:9"])
    rows = read_rows(loop)
    (loop / EPISODE).rename(loop / episode_name(rows[0]["seed"], call=3))
    write_rows(loop / "scores.csv", [{**rows[0], "gepa_call": "3"}], list(rows[0]))
    sampled = _copy(scripted("rig"), tmp_path / "sampled")
    _set_meta(sampled, mode=training.TRAINING_MODE)
    for run in (loop, sampled):
        assert verify_results.check_replay(run) == ("1 episodes replayed", []), run.name


def _on_the_registered_split(argv):
    """A recorded command line without its --training-split and the value after it, a run of the registered split."""
    at = argv.index("--training-split")
    return argv[:at] + argv[at + 2:]


def _widened_keys(run_dir, argv):
    """The keys of the configuration of run.json that the preset of the command line argv composes otherwise, those that the
    widening to the development starts set."""
    from CausalDiagnosis.experiments.episode_runners import run_episodes

    plain = run_episodes.configure(run_episodes.build_parser().parse_args(argv))[1].to_json()
    played = read_json(Path(run_dir) / "run.json")["config"]
    return sorted(key for key in set(plain) | set(played) if plain.get(key) != played.get(key))


def test_a_trial_on_the_development_split_replays_with_the_configuration_it_played(scripted, tmp_path):
    """A run of --training-split development plays the preset widened to the development starts (run_episodes.played_config), and
    the replay composes its recorded command through the same function, so the trial replays with no problem. Its command on the
    registered split composes the preset itself, and the keys that the widening set are named."""
    run = scripted("trial")
    argv = read_json(run / ledger.LEDGER)["command"]["argv"]
    keys = _widened_keys(run, _on_the_registered_split(argv))
    assert keys, "the trial played the preset as it is, so the test would compare nothing"
    result = replay_check.replay_run(run)
    assert (result["problems"], result["config_equal"]) == ([], True)
    registered = _copy(run, tmp_path)
    _set_command(registered, _on_the_registered_split(argv))
    assert replay_check.replay_run(registered)["problems"] == [
        f"{registered.name}: the configuration that the command of the run composes in this tree differs from run.json in {', '.join(keys)}"]


def test_a_gepa_loop_of_a_trial_replays_with_the_configuration_it_played(scripted, tmp_path):
    """A loop of GEPA on the development split composes the settings of the driver and plays what run_episodes.played_config gives
    for them, the preset widened to the development starts, and the replay composes its command the same way, so the loop replays
    with no problem. The same loop on the registered split composes the preset itself, and the keys that the widening set are
    named."""
    from CausalDiagnosis.training import gepa_prompt_optimization as gepa_online

    trial = scripted("trial")
    argv = read_json(trial / ledger.LEDGER)["command"]["argv"]
    keys = _widened_keys(trial, _on_the_registered_split(argv))
    loop = _copy(trial, tmp_path)
    _set_meta(loop, mode=gepa_online.MODE)
    command = ["--config", argv[argv.index("--config") + 1], "--model", "m", "--training-fold", "0", "--reflection-model", "r",
               "--reflection-base-url", "http://127.0.0.1:9"]
    rows = read_rows(loop)
    (loop / EPISODE).rename(loop / episode_name(rows[0]["seed"], call=3))
    write_rows(loop / "scores.csv", [{**rows[0], "gepa_call": "3"}], list(rows[0]))
    _set_command(loop, [*command, "--training-split", "development"])
    assert replay_check.replay_run(loop)["problems"] == []
    _set_command(loop, command)
    assert replay_check.replay_run(loop)["problems"] == [
        f"{loop.name}: the configuration that the command of the run composes in this tree differs from run.json in {', '.join(keys)}"]


def test_a_recorded_command_that_composes_another_configuration_here_is_named_by_its_keys(scripted, tmp_path, monkeypatch):
    """The command that the ledger of a run recorded is composed in this tree and compared with the configuration of run.json: a
    preset that composes another horizon here is named by that key alone and never by a value, a command that composes nothing is
    named with the reason, and a ledger without a command is named, each as the one problem of a run whose episodes replay."""
    from CausalDiagnosis.experiments.episode_runners import run_episodes

    run = scripted("rig")
    composes = run_episodes.command_line.load_config

    def longer(spec, *args, **kwargs):
        name, config = composes(spec, *args, **kwargs)
        return name, dataclasses.replace(config, horizon=config.horizon + 1)

    with monkeypatch.context() as patched:
        patched.setattr(run_episodes.command_line, "load_config", longer)
        result = replay_check.replay_run(run)
    assert result["problems"] == [f"{run.name}: the configuration that the command of the run composes in this tree differs from run.json in horizon"]
    assert result["config_equal"] is False
    unknown = _copy(run, tmp_path / "unknown")
    argv = read_json(unknown / ledger.LEDGER)["command"]["argv"]
    _set_command(unknown, [arg if place != argv.index("--config") + 1 else "no_such_preset" for place, arg in enumerate(argv)])
    problems = replay_check.replay_run(unknown)["problems"]
    assert len(problems) == 1 and problems[0].startswith(f"{unknown.name}: the command of the run composes no configuration in this tree ("), problems
    none = _copy(run, tmp_path / "none")
    _set_command(none, None)
    assert replay_check.replay_run(none)["problems"] == [f"{none.name}: the ledger of the run records no command, so the replay cannot compose its configuration"]


def test_a_run_without_a_transcript_is_named_and_holds_no_certificate(scripted, tmp_path):
    run = _copy(scripted("rig"), tmp_path)
    (run / EPISODE / transcript.NAME).unlink()
    assert replay_check.replay_run(run)["state"] == "no transcript"
    assert verify_results.check_replay(run) == ("not replayed, no transcript", [])
    status, _result, path = replay_check.certify(run)
    assert (status, path) == (1, None) and not list(run.glob("replay_*.json"))


# ----- the certificate -----

COPY_SKIPPED = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
COPY_SKIPPED_PATHS = {"CausalDiagnosis/results/runs", "CausalDiagnosis/results"}


# prints, as JSON, where the code of the tree it runs in reads each data file of the data record of a run
DATA_PROBE = ("import json, sys; from CausalDiagnosis.experiments.run_verification import run_ledger as ledger; "
              "from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path; "
              "print(json.dumps({name: entry['path'] for name, entry in ledger.data_files(str(default_facts_path())).items()}))")


def _tree_env(tree):
    """The environment of a process that runs the code of another tree."""
    return {**os.environ, "PYTHONPATH": os.pathsep.join([str(tree), str(tree / "src")]), "PYTHONDONTWRITEBYTECODE": "1"}


def _tree_copy(folder, path, old, new):
    """A copy of the tree under test without its environment, caches, runs and results, in which the one place of the text old in
    the file at path holds new. The session reads the generated tables as copies in pytest's folder (the repository conftest), so
    every data file that the copy's own code reads at another path than this session gets the bytes this session reads, and the copy
    differs from the tree under test in the one changed text alone."""
    def ignore(directory, names):
        relative = Path(directory).relative_to(ROOT).as_posix()
        return {name for name in names if name in COPY_SKIPPED or (name if relative == "." else f"{relative}/{name}") in COPY_SKIPPED_PATHS}

    tree = Path(shutil.copytree(ROOT, Path(folder) / "tree", ignore=ignore, symlinks=True))
    text = (tree / path).read_text(encoding="utf-8")
    assert text.count(old) == 1, f"{path} holds {old!r} {text.count(old)} times"
    (tree / path).write_text(text.replace(old, new), encoding="utf-8")
    probe = subprocess.run([sys.executable, "-c", DATA_PROBE], cwd=tree, env=_tree_env(tree), capture_output=True, text=True, timeout=300)
    assert probe.returncode == 0, probe.stdout + probe.stderr
    ours = {name: entry["path"] for name, entry in ledger.data_files(str(default_facts_path())).items()}
    for name, target in json.loads(probe.stdout.splitlines()[-1]).items():
        if Path(ours[name]).is_file() and Path(ours[name]).resolve() != Path(target).resolve():
            shutil.copyfile(ours[name], target)
    return tree


def _certify_in(tree, run_dir):
    """The certificate command of verify_results_folder.py run by the code of another tree on a run folder, the tree committed as a
    repository of its own and the run naming that commit as its own."""
    git = ["git", "-c", "user.name=test", "-c", "user.email=test@example.org"]
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "the tree under test"]):
        subprocess.run([*git, *args], cwd=tree, check=True, capture_output=True, timeout=300)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tree, check=True, capture_output=True, text=True).stdout.strip()
    _set_commit(run_dir, head)
    command = [sys.executable, "-m", "CausalDiagnosis.experiments.run_verification.verify_results_folder", "--certify", str(run_dir)]
    done = subprocess.run(command, cwd=tree, env=_tree_env(tree), capture_output=True, text=True, timeout=900)
    done.head = head
    return done


def test_the_tree_copy_that_differs_in_a_comment_certifies_and_the_copy_that_differs_in_a_tool_text_does_not(scripted, tmp_path):
    """The run of the tree under test is certified by a copy of the tree with one more comment line in a server module, whose
    certificate names that module alone among the changed files, and refused by a copy with one changed word in a tool
    description, which names the listing of that server and writes no certificate."""
    module = "CausalDiagnosis/mcp_servers/Environment_Actions_MCP/main.py"
    run = scripted("rig")
    commented = _tree_copy(tmp_path / "commented", module, "\ndef build_server(", "\n# a comment line that changes no behavior\ndef build_server(")
    reworded = _tree_copy(tmp_path / "reworded", module, "of what was found and done.", "of what was seen and done.")
    passed = _copy(run, tmp_path / "runs_commented")
    done = _certify_in(commented, passed)
    assert done.returncode == 0, done.stdout + done.stderr
    [certificate] = passed.glob("replay_*.json")
    assert read_json(certificate)["changed"] == {name: [module] for name in lists_of(module)}
    assert certificate.name == replay_check.certificate_name(done.head) and read_json(certificate)["checked_commit"] == done.head
    refused = _copy(run, tmp_path / "runs_reworded")
    done = _certify_in(reworded, refused)
    assert done.returncode == 1, done.stdout + done.stderr
    assert f"{refused.name}: the tool listing of the server {RIG_SERVER} differs from run.json" in done.stdout.splitlines()
    assert not list(refused.glob("replay_*.json"))


CERTIFICATE_KEYS = {"format", "run", "run_identity", "run_commit", "checked_identity", "checked_commit", "agent",
                    "agent_files_equal", "changed", "loaded_changed", "store", "platform", "data_equal", "data_differences", "listings_equal",
                    "scores_equal", "scoring_problems", "table_problems", "lines", "episodes", "transcripts", "date", "serving_card"}


def test_the_certificate_enters_the_ledger_and_the_ledger_check_accepts_it(scripted, tmp_path, monkeypatch):
    """certify writes the certificate with one event of the run's ledger, so the ledger check of verify_results_folder.py reports nothing,
    while a file dropped beside it without an event is reported. The certificate holds hashes and names and no value of a result."""
    run = _copy(scripted("rig"), tmp_path)
    _played_at(run, monkeypatch)
    assert verify_results.check_ledger(run) == ("checked", [])
    status, _result, path = replay_check.certify(run)
    assert status == 0 and path.parent == run
    event = ledger.load(run)["events"][-1]
    assert event["tool"] == replay_check.TOOL and event["files"] == {path.name: {"old": None, "new": ledger.sha256_of(path)}}
    certificate = read_json(path)
    assert set(certificate) == CERTIFICATE_KEYS and certificate["episodes"] == [EPISODE]
    assert certificate["serving_card"] == {"agent": None, "judge": None}  # the scripted check reaches no model endpoint
    assert (certificate["checked_commit"], certificate["scores_equal"], certificate["scoring_problems"]) == (COMMIT, True, [])
    assert (certificate["table_problems"], certificate["data_equal"], certificate["data_differences"]) == ([], True, [])
    assert certificate["transcripts"] == {EPISODE: transcript.file_sha256(run / EPISODE / transcript.NAME)}
    assert path.name == replay_check.certificate_name(COMMIT) and certificate["changed"] == {}
    assert verify_results.check_ledger(run) == ("checked", [])
    (run / "notes.txt").write_text("dropped beside the certificate\n", encoding="utf-8")
    assert verify_results.check_ledger(run) == ("checked", [f"{run.name}/notes.txt: added without a recorded event"])


def test_the_certificate_names_the_gpu_card_that_the_run_recorded(scripted, tmp_path, monkeypatch):
    """The certificate names the GPU card of the model and the judge endpoints as run.json of the run records them, here a made-up
    card that a copy of the scripted run names, with the change of run.json in its ledger."""
    run = _copy(scripted("rig"), tmp_path)
    card = {"gpu_name": "MADE_UP_GPU", "gpu_mib": "1024"}
    before = ledger.snapshot(run)
    _set_meta(run, serving={"agent": {**card, "gpu_uuid": "GPU-0", "sampler": {"path": None, "sha256": None, "bytes": None}}, "judge": None})
    ledger.record_changes(run, before, TEST, "the copy names a GPU card")
    _played_at(run, monkeypatch)
    status, _result, path = replay_check.certify(run)
    assert status == 0 and read_json(path)["serving_card"] == {"agent": card, "judge": None}


def test_a_replay_on_another_store_kind_or_another_platform_is_advisory_and_writes_no_certificate(scripted, tmp_path, monkeypatch):
    """A run whose record names CouchDB, replayed on the memory store where no CouchDB is named, and a run whose platform record
    differs from this machine's, both pass with an advisory reason and the exit status of a replay that counts nowhere."""
    monkeypatch.delenv("COUCHDB_URL", raising=False)
    couch = _copy(scripted("aob"), tmp_path / "couch")
    _set_meta(couch, backend={"kind": "couchdb"})
    elsewhere = _copy(scripted("rig"), tmp_path / "elsewhere")
    _set_meta(elsewhere, platform={**read_json(elsewhere / "run.json")["platform"], "python": "0.0.0"})
    for run, reason in ((couch, "the run played on the store couchdb and the replay on the memory store"),
                        (elsewhere, "the platform here differs from the run's in python")):
        status, result, path = replay_check.certify(run)
        assert (status, result["problems"], result["advisory"], path) == (replay_check.ADVISORY, [], [reason], None), run.name
        assert verify_results.main(["--certify", str(run)]) == replay_check.ADVISORY
        assert not list(run.glob("replay_*.json"))


@pytest.mark.skipif(not os.environ.get("COUCHDB_URL"), reason="needs a CouchDB that COUCHDB_URL names")
def test_the_memory_store_and_couchdb_answer_a_scripted_episode_alike(scripted, tmp_path):
    """A scripted episode played on CouchDB replays on the memory store with equal answer hashes, and one played on the memory store
    replays on CouchDB."""
    from CausalDiagnosis.experiments.episode_runners import run_episodes
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world

    couch_store = run_episodes.store_settings(SimpleNamespace(servers="aob", store="couchdb"))
    couch_run = Path(run_episodes.main([*RUNS["aob"], "--store", "couchdb", "--agent", "scripted", "--seeds", "1", "--out", str(tmp_path / "couch")]))
    for run, store in ((couch_run, {"kind": "memory"}), (scripted("aob"), couch_store)):
        meta = read_json(run / "run.json")
        world = load_world(None, meta.get("asset_id") or DEFAULT_ASSET)
        replayed = replay_check.replay_episode(run / EPISODE, meta, tmp_path / f"replay_{store['kind']}", store, world)
        assert replayed["problems"] == [] and replayed["lines"] == len(transcript.read(run / EPISODE)), store["kind"]


def test_a_run_whose_only_differences_are_generated_tables_is_certified_with_them_named(scripted, tmp_path, monkeypatch):
    """After a fix the tree computes a generated table again, so its hash differs from the one a run recorded although no answer
    changes. certify writes the certificate of such a run and names the table, and the replay check of verify_results_folder.py counts
    the run identical and names the table, while a run whose score cell differs as well gets no certificate. A recording that
    differs writes no certificate. The table is the first generated entry of the run's data record, so no table of one asset is
    named in the test."""
    root = tmp_path / "runs"
    tables_run = _copy(scripted("rig"), root / "tables")
    cells_run = _copy(scripted("rig"), root / "cells")
    for run in (tables_run, cells_run):
        _played_at(run, monkeypatch)
    rows = read_rows(cells_run)
    column = next(c for c in SCORE_COLUMNS if c not in replay_check.JUDGE_CELLS)
    write_rows(cells_run / "scores.csv", [{**rows[0], column: "differs"}, *rows[1:]], list(rows[0]))
    table = next(name for name, digest in read_json(tables_run / "run.json")["data"].items() if name not in replay_check.RECORDED_DATA and digest)
    data_record = ledger.run_data.data_record

    def computed_again(*names):
        return lambda facts_path, asset_id=None: {**data_record(facts_path, asset_id), **{name: "0" * 64 for name in names}}

    with monkeypatch.context() as patched:
        patched.setattr(ledger.run_data, "data_record", computed_again(table))
        status, result, certificate = replay_check.certify(tables_run)
        refused, refused_result, none = replay_check.certify(cells_run)
        checked = verify_results.check_replay(tables_run), verify_results.check_replay(cells_run)
    problem = f"{tables_run.name}: the generated table {table} differs from the one the run recorded"
    assert (status, result["problems"], result["table_problems"], result["answers_equal"]) == (0, [problem], [problem], True)
    written = read_json(certificate)
    assert (written["data_equal"], written["data_differences"], written["scores_equal"]) == (False, [table], True)
    assert (refused, none) == (1, None) and f"{cells_run.name}/{EPISODE}: the cells {column} differ from scores.csv" in refused_result["problems"]
    computed = f"; generated tables computed again with every answer and score cell equal: {table}"
    assert checked == ((f"{result['episodes']} episodes replayed{computed}", []), (f"{result['episodes']} episodes replayed", refused_result["problems"]))
    recording = _copy(scripted("rig"), tmp_path / "recording")
    with monkeypatch.context() as patched:
        patched.setattr(ledger.run_data, "data_record", computed_again("facts"))
        status, result, written_path = replay_check.certify(recording)
    assert (status, written_path, result["answers_equal"]) == (1, None, False)
    assert f"{recording.name}: the data file facts differs from the one the run recorded" in result["problems"]
    assert not list(recording.glob("replay_*.json"))


def test_certify_writes_only_on_a_clone_of_the_run_s_commit(scripted, tmp_path, monkeypatch):
    """certify refuses a tree at another commit than the run's and a tree that is no repository of its own, names both, and writes
    no certificate; on a clone of the run's commit it writes the certificate named by that commit."""
    for name, here, reason in (("elsewhere", "d" * 40, f"stands at commit {'d' * 12}"), ("upload", None, "is no repository of its own")):
        run = _copy(scripted("rig"), tmp_path / name)
        _played_at(run, monkeypatch, here=here)
        status, result, path = replay_check.certify(run)
        assert (status, path) == (1, None) and not list(run.glob("replay_*.json")), name
        assert any(reason in problem and COMMIT[:12] in problem for problem in result["problems"]), result["problems"]
    run = _copy(scripted("rig"), tmp_path / "clone")
    _played_at(run, monkeypatch)
    status, _result, path = replay_check.certify(run)
    assert status == 0 and path.name == replay_check.certificate_name(COMMIT)


def test_certified_holds_the_certificate_to_the_commit_and_the_platform_of_the_run(scripted, tmp_path, monkeypatch):
    """certified() accepts a run whose certificate was written at its commit on its platform, and names the reason for a run.json
    that names another commit and for one that records another platform."""
    run = _copy(scripted("rig"), tmp_path)
    _played_at(run, monkeypatch)
    assert replay_check.certify(run)[0] == 0
    meta = read_json(run / "run.json")
    assert replay_check.certified(run, meta) is None
    other = {**meta, "code": {**meta["code"], "commit": "e" * 40}}
    assert replay_check.certified(run, other) == f"its folder holds no replay certificate of its commit {'e' * 12}"
    moved = {**meta, "platform": {**meta["platform"], "python": "0.0.0"}}
    assert replay_check.certified(run, moved) == "its replay certificate was written on another platform than its run.json records"


def test_a_no_call_line_is_replayed_through_the_entry_the_driver_bound_and_builds_no_server(tmp_path, monkeypatch):
    """The no_call step is no listed tool, so its call line goes to the same entry the driver called (episode.call on the
    rig's rows), never to a server object, where it would be an unlisted tool."""
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

    sent = []
    monkeypatch.setattr(episode, "call", lambda state_path, tool, arguments, action: sent.append((Path(state_path), tool, action)))
    monkeypatch.setattr(replay_check.episode_replay, "server_object", lambda *a, **k: pytest.fail("a no_call line built a server object"))
    (tmp_path / "released.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(replay_check.episode_replay, "RELEASED_FILE", "released.json")
    line = {"n": 1, "kind": "call", "server": RIG_SERVER, "pid": 1, "tool": "no_call", "arguments": {}}
    problems = asyncio.run(replay_check._feed_lines([line], tmp_path, {}, (), None, "ep"))
    assert problems == [] and sent == [(tmp_path / STATE_FILE, "no_call", {"type": "no_call"})]
