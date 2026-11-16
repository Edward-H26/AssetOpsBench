import asyncio
import importlib
import re
from pathlib import Path
from CausalDiagnosis.experiments.episode_runners.episode_launchers import DEFAULT_ABSTENTION_RULE, DEFAULT_RANKING_MODEL  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import HIDDEN, PREDICTIONS_FILE, RELEASED_FILE, SCORE_COLUMNS, STATE_FILE  # noqa: E402
from CausalDiagnosis.utils.json_files import read_json  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.training_variants import asset_of  # noqa: E402  the asset of a run as the tables read it
from . import run_record_checks


# the files an episode leaves that a replay writes again, compared as parsed JSON: the path in the episode folder, what the
# file is, and the keys that a replay cannot reproduce (the data path of the state file, the byte offset of the transcript and the
# processes named by their ids, since the replayed lines carry other process ids, and the judge)
LEFT_FILES = ((STATE_FILE, "state file", ("facts_path", "transcript_cursor", "refused_processes")),
              (RELEASED_FILE, "released evidence", ()), (PREDICTIONS_FILE, "stored predictions", ()),
              (f"{HIDDEN}/log.json", "hidden log", ()), (f"{HIDDEN}/oracle.json", "oracle record", ()),
              (f"{HIDDEN}/verdict.json", "verdict", ()), (f"{HIDDEN}/score.json", "score", ("judge", "judge_pass")))
# what a fix of the verifier may change: the files that the verifier writes from what it reads, with the score cells. It reads the
# transcript, the other left files and the inputs that the verdict repeats, which rescore_under_verifier_scope.py takes from the run when it scores
# the run again, so a replay where one of those differs shows more than other scores.
SCORE_FILES = (f"{HIDDEN}/verdict.json", f"{HIDDEN}/score.json")
VERDICT_INPUTS = ("grade", "cost_oracle", "scope")


# ----- one episode -----
def abstention_of(meta):
    """The abstention rule of the diagnostics server with its cut as the driver passes it, None for the default rule."""
    rule = meta.get("abstention_rule")
    return None if rule in (None, DEFAULT_ABSTENTION_RULE) else (rule, meta.get("abstention_cut"))


def server_object(server, folder, meta, left_out, store):
    """The low level MCP server of one server of an episode folder, built by build_server of the module that the driver's launcher
    of that server names, on the file that the launcher binds, with the options that the launcher passes (common.launch_options),
    the models file of the run's asset among them, as the driver names it for the diagnostics server."""
    from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER
    from CausalDiagnosis.experiments.episode_runners.episode_launchers import LAUNCHER_MODULES, STATE_FILES, launch_options
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import fit_models as fit
    from CausalDiagnosis.mcp_servers.utils.transcript_hooks import lowlevel_of

    module = importlib.import_module(LAUNCHER_MODULES[server])
    models_path = fit.models_path(asset_of(meta)) if server == CAUSAL_SERVER else None
    options = launch_options(server, ranking_model=meta.get("ranking_model") or DEFAULT_RANKING_MODEL, abstention=abstention_of(meta),
                             store=store, left_out=left_out, models_path=models_path)
    return lowlevel_of(module.build_server(Path(folder) / STATE_FILES[server], **options))


async def feed(app, line):
    """Hand one line of a transcript to the request handler that the MCP package calls for a client: a listing as a tools/list
    request, a call as a tools/call request with the recorded arguments. The server records what it answered."""
    import mcp.types as mcp_types

    if line["kind"] == "list":
        await app.request_handlers[mcp_types.ListToolsRequest](mcp_types.ListToolsRequest(method="tools/list"))
    else:
        params = mcp_types.CallToolRequestParams(name=line["tool"], arguments=line.get("arguments"))
        await app.request_handlers[mcp_types.CallToolRequest](mcp_types.CallToolRequest(method="tools/call", params=params))


async def feed_no_call(folder, store):
    """Replay the no_call step of an iteration that called no tool, which no server lists, through the entry the driver bound
    for it (CausalDiagnosis.agents.utils.idle_hook): the meter of the episode's store on the AssetOpsBench rows, episode.call on the rig's."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import NO_CALL_ACTION
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

    state_path = Path(folder) / STATE_FILE
    if store:
        from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import tool_surface
        from CausalDiagnosis.utils import frameworks
        surface = next(name for name, entry in frameworks.surfaces().items() if store["kind"] in entry["stores"])
        await tool_surface(surface).store(state_path, store).meter().no_call()
    else:
        episode.call(state_path, NO_CALL_ACTION, {}, {"type": NO_CALL_ACTION})


async def _feed_lines(lines, folder, meta, left_out, store, label, ledger=None):
    """Feed every line after the start, one server object per recorded server and process id, and return the problems of the
    evidence versions. Every version of the released evidence that the replay produced is kept, so a call of a server that only
    reads the evidence, whose line names the version it read by hash, is served that version and the current one is put back.
    With the bytes of the recorded failure ledger, each stepped call is preceded by the ledger up to the offset its line names,
    so its clock reads the failed attempts the recorded call read, a late record and one after the last call included. A counted
    call cancelled while it was served, as the cut at the wall limit cancels an open call, left its entry and an exit line and no
    call line, and saved nothing; no server writes those lines again, so the replay writes them back in their places, the entry as
    recorded, as it does for the ready lines, and the exit as a cancelled call writes it, so that a changed exit line is named."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import NO_CALL_ACTION
    from CausalDiagnosis.utils import model_failures
    from CausalDiagnosis.mcp_servers.utils import transcript
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

    released = Path(folder) / RELEASED_FILE
    versions, apps, problems = {}, {}, []

    def keep():
        data = released.read_bytes()
        versions[transcript.bytes_sha256(data)] = data
        return data

    current = keep()
    entries = {line["n"]: line for line in lines if line.get("kind") == "enter"}
    called = {line["enter_n"] for line in lines if line.get("kind") == "call" and line.get("enter_n") is not None}
    unserved = {}  # the recorded number of the entry of a call cancelled in its service: (its number in the replay, its server)
    for line in lines:
        kind = line.get("kind")
        recorded = {k: v for k, v in line.items() if k not in ("n", "kind", "pid")}
        if kind == "enter" and line["n"] not in called:
            unserved[line["n"]] = transcript.append(folder, "enter", **recorded), line.get("server")
            continue
        if kind == "exit" and line.get("enter_n") in unserved:
            entered, server = unserved[line["enter_n"]]
            transcript.append(folder, "exit", server=server, enter_n=entered, outcome="cancelled")  # as episode.leave writes it
            continue
        if kind in ("enter", "exit"):
            continue  # the server writes these lines itself while the replay serves the call
        if kind == "handover":
            episode.hand_over(Path(folder) / STATE_FILE, wall=line["wall"], mono=line["mono"])
            continue
        if kind == "ready":  # the ready line of a server process, in its place, so that the clock credits the same starts
            transcript.append(folder, "ready", **recorded)
            continue
        if ledger is not None and "failures_read" in line:
            (Path(folder) / model_failures.NAME).write_bytes(ledger[:line["failures_read"]])
        key = (line.get("server"), line.get("pid"))
        no_call = line.get("tool") == NO_CALL_ACTION
        if key not in apps and not no_call:
            apps[key] = server_object(line["server"], folder, meta, left_out, store)
        send = (lambda: feed_no_call(folder, store)) if no_call else (lambda: feed(apps[key], line))
        wanted = line.get("released_sha256") if "wall" not in line else None
        earlier = wanted is not None and wanted != transcript.bytes_sha256(current) and wanted in versions
        if wanted is not None and wanted not in versions:
            problems.append(f"{label}: line {line['n']} names a version of the released evidence that the replay never produced")
        if earlier:
            released.write_bytes(versions[wanted])
        try:
            if "wall" in line:
                stamps = recorded_stamps(line, entries)
                with episode.recorded_clock(line["wall"], monotonic=stamps[0][1] if stamps else None, generated=line.get("generated_tokens"),
                                            stamps=stamps):
                    await send()
            else:
                await send()
        finally:
            if earlier:
                released.write_bytes(current)
        current = keep()
    return problems


def recorded_stamps(line, entries):
    """The readings of a recorded call that its server takes again in a replay, in the order it takes them: the entry from the
    enter line that the call line names, and the end of the service of a stepped call. None for a call without an entry."""
    entry = entries.get(line.get("enter_n"))
    if entry is None:
        return []
    stamps = [(entry["wall"], entry["mono"], entry["thread_cpu"])]
    if line.get("end_mono") is not None:
        stamps.append((line["end_wall"], line["end_mono"], line["end_thread_cpu"]))
    return stamps


def service_order(lines):
    """(lines, numbers): a transcript in the order of service, where the entry line and the exit line of a call stand beside the
    line of that call, the entry first and the call line and the exit line after it in the order they were written, every line
    renumbered by its place and every enter_n by the place of its entry; numbers holds the number each line has in the file. The
    hold is one lock per server process, so a line that another process wrote while a call was served stands between the entry and
    the exit of that call in the record, while a replay feeds one line at a time and the server of a fed call writes its entry, its
    line and its exit together. A line that names no entry, and the entry and the exit of a call whose line was never written, keep
    their places."""
    called = {line["enter_n"] for line in lines if line.get("kind") == "call" and line.get("enter_n") is not None}
    entries = {line["n"]: line for line in lines if line.get("kind") == "enter" and line["n"] in called}
    named = {}
    for line in lines:
        if line.get("kind") in ("call", "exit") and line.get("enter_n") in entries:
            named.setdefault(line["enter_n"], []).append(line)
    ordered, placed = [], set()
    for line in lines:
        entry = line["n"] if line.get("kind") == "enter" else line.get("enter_n")
        if entry not in entries:
            ordered.append(line)
        elif line.get("kind") == "call" and entry not in placed:  # the first call line that names the entry places the call
            ordered += [entries[entry], *named[entry]]
            placed.add(entry)
    place_of = {line["n"]: place for place, line in enumerate(ordered)}
    renumbered = [{**line, "n": place, **({"enter_n": place_of[line["enter_n"]]} if line.get("enter_n") in place_of else {})}
                  for place, line in enumerate(ordered)]
    return renumbered, [line["n"] for line in ordered]


def transcript_problems(recorded, replayed, label):
    """(problems, lines compared) of two transcripts, line by line in the order of service (service_order) and in every field but
    the process id. A problem names a line by its number in the recorded file, or in the replay's for a line the run did not
    record."""
    (recorded, numbers), (replayed, replayed_numbers) = service_order(recorded), service_order(replayed)
    problems = []
    for place in range(max(len(recorded), len(replayed))):
        if place >= len(replayed):
            problems.append(f"{label}: line {numbers[place]} has no line in the replay")
        elif place >= len(recorded):
            problems.append(f"{label}: line {replayed_numbers[place]} of the replay was not recorded")
        elif differing := sorted(field for field in set(recorded[place]) | set(replayed[place])
                                 if field != "pid" and recorded[place].get(field) != replayed[place].get(field)):
            problems.append(f"{label}: line {numbers[place]} differs in {', '.join(differing)}")
    return problems, min(len(recorded), len(replayed))


def _without(value, keys):
    return {k: v for k, v in value.items() if k not in keys} if isinstance(value, dict) else value


def file_problems(ep_dir, folder, files, label, places=({}, {})):
    """The left files of an episode that differ from those of its replay, as parsed JSON, by path and never by value. The state
    file names the counted call in service by the number of its entry line, which a replay writes at another number where the
    record interleaves, so each side names that entry by its place in the order of service (places: the place of every line
    number of the recorded transcript and of the replayed one)."""
    problems = []
    for path, what, volatile in files:
        recorded, replayed = Path(ep_dir) / path, Path(folder) / path
        if not recorded.exists() and not replayed.exists():
            continue
        if not replayed.exists():
            problems.append(f"{label}: the replay left no {path}")
        elif not recorded.exists():
            problems.append(f"{label}: the replay left {path}, which the run did not")
        else:
            left = [_without(read_json(file), volatile) for file in (recorded, replayed)]
            for value, place_of in zip(left, places):
                service = value.get("open_service") if isinstance(value, dict) else None
                if isinstance(service, dict) and service.get("n") in place_of:
                    value["open_service"] = {**service, "n": place_of[service["n"]]}
            if left[0] != left[1]:
                problems.append(f"{label}: the {what} differs from {path}")
    return problems


def verdict_input_problems(ep_dir, folder, label):
    """The inputs of the verifier that the verdict repeats (VERDICT_INPUTS) and that differ between an episode and its replay, by
    name and never by value. A run scored again from its recorded log takes these from the run, so a replay where one differs shows
    more than other scores."""
    recorded, replayed = read_json(Path(ep_dir) / HIDDEN / "verdict.json"), read_json(Path(folder) / HIDDEN / "verdict.json")
    if not isinstance(recorded, dict) or not isinstance(replayed, dict):
        return []
    differing = [key for key in VERDICT_INPUTS if recorded.get(key) != replayed.get(key)]
    return [f"{label}: the verdict was computed from another {', '.join(differing)} than {HIDDEN}/verdict.json records"] if differing else []


def torn(exc):
    """The place a transcript reader named for a torn or edited file, without the path."""
    found = re.search(r"line (\d+)", str(exc))
    return f"line {found.group(1)} of the transcript is torn or out of place" if found else "the transcript cannot be read"


def replay_episode(ep_dir, meta, scratch, store, world, listing=False):
    """{"problems", "scoring_problems", "row", "lines", "listings"} of one episode folder replayed into a folder of scratch: the
    problems name the episode, the line and the field or the file that differs, never a value; scoring_problems are those of them
    that the files the verifier writes from what it reads give (SCORE_FILES); row holds the score cells that the verifier of this
    tree gave the replay; lines counts the transcript lines compared; with listing, listings holds the hash of the tool listing that
    every server of the run sends on the replayed episode."""
    from CausalDiagnosis.experiments.episode_runners import run_episodes
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
    from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import SURFACES, tool_surface
    from CausalDiagnosis.utils import model_failures
    from CausalDiagnosis.mcp_servers.utils import transcript
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

    ep_dir = Path(ep_dir)
    label = f"{ep_dir.parent.name}/{ep_dir.name}"
    out = {"problems": [], "scoring_problems": [], "row": None, "lines": 0, "listings": {}}
    try:
        recorded = transcript.read(ep_dir)
    except ValueError as exc:
        out["problems"].append(f"{label}: {torn(exc)}")
        return out
    if not recorded or recorded[0].get("kind") != "start":
        out["problems"].append(f"{label}: no transcript, so the episode cannot be replayed")
        return out
    state = read_json(ep_dir / STATE_FILE)
    config = EpisodeConfig.from_json(state["config"])
    # the episode starts from its state file, and a copy that rescore_under_verifier_scope.py scored again is verified under the scope of the rescoring
    scope = run_record_checks.rescored_scope(meta)
    scoring = config if scope is None else EpisodeConfig.from_json({**state["config"], "verifier_scope": scope})
    folder = Path(scratch) / ep_dir.name
    folder.mkdir(parents=True)
    # the declared surface whose servers the run registered serves the engine, the start, the store and the export file
    surface = next((name for name, entry in SURFACES.items() if set(meta.get("servers") or ()) & set(entry["servers"])), None) if store else None
    env_class = tool_surface(surface).env_class if surface else HydraulicRigEnv
    start = tool_surface(surface).start if surface else episode.start
    # both readings of the start line, since a preset that charges wall seconds keeps the gap from the start to the first call
    with episode.recorded_clock(recorded[0].get("wall"), monotonic=recorded[0].get("mono")):
        start(folder / STATE_FILE, state["seed"], config, world[0])
    left_out = tuple(episode.replay(episode.load_state(folder / STATE_FILE), env_class=env_class).initial)
    files = list(LEFT_FILES)
    episode_store = None
    if surface:
        files.append((SURFACES[surface]["export_file"], "documents of the store", ()))
        episode_store = tool_surface(surface).store(folder / STATE_FILE, store)
    ledger_file = ep_dir / model_failures.NAME
    ledger = ledger_file.read_bytes() if ledger_file.is_file() else None
    if ledger is not None and not any("failures_read" in line for line in recorded):
        # a transcript that names no ledger offset (played before the offset was recorded) gets the whole ledger at once
        (folder / model_failures.NAME).write_bytes(ledger)
        ledger = None
    try:
        if episode_store:
            episode_store.create()
        out["problems"] += asyncio.run(_feed_lines(recorded[1:], folder, meta, left_out, store, label, ledger))
        # the fields the driver wrote outside any call have no line to replay, so the replay takes them as the driver wrote them
        driven = {key: state[key] for key in run_episodes.DRIVER_STATE_FIELDS if key in state}
        if driven:
            run_episodes.play_one_episode.update_state(folder / STATE_FILE, **driven)
        replayed_lines = transcript.read(folder)
        problems, out["lines"] = transcript_problems(recorded, replayed_lines, label)
        out["problems"] += problems
        places = [{number: place for place, number in enumerate(service_order(lines)[1])} for lines in (recorded, replayed_lines)]
        if listing:  # after the comparison, since the servers built for the listings add lines to the scratch transcript
            out["listings"] = asyncio.run(_listings(meta.get("servers") or [], folder, meta, left_out, store))
        options = {"judge": None, "grade": meta.get("grade_lookup", True), "ranking_model": meta.get("ranking_model") or DEFAULT_RANKING_MODEL}
        try:
            out["row"], _ = run_episodes.verify_episode(folder, scoring, world[1], world[2], state["seed"], options, env_class)
        except Exception:  # noqa: BLE001 - a log the verifier cannot read: the left files and the cells say whether the run met the same
            out["row"] = {column: None for column in SCORE_COLUMNS}
        if episode_store:
            episode_store.export(folder)
    finally:
        if episode_store:
            episode_store.drop()
    out["scoring_problems"] = file_problems(ep_dir, folder, [left for left in files if left[0] in SCORE_FILES], label)
    out["problems"] += file_problems(ep_dir, folder, [left for left in files if left[0] not in SCORE_FILES], label, places) + out["scoring_problems"]
    out["problems"] += verdict_input_problems(ep_dir, folder, label)
    return out


async def _listings(servers, folder, meta, left_out, store):
    """{server: SHA-256 of the tool listing} that the servers of this tree send for the servers of a run, each built on a replayed
    episode folder, so that a listing reads the episode as the run's servers did."""
    import mcp.types as mcp_types
    from CausalDiagnosis.mcp_servers.utils import transcript

    hashes = {}
    for server in servers:
        app = server_object(server, folder, meta, left_out, store)
        result = await app.request_handlers[mcp_types.ListToolsRequest](mcp_types.ListToolsRequest(method="tools/list"))
        hashes[server] = transcript.listing_sha256(result.root.tools)
    return hashes
