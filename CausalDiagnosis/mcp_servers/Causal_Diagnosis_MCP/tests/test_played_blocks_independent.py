"""Leak tests of the exclusion of served blocks from the fits of the belief of the CausalDiagnosis MCP.

The CausalDiagnosis MCP leaves out of the fits behind its belief every recorded block that served a released reading, which it
finds by an exact lookup of the released values in its own block table, keeps every number about a named combination on the fits
without the start's block, and pads the build of the belief's fits so that the time of a call does not follow the number of blocks
left out (CausalDiagnosis/docs/causal-diagnosis-mcp.md, Repairs and drops). The tests play one development start through the
driver on three rows, the base row with a repair, the deterioration row with a scheduled drop that no reading shows before the
diagnostics calls that follow it, and an AssetOpsBench row with a repair, and use CausalTools where the driver cannot reach. Five
channels are checked: the answers, the receipts and episode files, the time of a call, a hidden drop and a decoy block. Each check
is also run on a planted leaky variant, which it must catch; a timing check whose plant goes uncaught under the load of the
machine is skipped as inconclusive.

The seams of the server the tests use: Models.without(start, also_without), through which every fit of an episode is built and
which a spy reads and, for the twin without exclusions, the decoy and the planted variants, changes; Models.served_blocks and
Models.without_longest, replaced in the time test and its planted variant; CausalTools._released, CausalTools._belief_model and the
module time of causal_tools, through which the time test reaches the padded build; and CausalTools._timed_build and
CausalTools._belief, through which the decoy test builds and reads the belief of each arm; the planted belief of a verification
replaces CausalTools._belief_model, and the check of the state built after a build of the belief's fits wraps
CausalTools._timed_build and Resampler.table. Everything else goes through the four
tools, released.json, the launcher and the hidden log of the episode, whose served readings carry the combination they were
recorded in. Run folders go under pytest's tmp_path."""
import asyncio
import contextlib
import dataclasses
import gzip
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import build_blocks, default_facts_path, load_cycles

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)
if importlib.util.find_spec("mcp") is None:
    pytest.skip("needs the mcp package", allow_module_level=True)

from CausalDiagnosis.datasets.dataset_loader.evidence_source import BLOCK_CAP  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile  # noqa: E402
from CausalDiagnosis.datasets.splits.development_split import development_starts  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HIDDEN_FIELDS, HydraulicRigEnv  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import run_episodes  # noqa: E402
from CausalDiagnosis.agents.utils.mcp_sessions import open_sessions  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.episode_launchers import write_launchers  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_episodes import play_one_episode  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import HIDDEN  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER, IOT_SERVER, RIG_SERVER, TASK_SERVER  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.agents import scripted_wiring_check  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice import conformal_intervals  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.conformal_intervals import interval, predict_summaries  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.diagnostics_model import EffectDiagnostics, diagnostics_backend  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.effect_model import combo_key, parse_combo  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.effect_model.episode_fit import Resampler  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import CausalTools  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import causal_tools as causal_tools_module  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import fitted_models, tool_texts  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.fitted_models import load_models  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.intervention_effect import CONFIRMATION_MASS, CONFIRMATION_RATIO  # noqa: E402
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.fixture_episode import FixtureEpisode  # noqa: E402
from CausalDiagnosis.utils.paths import preset_path  # noqa: E402

PROFILE = get_profile(DEFAULT_ASSET)
COMPONENTS = tuple(PROFILE.components)
# a development start with no component at its worst level, so the failure card ends no episode before its drop, whose scripted
# repair of the pump moves the rig to the development combination (20, 80, 0, 115), and every one level drop of which is a
# development combination as well
DEV_START = (20, 80, 1, 115)
TOOLS = ("get_episode_evidence", "rank_root_causes", "estimate_intervention_effect", "verify_outcome")
BELIEF_TOOLS = ("rank_root_causes", "estimate_intervention_effect", "verify_outcome")
TOOL_WRITES = {"predictions.json", "predictions.json.lock"}  # the only files a tool writes (docs: Read only)
# the hidden fields of the environment, less combination, which names the candidate of every ranking; seed is also the declared
# seed of the Monte Carlo of a ranking, under the key monte_carlo, where it is allowed
HIDDEN_KEYS = set(HIDDEN_FIELDS) - {"combination"}
DECLARED_KEYS = {("monte_carlo", "seed")}
# files of an episode folder that carry answers, the start the launcher names, or hidden state by design; every other file is
# scanned for a block the episode served
ANSWER_FILES = {"transcript.jsonl", "raw_answers.jsonl.gz", "trajectory.json", "llm_calls.json", "report.md", "answer.txt", "plan.txt",
                "predictions.json", "released.json", "episode.json", "launch_private.json"}
ALPHA = 0.01  # the level of every test of a trend or a shift below, for the clean code and for its planted variant alike
PERMUTATIONS = 4000
# randomized complete blocks of the timing tests, each holding every number of blocks left out once, at the boundary of the
# tool, at that of the build of the belief's fits and for the work of a call outside that build
TIMING_BLOCKS = {"call": 40, "build": 400, "outside": 300}
TIMING_SIZES = (1, 2, 3, 4, 5)  # the number of blocks left out, the start's included


# ----- the seam: the blocks every build of fits leaves out beside the start's -----

@contextlib.contextmanager
def exclusions(transform=None):
    """Spies on Models.without for the length of the block: transform(start, blocks), when given, replaces the blocks a build was
    asked to leave out beside the start's, and every build appends the blocks it leaves out to the list it yields."""
    seen, original = [], fitted_models.Models.without

    def spied(self, start, also_without=()):
        also = tuple(tuple(h) for h in also_without if tuple(h) != tuple(start))
        if transform is not None:
            also = tuple(transform(tuple(start), also))
        seen.append(also)
        return original(self, start, also)

    fitted_models.Models.without = spied
    try:
        yield seen
    finally:
        fitted_models.Models.without = original


def no_exclusion(start, also):
    return ()


# ----- the world and the fits -----

@pytest.fixture(scope="module")
def world():
    cycles = load_cycles(default_facts_path())
    return cycles, build_blocks(cycles)


@pytest.fixture(scope="module")
def models():
    """The process-wide models. The seams of this file patch the class Models, so an instance attribute of these models that
    shadows one of its methods (what pytest's undo of a monkeypatch on the instance leaves behind) would bypass them; it fails
    here by name."""
    loaded = load_models()
    shadowed = sorted(name for name in vars(loaded) if callable(getattr(type(loaded), name, None)))
    assert not shadowed, f"the process-wide models carry instance attributes that shadow methods of Models: {shadowed}"
    return loaded


_START_ONLY = {}


def start_only(models, start):
    """The fits of an episode of the start, built through the seam's original, as build_server builds them at launch."""
    if start not in _START_ONLY:
        _START_ONLY[start] = models.without(tuple(start))
    return _START_ONLY[start]


def as_combo(value):
    return tuple(value[c] for c in COMPONENTS) if isinstance(value, dict) else tuple(parse_combo(value))


def test_the_fixed_combinations_are_development_starts(world):
    starts = set(development_starts(PROFILE, set(world[1])))
    one_drop = {tuple(PROFILE.degradation[c][PROFILE.degradation[c].index(level) + 1] if k == j else other
                      for k, other in enumerate(DEV_START)) for j, (c, level) in enumerate(zip(COMPONENTS, DEV_START))}
    assert {DEV_START, (20, 80, 0, 115)} | one_drop <= starts


# ----- the driver with a recorder in the diagnostics server -----

# executed by the launcher of the diagnostics server before main: every call of a tool of CausalTools leaves one record with the
# released evidence and the predictions it read, its raw result, its seconds and the files of the folder before and after it.
# PLANT hidden_state is the planted leak of the answers and of the hidden drop: every fit built during a call also leaves out the
# combination the hidden state of the episode holds now, read by replaying the state file, as a design that excluded the blocks
# of the hidden history would. PLANT left_out_cache is the planted leak of a persistent server: the belief's fits are built again
# only when the blocks left out change, so a call is faster when they do not. Every record also counts the builds of the belief's
# fits during its call and names the process.
HOOK = '''
import json as _json
import os as _os
import time as _time
from pathlib import Path as _Path

from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import causal_tools as _causal_tools
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import fitted_models as _fitted_models

_RECORDS = _Path(@RECORDS@)
_PLANT = @PLANT@
_IN_CALL = []


def _files(folder):
    return {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(folder.iterdir()) if p.is_file()}


def _hidden_now(folder):
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state
    return tuple(episode_state.replay(episode_state.load_state(folder / "episode.json")).hidden)


_without = _fitted_models.Models.without


def _planted_without(self, start, also_without=()):
    if _PLANT == "hidden_state" and _IN_CALL:
        hidden = _hidden_now(_IN_CALL[-1])
        if hidden != tuple(start) and hidden not in [tuple(h) for h in also_without]:
            also_without = (*also_without, hidden)
            (_RECORDS / "plant_fired").write_text("1", encoding="utf-8")
    return _without(self, start, also_without)


_fitted_models.Models.without = _planted_without


def _recorded(name):
    original = getattr(_causal_tools.CausalTools, name)

    def call(self, *args, **kwargs):
        folder = self.released_path.parent
        released = (folder / "released.json").read_text(encoding="utf-8")
        predictions = folder / "predictions.json"
        before = predictions.read_text(encoding="utf-8") if predictions.exists() else None
        files = _files(folder)
        _IN_CALL.append(folder)
        _BUILDS[:] = []
        started = _time.perf_counter()
        try:
            result = original(self, *args, **kwargs)
        finally:
            _IN_CALL.pop()
        seconds = _time.perf_counter() - started
        record = {"tool": name, "args": list(args), "kwargs": dict(kwargs), "released": released, "predictions_before": before,
                  "predictions_after": predictions.read_text(encoding="utf-8") if predictions.exists() else None,
                  "result": result.model_dump(mode="json"), "result_type": type(result).__name__, "seconds": seconds,
                  "files_before": files, "files_after": _files(folder), "builds": len(_BUILDS), "pid": _os.getpid()}
        (_RECORDS / f"{_time.time_ns():020d}_{_os.getpid()}_{name}.json").write_text(_json.dumps(record), encoding="utf-8")
        return result

    setattr(_causal_tools.CausalTools, name, call)


for _name in ("get_episode_evidence", "rank_root_causes", "estimate_intervention_effect", "verify_outcome"):
    _recorded(_name)

_BUILDS = []
_timed_build = _causal_tools.CausalTools._timed_build


def _counted_build(self, source, blocks):
    _BUILDS.append(tuple(blocks))
    return _timed_build(self, source, blocks)


_causal_tools.CausalTools._timed_build = _counted_build
_belief_model = _causal_tools.CausalTools._belief_model


def _keyed_by_left_out(self, released):
    served = tuple(self.models.source.served_blocks(released.raw.get("observations", [])))
    if getattr(self, "_planted_served", None) == served and self._belief_backend is not None:
        (_RECORDS / "plant_fired").write_text("1", encoding="utf-8")
        return self._belief_backend
    self._planted_served, self._belief_key = served, None
    return _belief_model(self, released)


if _PLANT == "left_out_cache":
    _causal_tools.CausalTools._belief_model = _keyed_by_left_out
'''
LAUNCH_LINE = "from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.main import main"

RANKING_ARGS = json.dumps({"top_k": 200, "branch_detail": "summary"})  # every combination with positive mass, with the attribution
EXTRA = [("Rank the candidates again", CAUSAL_SERVER, "rank_root_causes", RANKING_ARGS),
         ("List the evidence again", CAUSAL_SERVER, "get_episode_evidence", json.dumps({"sensor_group": "all"})),
         ("Predict one more cycle without a repair", CAUSAL_SERVER, "estimate_intervention_effect", json.dumps({"action": "monitor"}))]


def _ranked(steps):
    return [(task, server, tool, RANKING_ARGS if tool == "rank_root_causes" else args) for task, server, tool, args in steps]


def base_plan(steps):
    """The rig's scripted plan, with the reading that follows the repair taken of the full sensor group, which holds the
    indicating channel of the repaired component, so that the verification is scored, and the diagnostics calls asked again
    after that reading."""
    steps = _ranked(steps)
    full = next(step[3] for step in steps if step[2] == "read_sensors" and "full sensor group" in step[0])
    at = next(i for i, step in enumerate(steps) if step[2] == "read_sensors" and "after the repair" in step[0])
    return steps[:at] + [(steps[at][0], steps[at][1], steps[at][2], full)] + EXTRA + steps[at + 1:]


DROP_WAIT = 24  # the timeout of the wait of drop_plan, in cycles


def drop_plan(steps):
    """The rig's scripted plan, with one wait before the diagnosis, during which the scheduled drop falls, and the diagnostics
    calls asked again after it with no reading in between, while no diagnosis has stopped the advice of a ranking."""
    steps = _ranked(steps)
    at = next(i for i, step in enumerate(steps) if step[2] == "submit_diagnosis")
    wait = ("Wait for a notification", RIG_SERVER, "wait_for_notification", json.dumps({"timeout": DROP_WAIT}))
    return steps[:at] + [wait] + EXTRA + steps[at:]


def aob_plan(steps):
    """The plan over AssetOpsBench's servers, with waits after the work order until it closes and two more cycles play, the
    newest recorded row read, and the diagnostics calls asked again."""
    steps = _ranked(steps)
    latest = next(step for step in steps if step[2] == "latest_reading")
    at = next(i for i, step in enumerate(steps) if step[2] == "generate_work_order") + 1
    waits = [("Wait for a notification", TASK_SERVER, "wait_for_notification", json.dumps({"timeout": timeout})) for timeout in (10, 10, 2)]
    return steps[:at] + waits + [("Read the newest recorded row again", IOT_SERVER, "latest_reading", latest[3])] + EXTRA + steps[at:]


ROWS = {"base": ("layer_presets/base", "rig", [], base_plan),
        "drop": ("ablation_variants/scheduled", "rig", ["--clock-mode", "declared"], drop_plan),
        "aob": ("layer_presets/real_time", "aob", ["--clock-mode", "declared"], aob_plan)}


def preset_of(row):
    """The row's preset widened to the one development start, as on_starts of the development split widens a preset."""
    name = ROWS[row][0].split("/")[-1]
    preset = json.loads(Path(preset_path(name)).read_text(encoding="utf-8"))
    faulty = sum(level != PROFILE.healthy[c] for c, level in zip(COMPONENTS, DEV_START))
    return {**preset, "name": f"leak_{row}", "allowed": [list(DEV_START)], "fault_rule": {"min_faulty": faulty, "max_faulty": faulty}}


def drop_seed(world, preset_file, after):
    """The smallest seed whose scheduled first drop falls after the cycle after and before the wait of drop_plan ends."""
    _, config = load_config(str(preset_file))
    config = type(config).from_json(dataclasses.replace(config, clock_mode="declared").to_json())
    for seed in range(1000):
        env = HydraulicRigEnv(*world, config)
        env.reset(seed)
        schedule = env.log[0]["schedule"]
        if schedule and after < min(drop["t"] for drop in schedule) < after + DROP_WAIT:
            return seed
    raise AssertionError("no seed drops after the last reading of the plan")


@dataclasses.dataclass
class Episode:
    folder: Path
    records: list
    log: list
    plant: str

    @property
    def start(self):
        return tuple(self.log[0]["hidden"])

    def served(self):
        """{reading number: the combination whose block served it}, from the hidden log."""
        return {o["n"]: tuple(o["combination"]) for e in self.log if e.get("event") == "read" for o in e.get("served") or ()}

    def played(self, released):
        """The blocks whose recordings released.json holds, the start's included: the truth the exclusions are checked against."""
        served = self.served()
        return {self.start} | {served[o["n"]] for o in json.loads(released)["observations"]}


def play(tmp, world, row, plant="none"):
    """One scripted episode of the row through the driver, with the recorder and the plant in the diagnostics server."""
    preset_dir, servers, extra, plan = ROWS[row]
    preset_file = tmp / f"leak_{row}.json"
    preset_file.write_text(json.dumps(preset_of(row)), encoding="utf-8")
    records = tmp / "records"
    records.mkdir()
    hook = tmp / "hook.py"
    hook.write_text(HOOK.replace("@RECORDS@", repr(str(records))).replace("@PLANT@", repr(plant)), encoding="utf-8")
    original_steps, original_launchers = scripted_wiring_check.scripted_steps, play_one_episode.write_launchers

    def steps(profile, aob=False):
        return plan(original_steps(profile, aob=aob))

    def launchers(ep_dir, *args, **kwargs):
        paths = original_launchers(ep_dir, *args, **kwargs)
        launcher = Path(ep_dir) / f"{CAUSAL_SERVER}_server.py"
        text = launcher.read_text(encoding="utf-8")
        assert LAUNCH_LINE in text
        launcher.write_text(text.replace(LAUNCH_LINE, f"exec(compile(open({str(hook)!r}).read(), {str(hook)!r}, 'exec'))\n{LAUNCH_LINE}"),
                            encoding="utf-8")
        return paths

    seed = drop_seed(world, preset_file, after=DROP_AFTER) if row == "drop" else 0
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(scripted_wiring_check, "scripted_steps", steps)
        mp.setattr(play_one_episode, "write_launchers", launchers)
        run_episodes.main(["--config", str(preset_file), "--agent", "scripted", "--seeds", "1", "--first-seed", str(seed), "--no-grade",
                           "--servers", servers, *extra, "--out", str(tmp / "runs")])
    run = next(p for p in (tmp / "runs").iterdir() if p.is_dir())
    folder = next(p for p in run.iterdir() if p.is_dir() and p.name.startswith("seed"))
    calls = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(records.glob("*.json"))]
    episode = Episode(folder, calls, json.loads((folder / HIDDEN / "log.json").read_text(encoding="utf-8")), plant)
    episode.fired = (records / "plant_fired").exists()
    return episode


# the cycle at which the last reading of drop_plan before its wait ends on the deterioration row, a property of the plan's timing
# on the engine; the precondition test of the row checks that the drop falls after every reading the calls after the wait read
DROP_AFTER = 2


@pytest.fixture(scope="module")
def episodes(tmp_path_factory, world):
    played = {}

    def get(row, plant="none"):
        if (row, plant) not in played:
            played[(row, plant)] = play(tmp_path_factory.mktemp(f"{row}_{plant}"), world, row, plant)
        return played[(row, plant)]

    return get



# ----- calls made again in process -----

_SEEN = []  # the builds of the call running now, which the planted variants read


def files_of(folder):
    return {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(folder.iterdir()) if p.is_file()}


def replay(record, folder, models, start, transform=None, tools_class=CausalTools, start_fits=None):
    """The call of the record made again by tools_class in a fresh folder that holds only the released evidence and the
    predictions the call read, on the fits of the start as build_server builds them unless start_fits names others. Returns a
    record of the same form, with the blocks every build of the call was asked to leave out beside the start's under builds."""
    folder.mkdir(parents=True)
    (folder / "released.json").write_text(record["released"], encoding="utf-8")
    if record["predictions_before"] is not None:
        (folder / "predictions.json").write_text(record["predictions_before"], encoding="utf-8")
    fits = start_only(models, start) if start_fits is None else start_fits
    before = files_of(folder)
    with exclusions(transform) as seen:
        _SEEN[:] = [seen]
        result = getattr(tools_class(fits, folder / "released.json"), record["tool"])(*record["args"], **record["kwargs"])
    predictions = folder / "predictions.json"
    return {**record, "result": result.model_dump(mode="json"), "result_type": type(result).__name__, "builds": list(seen),
            "predictions_after": predictions.read_text(encoding="utf-8") if predictions.exists() else None,
            "files_before": before, "files_after": files_of(folder), "folder": str(folder)}


def replayed(episode, tmp, models, tag, **options):
    return [replay(record, tmp / tag / f"{i:03d}", models, episode.start, **options) for i, record in enumerate(episode.records)]


@pytest.fixture(scope="module")
def replays(tmp_path_factory, episodes, models):
    """The calls of an episode made again, clean (as the server makes them) or twin (every build without exclusions)."""
    made = {}

    def get(row, tag):
        if (row, tag) not in made:
            transform = {"clean": None, "twin": no_exclusion}[tag]
            made[(row, tag)] = replayed(episodes(row), tmp_path_factory.mktemp(f"{row}_{tag}"), models, tag, transform=transform)
        return made[(row, tag)]

    return get


def released_of(record):
    return json.loads(record["released"])


def before_release(episode, record):
    """True while every reading the call read came from the start's block."""
    return episode.played(record["released"]) == {episode.start}


# ----- the checks, each a list of problems, empty for the server and not for its planted variant -----

def hidden_input(episode, again):
    """Calls whose raw answer differs from the same call made again from the released evidence and predictions it read alone."""
    return [(i, r["tool"]) for i, (r, a) in enumerate(zip(episode.records, again)) if r["result"] != a["result"]]


def exclusion_errors(episode, again):
    """Calls whose fits of the belief, the last build of the call, leave out another set of blocks than those whose recordings
    the released evidence holds, calls with any build that leaves out a block no released reading came from (a build that
    only times the padding leaves out the start's block alone), and rankings or estimates that built no fit of the belief at
    all (a verification builds one only for its recovery)."""
    errors = []
    for i, a in enumerate(again):
        played = episode.played(a["released"])
        if a["tool"] in ("rank_root_causes", "estimate_intervention_effect") and a["result_type"] != "ErrorResult" and not a["builds"]:
            errors.append((i, a["tool"], "no fit built through Models.without"))
        if a["builds"] and {episode.start, *a["builds"][-1]} != played:
            errors.append((i, a["tool"], sorted(a["builds"][-1]), sorted(played)))
        errors += [(i, a["tool"], "beyond the released readings", sorted(built)) for built in a["builds"] if not {episode.start, *built} <= played]
    return errors


def e14_reference(fits, stored):
    """The summaries of a stored prediction as the fits without the start's block give them for its stored belief after the
    action, {field: [mean, low, high, quantile]}."""
    after = {parse_combo(key): mass for key, mass in stored["after"].items()}
    return {s.field: [s.mean, s.low, s.high, s.quantile] for s in predict_summaries(fits, after, stored["alpha"])}


def named_number_errors(episode, records, models):
    """(index of the call, kind, detail) for every number about a named combination that is not that of the fits without the
    start's block: the anomaly score of an evidence item (E3); the summaries of an estimate and of every stored prediction, pred/N
    and pred/bN alike, mean, interval and quantile, against those fits for the stored belief after the action (E14); the conformal
    quantile of a summary, which those fits fix for the whole episode (quantile); and the predicted means, channels, combined
    score, p value and verdict of a scored verification, against the mean of those fits and their outcome check on the same
    predictions and readings (E16)."""
    fits = start_only(models, episode.start)
    outcome, index = diagnostics_backend(EffectDiagnostics(fits)), {field: k for k, field in enumerate(fits.effect.fields)}
    errors, quantiles, checked = [], {}, {}
    for i, record in enumerate(records):
        result, kind = record["result"], record["result_type"]
        stored = {p["id"]: p for p in json.loads(record["predictions_after"] or "[]")}
        for prediction_id, prediction in stored.items():
            key = json.dumps(prediction, sort_keys=True)
            if key not in checked:
                checked[key] = prediction["predicted"] == e14_reference(fits, prediction)
            if not checked[key]:
                errors.append((i, "E14", f"stored {prediction_id}"))
        if kind == "EpisodeEvidence":
            errors += [(i, "E3", item["evidence_id"]) for item in result["items"]
                       if item["anomaly_score"] != fits.effect.anomaly_scores({item["field"]: item["value"]})[item["field"]]]
        elif kind == "InterventionEffect":
            reference = e14_reference(fits, stored[result["prediction_id"]])
            errors += [(i, "E14", s["field"]) for s in result["predicted"] if [s["mean"], s["low"], s["high"], s["quantile"]] != reference[s["field"]]]
            for s in result["predicted"]:
                quantiles.setdefault(s["field"], set()).add(s["quantile"])
        elif kind == "OutcomeVerification" and result["channels"]:
            mu = fits.effect.mu(as_combo(result["predicted_combination"]))
            errors += [(i, "E16", c["field"]) for c in result["channels"] if c["predicted"] != float(mu[index[c["field"]]])]
            prediction = {"predicted": {c["field"]: [c["predicted"], c["low"], c["high"], None] for c in result["channels"]}, "alpha": result["alpha"]}
            checks, combined, p_value, consistent = outcome.outcome_check(prediction, {c["field"]: c["observed"] for c in result["channels"]})
            if ([(c["field"], c["predicted"], c["observed"], c["low"], c["high"], c["score"]) for c in result["channels"]] != [tuple(c) for c in checks]
                    or (result["combined_score"], result["p_value"], result["consistent"]) != (combined, p_value, consistent)):
                errors.append((i, "E16", "verdict"))
    errors += [(None, "quantile", (field, sorted(q))) for field, q in quantiles.items() if len(q) > 1]
    return errors


VERDICT_KEYS = ("status", "observation", "observed_t", "channels", "n_channels", "alpha", "level_reachable", "combined_score", "p_value",
                "consistent", "predicted_combination", "predicted_mass")


def twin_errors(episode, real, twin):
    """(problems, moved): no answer may differ from that of the twin without exclusions while every reading came from the
    start's block; after that the evidence stays equal, a verification keeps its verdict, and a ranking with the same top
    combination keeps its attribution (E4). moved says that some ranking's masses differ from the twin's after a reading of a
    new block, the sign that the exclusions are in effect."""
    errors, moved = [], False
    for i, (r, t) in enumerate(zip(real, twin)):
        a, b, kind = r["result"], t["result"], r["result_type"]
        if before_release(episode, r):
            if a != b:
                errors.append((i, r["tool"], "differs before a reading of a new block"))
            continue
        if kind == "EpisodeEvidence" and a != b:
            errors.append((i, r["tool"], "evidence"))
        elif kind == "OutcomeVerification":
            errors += [(i, r["tool"], key) for key in VERDICT_KEYS if a.get(key) != b.get(key)]
        elif kind == "RootCauses":
            if a["candidates"][0]["combination"] == b["candidates"][0]["combination"] and explained(a) != explained(b):
                errors.append((i, r["tool"], "attribution"))
            moved = moved or [c["probability"] for c in a["candidates"]] != [c["probability"] for c in b["candidates"]]
    return errors, moved


def explained(ranking):
    """The numbers of the attribution (E4) about the top combination: per component its explained anomaly and the share of every
    channel; the fault probability beside them is the belief's."""
    return [(c["component"], c["explained_anomaly"], c["channels"]) for c in ranking.get("attribution") or ()]


# the openers of the assumption sentences that hold numbers of the call, which follow the belief (tool_texts.CALL_KEYS)
CALL_OPENERS = tuple(opener for opener, _ in tool_texts.CALL_KEYS)


def comparable(text):
    """An assumption text as the twin must match it: a sentence that holds numbers of the call with its numbers masked, every
    other sentence as it is."""
    return re.sub(r"-?\d+(\.\d+)?(e-?\d+)?", "#", text) if text.startswith(CALL_OPENERS) else text


def key_paths(node, path=""):
    if isinstance(node, dict):
        return {path} | {p for key, value in node.items() for p in key_paths(value, f"{path}.{key}")}
    if isinstance(node, list):
        return {path} | {p for value in node for p in key_paths(value, f"{path}[]")}
    return {path}


def text_errors(real, twin):
    """Calls whose assumption texts or key structure differ from those of the twin without exclusions: a text or a field that
    told what the fits leave out would differ there, while the numbers of the call in the sentences that hold them follow the
    belief."""
    errors = []
    for i, (r, t) in enumerate(zip(real, twin)):
        if [comparable(x) for x in r["result"].get("assumptions") or ()] != [comparable(x) for x in t["result"].get("assumptions") or ()]:
            errors.append((i, r["tool"], "assumptions"))
        if key_paths(r["result"]) != key_paths(t["result"]):
            errors.append((i, r["tool"], sorted(key_paths(r["result"]) ^ key_paths(t["result"]))))
    return errors


def tool_write_errors(records):
    """Calls during which a file of the episode folder other than predictions.json and its lock changed."""
    errors = []
    for i, r in enumerate(records):
        before, after = r["files_before"], r["files_after"]
        changed = {name for name in set(before) | set(after) if before.get(name) != after.get(name)} - TOOL_WRITES
        if changed:
            errors.append((i, r["tool"], sorted(changed)))
    return errors


def hidden_history(log):
    """Every combination the hidden state of the episode held, in order, from the hidden log: the start, the combination after
    every closure that changed it and after every drop."""
    held = [tuple(log[0]["hidden"])]
    for entry in log:
        if entry.get("combination_after") is not None:
            held.append(tuple(entry["combination_after"]))
        elif entry.get("event") in ("scheduled_drop", "hazard_drop"):
            now = list(held[-1])
            now[COMPONENTS.index(entry["component"])] = entry["to_level"]
            held.append(tuple(now))
    return list(dict.fromkeys(held))


def forms(combination):
    levels = list(combination)
    return {combo_key(combination), "/".join(map(str, levels)), json.dumps(levels), json.dumps(levels, separators=(",", ":")),
            json.dumps(dict(zip(COMPONENTS, levels))), json.dumps(dict(zip(COMPONENTS, levels)), separators=(",", ":"))}


def text_of(path):
    data = path.read_bytes()
    return (gzip.decompress(data) if path.suffix == ".gz" else data).decode("utf-8", errors="replace")


def scanned_files(folder):
    """The files of an episode folder that no answer, launcher or state of the servers fills: every file outside the hidden
    folder but those of ANSWER_FILES, the launchers and the runner's own record, and the stderr of the diagnostics server."""
    for path in sorted(Path(folder).rglob("*")):
        rel = path.relative_to(folder)
        if not path.is_file():
            continue
        if rel.parts[0] == HIDDEN:
            if path.name.startswith(CAUSAL_SERVER) and path.suffix == ".stderr":
                yield path
            continue
        if path.name in ANSWER_FILES or path.name.endswith("_server.py") or re.search(r"-seed\d+\.json$", path.name):
            continue
        yield path


def served_block_files(folder, combinations):
    """Files of scanned_files that name one of the combinations in any of the forms a file would write it in."""
    patterns = {form for combination in combinations for form in forms(combination)}
    return [str(path.relative_to(folder)) for path in scanned_files(folder) if any(p in text_of(path) for p in patterns)]


def walk(node, parent=None):
    """(node, the key of the dictionary that holds it) for every node of a JSON tree, the containers included."""
    yield node, parent
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, key)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value, parent)


def hidden_field_errors(episode, outputs):
    """Outputs of the diagnostics server that hold a hidden field of the environment as a key, or a recorded cycle of a served
    reading: as an integer beside the number of its reading, or as a number of its own in a text."""
    cycle_of = {o["n"]: o["cycle"] for e in episode.log if e.get("event") == "read" for o in e.get("served") or ()}
    tokens = re.compile(r"(?<![\w.])(" + "|".join(str(c) for c in sorted(set(cycle_of.values()), reverse=True)) + r")(?![\w.])")
    errors = []
    for k, output in enumerate(outputs):
        for node, parent in walk(output):
            if isinstance(node, dict):
                errors += [(k, "key", key) for key in node if key in HIDDEN_KEYS and (parent, key) not in DECLARED_KEYS]
                n = node.get("n")
                if isinstance(n, int) and n in cycle_of and any(type(v) is int and v == cycle_of[n] for key, v in node.items() if key != "n"):
                    errors.append((k, "cycle beside its reading", n))
            elif isinstance(node, str) and tokens.search(node):
                errors.append((k, "cycle in a text", node[:80]))
    return errors


def outputs_of(records):
    """Every answer and every stored prediction of the records."""
    outputs = [r["result"] for r in records]
    for r in records:
        if r["predictions_after"]:
            outputs += json.loads(r["predictions_after"])
    return outputs


def transcript_errors(episode):
    """The lines of the diagnostics server in the transcript: one per recorded call with the hash of the evidence that call
    read, one set of keys for every line, no hidden field, and every raw answer the transcript keeps equal to the recorded
    raw answer of its call."""
    lines = [json.loads(line) for line in (episode.folder / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
    calls = [line for line in lines if line.get("server") == CAUSAL_SERVER and line["kind"] == "call"]
    errors = []
    if [line["released_sha256"] for line in calls] != [hashlib.sha256(r["released"].encode("utf-8")).hexdigest() for r in episode.records]:
        errors.append("the calls of the transcript are not the recorded calls")
    if len({frozenset(line) for line in calls}) > 1:
        errors.append(("keys", sorted({frozenset(line) for line in calls}, key=len)))
    errors += [("key", key) for line in calls for key in line if key in HIDDEN_KEYS]
    kept = [json.loads(json.loads(line)["raw"]) for line in gzip.open(episode.folder / "raw_answers.jsonl.gz", "rt", encoding="utf-8")]
    if kept != [r["result"] for r in episode.records]:
        errors.append("a raw answer the transcript keeps differs from the recorded answer of its call")
    return errors


# ----- the rows played through the driver -----

@pytest.mark.parametrize("row", ["base", "aob"])
def test_the_row_serves_a_reading_of_a_new_block_before_diagnostics_calls(episodes, row):
    """The precondition of the checks: after a repair the row serves a reading of another block than the start's, and every
    tool that builds the fits of the belief is called after it."""
    episode = episodes(row)
    assert episode.records, "the recorder in the diagnostics server wrote nothing"
    assert set(episode.served().values()) - {episode.start}, "no reading of a block other than the start's"
    after = {r["tool"] for r in episode.records if not before_release(episode, r)}
    assert {"rank_root_causes", "get_episode_evidence", "estimate_intervention_effect"} <= after


def drop_of(episode):
    """(cycle of the scheduled drop, the combination after it)."""
    drops = [e for e in episode.log if e.get("event") == "scheduled_drop"]
    assert drops, "the scheduled drop did not fall during the episode"
    held, combination = list(episode.start), None
    for entry in episode.log:
        if entry.get("combination_after") is not None:
            held = list(entry["combination_after"])
        elif entry is drops[0]:
            held[COMPONENTS.index(entry["component"])] = entry["to_level"]
            combination = tuple(held)
    return drops[0]["t_end"], combination


def unread_drop_calls(episode):
    """The indices of the calls made after the drop that read no reading served at or after its cycle, while the episode ran
    with no failure released and no diagnosis submitted, so that nothing public tells of the drop and a ranking still gives its
    advice."""
    t_drop, _ = drop_of(episode)
    calls = []
    for i, r in enumerate(episode.records):
        released = released_of(r)
        if (released["status"]["time"] >= t_drop and all(o["t"] < t_drop for o in released["observations"])
                and not released["status"]["episode_over"] and not {e["event"] for e in released["events"]} & {"failure", "submit_diagnosis"}):
            calls.append(i)
    return calls


def test_the_drop_falls_after_the_readings_of_diagnostics_calls_that_follow_it(episodes):
    """The precondition of the hidden drop: every tool that builds the fits of the belief is called after the scheduled drop and
    before any reading shows its block, while the episode runs, no failure is released and no diagnosis has stopped the
    advice of a ranking."""
    episode = episodes("drop")
    _, dropped = drop_of(episode)
    calls = [episode.records[i] for i in unread_drop_calls(episode)]
    assert {"rank_root_causes", "get_episode_evidence", "estimate_intervention_effect"} <= {r["tool"] for r in calls}
    served = episode.served()
    assert all(dropped not in {served[o["n"]] for o in released_of(r)["observations"]} for r in calls)
    assert all(r["result"]["advice"] is not None for r in calls if r["result_type"] == "RootCauses")


@pytest.mark.parametrize("row", sorted(ROWS))
def test_every_answer_comes_from_what_its_call_read_and_leaves_out_the_played_blocks(episodes, replays, row):
    """Every raw answer of the episode is made again from the released evidence and the predictions its call read,
    in a folder with no other file, and every fit of the belief leaves out exactly the blocks whose readings that evidence holds,
    so an answer cannot change before such a reading and a drop no reading shows changes none."""
    episode = episodes(row)
    again = replays(row, "clean")
    assert not hidden_input(episode, again)
    assert not exclusion_errors(episode, again)


@pytest.mark.parametrize("row", sorted(ROWS))
def test_no_number_about_a_named_combination_moves_with_the_blocks_left_out(episodes, replays, models, row):
    """The anomaly scores, the predicted means and the verification stay on the fits without the start's block; no answer
    differs from the twin without exclusions before a reading of a new block, and the attribution and the verdict agree with
    the twin after it, while the masses move."""
    episode = episodes(row)
    assert not named_number_errors(episode, episode.records, models)
    real, twin = replays(row, "clean"), replays(row, "twin")
    errors, moved = twin_errors(episode, real, twin)
    assert not errors
    if row == "base":  # the one row whose plan verifies after a reading of the repaired block's full sensor group
        assert any(r["result_type"] == "OutcomeVerification" and r["result"]["channels"] for r in episode.records), "no verification was scored"
    if row != "drop":  # the drop row serves no reading of a new block before its calls, so its exclusions do not grow
        assert moved
        assert any(r["result_type"] == "RootCauses" and not before_release(episode, r) and explained(r["result"])
                   and r["result"]["candidates"][0]["combination"] == t["result"]["candidates"][0]["combination"]
                   for r, t in zip(real, twin)), "no ranking after a reading of a new block had its attribution compared"


@pytest.mark.parametrize("row", sorted(ROWS))
def test_no_receipt_or_episode_file_names_a_played_block_or_a_recorded_cycle(episodes, replays, row):
    """No file of the episode outside the answers names a combination the hidden state held other than the start, a call
    writes no file but its predictions, the transcript lines carry one set of keys and the hash of what each call read, no
    answer holds a hidden field or a recorded cycle, and the texts and keys of every answer are those of the twin."""
    episode = episodes(row)
    held = set(hidden_history(episode.log)) - {episode.start}
    assert held, "the hidden state never left the start, so the scan has nothing to look for"
    scanned = list(scanned_files(episode.folder))
    assert any(p.parent.name == HIDDEN for p in scanned) and any(p.parent.name != HIDDEN for p in scanned), scanned
    assert not served_block_files(episode.folder, held)
    assert not tool_write_errors(episode.records)
    assert not transcript_errors(episode)
    assert not hidden_field_errors(episode, outputs_of(episode.records))
    assert not text_errors(replays(row, "clean"), replays(row, "twin"))




# ----- a scored verification, through CausalTools -----

class HandPlayed:
    """A repaired episode played by hand with FixtureEpisode, in the form the checks read: the start, the combination whose block
    served every reading, and the calls of the diagnostics server with what each read."""

    def __init__(self, start):
        self.start, self.combination_of, self.records = start, {}, []

    def served(self):
        return dict(self.combination_of)

    def played(self, released):
        return {self.start} | {self.combination_of[o["n"]] for o in json.loads(released)["observations"]}


def hand_player(episode, play, models, tmp):
    """(read, call) of a hand-played episode: read serves a sensor group and notes the block that served every reading, and call
    makes a call by the server in process from what it read, the predictions of every call feeding the next."""

    def read(group):
        for served in episode.read(group):
            play.combination_of[served["n"]] = tuple(episode.combo)  # no order closes during a read here, so this block served it

    def call(tool, *args):
        before = play.records[-1]["predictions_after"] if play.records else None
        record = {"tool": tool, "args": list(args), "kwargs": {}, "released": episode.released_path.read_text(encoding="utf-8"), "predictions_before": before}
        play.records.append(replay(record, tmp / f"{len(play.records):03d}", models, play.start))

    return read, call


def hand_played(world, models, tmp, again=False):
    """An episode with a scored verification apart from the driver: it reads the start's block, estimates replacing the pump,
    orders it with that prediction, waits for the close, reads the full sensor group of the repaired block and asks for a
    ranking, the evidence and the verification; with again, it then reads that group once more and asks for the verification
    again. Each call is made by the server in process from what it read, and its predictions feed the next call."""
    cycles, blocks = world
    episode, play = FixtureEpisode(tmp / "episode", cycles, blocks, DEV_START, seed=0), HandPlayed(DEV_START)
    read, call = hand_player(episode, play, models, tmp)
    read("tier1")
    read("tier3")
    call("estimate_intervention_effect", "replace:pump")
    order = episode.create_work_order("pump", prediction_id=play.records[-1]["result"]["prediction_id"])
    while any(o["id"] == order for o in episode.open):
        episode.advance(1)
    read("tier3")
    call("rank_root_causes", 200, "summary")
    call("get_episode_evidence", "all")
    call("verify_outcome", play.records[0]["result"]["prediction_id"])
    if again:
        read("tier3")
        call("verify_outcome", play.records[0]["result"]["prediction_id"])
    return play


# two first repairs on development starts, each a repaired component without a dependency and an entry seed; check_preconditions asserts what each case needs
REPAIRS = {"confirms": ((3, 73, 1, 90), "cooler", 0), "guard_refuses": ((3, 73, 2, 115), "accumulator", 0)}


def repaired(world, models, tmp, start, component, seed):
    """An episode of a first repair, two verifications apart from the driver: two reads of the full sensor group, the diagnosis of
    the start submitted, an estimate of replacing the component, its work order, a wait of the estimate's duration, a read of the
    full group and the verification, then one more read and the verification again."""
    cycles, blocks = world
    episode, play = FixtureEpisode(tmp / "episode", cycles, blocks, start, seed=seed), HandPlayed(start)
    read, call = hand_player(episode, play, models, tmp)
    read("tier3")
    read("tier3")
    episode.submit_diagnosis(dict(zip(COMPONENTS, start)))
    call("estimate_intervention_effect", f"replace:{component}")
    estimate = play.records[-1]["result"]
    episode.create_work_order(component, prediction_id=estimate["prediction_id"])
    episode.advance(estimate["duration"])
    for _ in range(2):
        read("tier3")
        call("verify_outcome", estimate["prediction_id"])
    return play


def verified_episode(case, world, models, tmp):
    """The hand-played episode asked twice, or the episode of a first repair of REPAIRS."""
    return hand_played(world, models, tmp, again=True) if case == "hand" else repaired(world, models, tmp, *REPAIRS[case])


def test_a_scored_verification_stays_on_the_fits_without_the_start_and_its_planted_leak_is_caught(world, models, tmp_path):
    """After a reading of the repaired block, the verification is scored, its numbers and those of the ranking and the evidence
    stay on the fits without the start's block and equal the twin's, while the masses move; a server whose named numbers come
    from the fits that leave out the played blocks moves the anomaly scores and the verdict of the verification."""
    play = hand_played(world, models, tmp_path)
    assert set(play.combination_of.values()) == {DEV_START, (20, 80, 0, 115)}
    verification = play.records[-1]["result"]
    assert verification["channels"] and verification["p_value"] is not None, verification["status"]
    assert not named_number_errors(play, play.records, models)
    twin = replayed(play, tmp_path, models, "twin", transform=no_exclusion)
    errors, moved = twin_errors(play, play.records, twin)
    assert not errors and moved
    leaky = [replay(r, tmp_path / "leaky" / f"{i:03d}", models, DEV_START,
                    start_fits=models.without(DEV_START, sorted(play.played(r["released"]) - {DEV_START})))
             for i, r in enumerate(play.records)]
    assert "E3" in {c[1] for c in named_number_errors(play, leaky, models)}
    # without exclusions the planted variant is the server, so the twin stands for the twin of both
    caught, _ = twin_errors(play, leaky, twin)
    assert "verify_outcome" in {tool for _, tool, _ in caught}


class StartOnlyBelief(CausalTools):
    """The planted leak of a confirmation: the belief on the fits without the start's block alone, which keep the blocks that the
    released readings come from."""

    def _belief_model(self, released):
        return self.model


class BeforeLeak(CausalTools):
    """The planted leak of the reading guard: the belief before the readings after the repair (a call of _belief given readings
    to leave out) on the fits without the start's block alone, which keep the block those readings came from, while the belief
    now stays the server's."""

    def _belief(self, released, *args, **kwargs):
        if not (args or kwargs):
            return CausalTools._belief(self, released)
        self._belief_model = lambda released: self.model
        try:
            return CausalTools._belief(self, released, *args, **kwargs)
        finally:
            del self._belief_model


def ranked(record, folder, models, start, released=None, served=None):
    """The candidates of a ranking by a fresh server on the released evidence of the record (or released) and its predictions,
    with Models.served_blocks returning served when given, every combination with positive mass."""
    with pytest.MonkeyPatch.context() as mp:
        if served is not None:
            mp.setattr(fitted_models.Models, "served_blocks", lambda self, observations: list(served))
        ranking = replay({**record, "tool": "rank_root_causes", "args": [200, "summary"], "kwargs": {},
                          "released": record["released"] if released is None else released}, folder, models, start)["result"]
    assert ranking["more_candidates"]["count"] == 0, "the ranking left out a combination with positive mass"
    return ranking["candidates"]


def left_out_readings(record, folder, models, start):
    """The numbers of the readings that the server's verification leaves out of the belief before the readings after the repair,
    as its call of CausalTools._belief given readings to leave out names them, by a replay of the verification."""
    cuts = []

    class Spy(CausalTools):
        def _belief(self, released, *args, **kwargs):
            without = args[0] if args else kwargs.get("without")
            if without:
                cuts.append({int(n) for n in without})
            return CausalTools._belief(self, released, *args, **kwargs)

    replay(record, folder, models, start, tools_class=Spy)
    assert len(cuts) == 1, cuts
    return cuts[0]


def readings_after_the_prediction(record):
    """The numbers of the readings released after the stored prediction that the verification scores."""
    prediction = next(p for p in json.loads(record["predictions_before"]) if p["id"] == record["result"]["prediction_id"])
    return {o["n"] for o in json.loads(record["released"])["observations"] if o["n"] > prediction["observations"]}


def reference_ratio(record, after, cut, folder, models, start):
    """The reading ratio of a verification on the fits that the belief now reads: the odds of the predicted combination against
    the rival (the first other candidate of the ranking now) in the ranking now over the same odds in the ranking without the
    readings of cut, both on the fits that leave out every block the released readings come from, in logs, a ratio beyond the
    largest float being that float; None at a zero mass."""
    raw = json.loads(record["released"])
    before = ranked(record, folder, models, start, served=models.served_blocks(raw["observations"]),
                    released=json.dumps({**raw, "observations": [o for o in raw["observations"] if o["n"] not in cut]}))
    predicted = record["result"]["predicted_combination"]
    rival = next((c["combination"] for c in after if c["combination"] != predicted), None)
    masses = [next((c["probability"] for c in candidates if c["combination"] == combination), 0.0)
              for candidates in (after, before) for combination in (predicted, rival)]
    if rival is None or 0.0 in masses:
        return None
    log_ratio = math.log(masses[0]) - math.log(masses[1]) - math.log(masses[2]) + math.log(masses[3])
    try:
        return math.exp(log_ratio)
    except OverflowError:
        return sys.float_info.max


def confirmation_errors(records, folder, models, start, every_reading_after=False):
    """(index of the call, field, given, expected) for every verification with a confirmation whose numbers are not those of the
    rankings that a fresh server gives on the same evidence: the mass of the predicted combination (0 without positive mass), the
    top combination and its mass, against the ranking now; the reading ratio against reference_ratio (equal within the rounding
    of floating point arithmetic), the readings the server leaves out before being readings after the stored prediction (with
    every_reading_after, all of them); and confirmed against the server's rule (confirms), a consistent verdict, that mass at least
    CONFIRMATION_MASS and that ratio at least CONFIRMATION_RATIO."""
    errors = []
    for i, record in enumerate(records):
        result = record["result"]
        if record["result_type"] != "OutcomeVerification" or result.get("confirmed") is None:
            continue
        after = ranked(record, folder / f"{i:03d}_now", models, start)
        mass = next((c["probability"] for c in after if c["combination"] == result["predicted_combination"]), 0.0)
        expected = {"predicted_posterior": mass, "belief_top_combination": after[0]["combination"], "belief_top_mass": after[0]["probability"]}
        errors += [(i, key, result[key], expected[key]) for key in expected if result[key] != expected[key]]
        cut = left_out_readings(record, folder / f"{i:03d}_cut", models, start)
        after_prediction = readings_after_the_prediction(record)
        if not cut or not cut <= after_prediction or (every_reading_after and cut != after_prediction):
            errors.append((i, "readings left out", sorted(cut), sorted(after_prediction)))
        ratio, given = reference_ratio(record, after, cut, folder / f"{i:03d}_before", models, start), result["reading_ratio"]
        if (ratio is None) != (given is None) or (ratio is not None and not math.isclose(given, ratio, rel_tol=1e-9)):
            errors.append((i, "reading_ratio", given, ratio))
        confirmed = result["consistent"] is True and mass >= CONFIRMATION_MASS and ratio is not None and ratio >= CONFIRMATION_RATIO
        if result["confirmed"] != confirmed:
            errors.append((i, "confirmed", result["confirmed"], confirmed))
    return errors


def check_preconditions(case, records):
    """The first verification of a first repair of REPAIRS decides as its case says: confirmed, or passing on the verdict and the
    mass while the reading guard refuses it."""
    first = next(r["result"] for r in records if r["tool"] == "verify_outcome")
    if case == "confirms":
        assert first["confirmed"] is True, first["message"]
    elif case == "guard_refuses":
        assert (first["consistent"] is True and first["predicted_posterior"] >= CONFIRMATION_MASS and first["reading_ratio"] is not None
                and first["reading_ratio"] < CONFIRMATION_RATIO and first["confirmed"] is False), first["message"]


@pytest.mark.parametrize("case", ["hand", "confirms", "guard_refuses", "base"])
def test_a_confirmation_reads_the_belief_without_the_played_blocks_and_its_planted_leaks_are_caught(world, models, episodes, replays, tmp_path, case):
    """Every verification with a confirmation, on the hand-played episode asked twice, on a first repair that confirms, on one
    that the reading guard refuses and on the base row: the planted belief that keeps the played blocks is caught first, and on
    the hand-played episodes the planted guard whose belief before the readings after the repair keeps them is caught on every
    verification, without which the clean verdict carries no information and the test fails; then the mass of the predicted
    combination, the top combination and its mass are those of a ranking by a fresh server on the same evidence, the reading
    ratio is that of the same fits with and without the readings after the repair, confirmed follows the server's rule, and the
    last build of the verification leaves out exactly the played blocks."""
    if case == "base":
        episode, records = episodes(case), replays(case, "clean")
    else:
        episode = verified_episode(case, world, models, tmp_path / "played")
        records = episode.records
        check_preconditions(case, records)
    with_confirmation = [i for i, r in enumerate(records) if r["result_type"] == "OutcomeVerification" and r["result"].get("confirmed") is not None]
    assert with_confirmation, "no verification with a confirmation"
    leaky = [replay(records[i], tmp_path / "leaky" / f"{i:03d}", models, episode.start, tools_class=StartOnlyBelief) for i in with_confirmation]
    caught = confirmation_errors(leaky, tmp_path / "leaky_rankings", models, episode.start)
    assert caught, "the planted belief that keeps the played blocks was not caught, so the clean verdict is inconclusive"
    print(f"{case}: {len(with_confirmation)} verification(s) with a confirmation, planted mismatches {caught}")
    if case != "base":
        guard = [replay(records[i], tmp_path / "guard" / f"{i:03d}", models, episode.start, tools_class=BeforeLeak) for i in with_confirmation]
        guard_caught = {e[0] for e in confirmation_errors(guard, tmp_path / "guard_rankings", models, episode.start) if e[1] == "reading_ratio"}
        print(f"{case}: planted guard caught on verifications {sorted(guard_caught)} of {len(guard)}")
        assert guard_caught == set(range(len(guard))), "the planted guard was not caught on every verification, so the clean verdict is inconclusive"
    assert not confirmation_errors(records, tmp_path / "rankings", models, episode.start, every_reading_after=case != "base")
    assert not [e for e in exclusion_errors(episode, records) if e[1] == "verify_outcome"]


# ----- the planted leaky variants, each of which its check must catch -----

def planted(name, after):
    """A subclass of CausalTools whose four tools pass their result through after(tools, tool, result)."""
    def tool(method):
        def call(self, *args, **kwargs):
            return after(self, method, getattr(CausalTools, method)(self, *args, **kwargs))
        return call
    return type(name, (CausalTools,), {method: tool(method) for method in TOOLS})


def last_build():
    return _SEEN[0][-1] if _SEEN and _SEEN[0] else ()


def _write_cache(tools, method, result):
    (tools.released_path.parent / "fits_cache.json").write_text(json.dumps({"left_out": [combo_key(h) for h in last_build()]}), encoding="utf-8")
    return result


def _tell_count(tools, method, result):
    if method not in BELIEF_TOOLS or not hasattr(result, "assumptions"):
        return result
    return result.model_copy(update={"assumptions": [*result.assumptions, f"the fits of the belief leave out {1 + len(last_build())} blocks"]})


FILE_LEAK = planted("FileLeak", _write_cache)  # a cache of the blocks left out beside the released evidence
TEXT_LEAK = planted("TextLeak", _tell_count)  # an assumption text that counts the blocks left out


def cycle_leak(episode):
    """An evidence answer whose message names the recorded cycle of every reading it lists, read from the hidden log."""
    cycle_of = {o["n"]: o["cycle"] for e in episode.log if e.get("event") == "read" for o in e.get("served") or ()}

    def tell(tools, method, result):
        if method != "get_episode_evidence" or not hasattr(result, "items"):
            return result
        cycles = sorted({cycle_of[item.n] for item in result.items})
        return result.model_copy(update={"message": f"{result.message}; recorded cycles {', '.join(map(str, cycles))}"})

    return planted("CycleLeak", tell)


def test_a_planted_leak_of_the_hidden_state_is_caught(episodes, models, tmp_path):
    """The planted leak of the answers and of the hidden drop: a server whose fits of the belief also leave out the combination the hidden
    state holds now, read by replaying the state file, answers after the unread drop otherwise than the same calls made again
    from what they read, and the check names those calls."""
    episode = episodes("drop", "hidden_state")
    assert episode.fired, "the planted leak changed no build"
    caught = hidden_input(episode, replayed(episode, tmp_path, models, "clean"))
    unread = [i for i in unread_drop_calls(episode) if episode.records[i]["tool"] in ("rank_root_causes", "estimate_intervention_effect")]
    assert unread and set(unread) <= {i for i, _ in caught}


def test_planted_numbers_about_a_named_combination_on_the_fits_of_the_belief_are_caught(episodes, models, tmp_path):
    """The planted leak of the named numbers: a server whose anomaly scores, predicted means and calibration come from the fits that leave
    out the played blocks moves them at the first reading of a new block, and the check finds it only after that reading."""
    episode = episodes("base")
    leaky = [replay(r, tmp_path / f"{i:03d}", models, episode.start,
                    start_fits=models.without(episode.start, sorted(episode.played(r["released"]) - {episode.start})))
             for i, r in enumerate(episode.records)]
    caught = named_number_errors(episode, leaky, models)
    assert {"E3", "E14", "quantile", "E16"} <= {c[1] for c in caught}, caught
    assert all(not before_release(episode, episode.records[c[0]]) for c in caught if c[0] is not None)


def sigma_plant(models, start, played):
    """The predict_summaries of the planted leak of the intervals: the mean and the quantile of the fits without the start's
    block, the interval from the within standard deviations of the fits of the belief, which leave out the played blocks."""
    sigma = models.without(start, sorted(set(played) - {start})).sigma

    def summaries(fits, after, alpha):
        return [s.model_copy(update=dict(zip(("low", "high"), interval(s.mean, float(sigma[k]), s.quantile))))
                for k, s in enumerate(conformal_intervals.predict_summaries(fits, after, alpha))]

    return summaries


@pytest.mark.parametrize("row", ["base", "aob"])
def test_planted_intervals_on_the_sigma_of_the_belief_are_caught(episodes, models, tmp_path, row):
    """The planted leak of the intervals: a server whose E14 intervals take the within standard deviations of the belief's fits,
    mean and quantile unchanged, moves the intervals of its estimates and stored predictions at the first reading of a new
    block, and the check finds it only after that reading."""
    episode = episodes(row)
    leaky = []
    for i, r in enumerate(episode.records):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(causal_tools_module, "predict_summaries", sigma_plant(models, episode.start, episode.played(r["released"])))
            leaky.append(replay(r, tmp_path / f"{i:03d}", models, episode.start))
    caught = named_number_errors(episode, leaky, models)
    assert "E14" in {c[1] for c in caught}, caught
    assert all(not before_release(episode, episode.records[c[0]]) for c in caught if c[0] is not None)


def test_planted_leaks_of_receipts_and_episode_files_are_caught(episodes, models, tmp_path):
    """The planted leaks of the receipts: a cache of the blocks left out beside the released evidence, an assumption text that counts them
    and an evidence message that names the recorded cycles are each caught, while the same harness on the server finds
    nothing."""
    episode = episodes("base")
    held = set(hidden_history(episode.log)) - {episode.start}
    clean = replayed(episode, tmp_path, models, "clean")
    assert not tool_write_errors(clean) and not any(served_block_files(Path(r["folder"]), held) for r in clean)
    cached = replayed(episode, tmp_path, models, "file", tools_class=FILE_LEAK)
    assert tool_write_errors(cached)
    assert any(served_block_files(Path(r["folder"]), held) for r in cached)
    told = text_errors(replayed(episode, tmp_path, models, "text", tools_class=TEXT_LEAK),
                       replayed(episode, tmp_path, models, "text_twin", tools_class=TEXT_LEAK, transform=no_exclusion))
    assert told and all(not before_release(episode, episode.records[i]) for i, *_ in told)
    assert not hidden_field_errors(episode, outputs_of(clean))
    assert hidden_field_errors(episode, outputs_of(replayed(episode, tmp_path, models, "cycle", tools_class=cycle_leak(episode))))


# ----- the decoy, through CausalTools -----

# The decoy test bounds any lift of a left out unplayed block by the planted variant it runs on the same cells, a raise of the
# left out combination by one place: the share of pairs whose decoy rises stays at or below the midpoint of no lift (one half)
# and the planted share, and the decoy's mean mass change at or below half the planted change.


def by_mass(post):
    """Every combination of a belief by mass and then by the combination, the order in which the server ranks candidates."""
    return [tuple(h) for _, h in sorted(zip(post.probabilities, post.candidates), key=lambda pair: (-pair[0], tuple(pair[1])))]


def raised(order, masses, left_out):
    """The planted leak of the decoy: (the ranking, {combination: mass}) with every left out combination one place above where
    the server put it, the smallest raise a ranking can show, taking the mass of the combination just above it, which takes its
    mass in turn."""
    order, masses = list(order), dict(masses)
    for k in range(1, len(order)):
        if order[k] in left_out and order[k - 1] not in left_out:
            upper, lower = order[k - 1], order[k]
            order[k - 1], order[k] = lower, upper
            masses[upper], masses[lower] = masses[lower], masses[upper]
    return order, masses


def one_sided_sign_p(up, down):
    """P(at least up of up + down fair coin flips come up), the sign test of a shift toward up."""
    n = up + down
    return sum(math.comb(n, k) for k in range(up, n + 1)) / 2 ** n if n else 1.0


_DECOY_CELLS = {}


@pytest.fixture(scope="module", params=[1, 2], ids=["one_reading", "two_readings"])
def decoy_pairs(request, tmp_path_factory, world, models):
    """{(start, decoy): one cell per entry offset}: every development start, every development combination one component away
    from it as the decoy, which the episode never plays, and every entry offset of the recorded blocks, with one or two readings
    of the cheapest sensor group of the start's block, the first live cycles of the entry (param). Every served recording is the
    first live cycle of one entry offset, so one reading reaches every cell of its kind, and two readings give cells of another. A cell holds the decoy's rank and mass in the belief of CausalTools with the
    start's block left out as served (without), with the decoy's block left out as well (with), in the planted variant of the
    second (planted) and in that variant raised once more (twice), the reference that judges the planted variant in the
    server's place, and whether the decoy is the first candidate in the first two. The belief's fits of each arm are
    built once per pair through CausalTools._timed_build, the server's own build, and read by CausalTools._belief."""
    if request.param in _DECOY_CELLS:
        return _DECOY_CELLS[request.param]
    cycles, blocks = world
    starts = development_starts(PROFILE, set(blocks))
    tmp = tmp_path_factory.mktemp(f"decoy_{request.param}")
    seeds = {}
    for seed in range(1000):
        offset = FixtureEpisode(tmp / "offsets" / str(seed), cycles, blocks, DEV_START, seed=seed).entry_offset
        seeds.setdefault(offset, seed)
        if len(seeds) == BLOCK_CAP:
            break
    assert len(seeds) == BLOCK_CAP
    pairs = {}
    for start in starts:
        fits = start_only(models, start)
        builder = CausalTools(fits, tmp / "released.json")
        decoys = sorted(h for h in starts if sum(a != b for a, b in zip(h, start)) == 1)
        backends = {None: builder._timed_build(models, [start])[0], **{d: builder._timed_build(models, [start, d])[0] for d in decoys}}
        for offset, seed in sorted(seeds.items()):
            episode = FixtureEpisode(tmp / f"{combo_key(start).replace(',', '_')}_{offset}", cycles, blocks, start, seed=seed)
            assert episode.entry_offset == offset
            for _ in range(request.param):
                episode.read("tier1")
            tools = CausalTools(fits, episode.released_path)
            released = tools._released(note=False)
            beliefs = {}
            for key, backend in backends.items():
                with pytest.MonkeyPatch.context() as mp:
                    mp.setattr(CausalTools, "_belief_model", lambda self, released, backend=backend: backend)
                    beliefs[key] = tools._belief(released)[1]
            without = by_mass(beliefs[None])
            for d in decoys:
                with_ = by_mass(beliefs[d])
                masses = {key: {tuple(h): float(m) for h, m in zip(beliefs[key].candidates, beliefs[key].probabilities)} for key in (None, d)}
                planted_order, planted_masses = raised(with_, masses[d], {d})
                twice_order, twice_masses = raised(planted_order, planted_masses, {d})
                pairs.setdefault((start, d), []).append({
                    "without": without.index(d) + 1, "with": with_.index(d) + 1, "planted": planted_order.index(d) + 1,
                    "twice": twice_order.index(d) + 1, "mass_without": masses[None][d], "mass_with": masses[d][d],
                    "mass_planted": planted_masses[d], "mass_twice": twice_masses[d],
                    "first_without": without[0] == d, "first_with": with_[0] == d, "readings": request.param})
    _DECOY_CELLS[request.param] = pairs
    return pairs


def rises(pairs, arm):
    """(up, down, tied): the pairs whose decoy's rank, averaged over the entries, is higher, lower or the same in the arm than
    without its block left out."""
    changes = [np.mean([c["without"] - c[arm] for c in cells]) for cells in pairs.values()]
    up, down = int(sum(change > 0 for change in changes)), int(sum(change < 0 for change in changes))
    return up, down, len(changes) - up - down


def decoy_summary(pairs, arm):
    """(rise share, up, down, tied, mean change of the decoy's mass) of the arm against the belief without the decoy's block left
    out."""
    up, down, tied = rises(pairs, arm)
    cells = [c for cells in pairs.values() for c in cells]
    lift = float(np.mean([c[f"mass_{arm}"] - c["mass_without"] for c in cells]))
    return (up / (up + down) if up + down else 0.0), up, down, tied, lift


def test_the_lift_of_a_left_out_unplayed_block_stays_below_half_a_planted_raise(decoy_pairs):
    """Leaving an unplayed block out of the fits of the belief raises its rank in a share of the pairs no larger than the midpoint
    of one half and the share of the planted one-place raise, and its mass on average by no more than half the planted raise,
    over every development start, every development neighbour and every entry, with one reading and with two."""
    share, up, down, tied, lift = decoy_summary(decoy_pairs, "with")
    planted_share, _, _, _, planted_lift = decoy_summary(decoy_pairs, "planted")
    cells = [c for cells in decoy_pairs.values() for c in cells]
    print(f"decoy left out, {next(iter(decoy_pairs.values()))[0]['readings']} reading(s): {len(decoy_pairs)} pairs, {len(cells)} cells; up {up}, down {down}, tied {tied}, rise share {share:.4f} "
          f"(bound {(0.5 + planted_share) / 2:.4f}), one sided sign p {one_sided_sign_p(up, down):.3g}; mean mass change {lift:+.5f} "
          f"(bound {planted_lift / 2:+.5f}); decoy first {np.mean([c['first_without'] for c in cells]):.4f} without, "
          f"{np.mean([c['first_with'] for c in cells]):.4f} with")
    assert share <= (0.5 + planted_share) / 2, (share, planted_share)
    assert lift <= planted_lift / 2, (lift, planted_lift)


def test_a_planted_decoy_leak_fails_the_bounds_in_the_servers_place(decoy_pairs):
    """The planted leak of the decoy: judged in the server's place, with its own one-place raise as the planted reference, the
    variant that ranks a left out block one place higher fails both assertions of the server."""
    share, up, down, tied, lift = decoy_summary(decoy_pairs, "planted")
    twice_share, _, _, _, twice_lift = decoy_summary(decoy_pairs, "twice")
    print(f"planted decoy in the server's place, {next(iter(decoy_pairs.values()))[0]['readings']} reading(s): up {up}, down {down}, tied {tied}, rise share {share:.4f} (bound "
          f"{(0.5 + twice_share) / 2:.4f}), mean mass change {lift:+.5f} (bound {twice_lift / 2:+.5f})")
    assert not share <= (0.5 + twice_share) / 2, (share, twice_share)
    assert not lift <= twice_lift / 2, (lift, twice_lift)


# ----- the time of a call, through CausalTools -----

class PlantedClock:
    """The clock of causal_tools in the deterministic time tests: perf_counter reads it, sleep moves it (every time for the
    server, never for the planted variant without padding, the first time alone for the planted padding of the first build), and
    every build of the belief's fits moves it by a planted time that falls with the blocks the build leaves out, as a real
    build's time does."""

    def __init__(self, sleeps="every"):
        self.now, self.sleeps, self.mode = 0.0, 0, sleeps

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps += 1
        if self.mode == "every" or (self.mode == "first" and self.sleeps == 1):
            self.now += seconds


def planted_build_seconds(left_out):
    """The planted time of a build that leaves out left_out blocks, the start's included."""
    return 0.1 - 0.01 * (left_out - 1)


@contextlib.contextmanager
def planted_clock(sleeps="every"):
    """causal_tools on a PlantedClock, every Models.without_longest moving it by its planted time."""
    clock, original = PlantedClock(sleeps), fitted_models.Models.without_longest

    def without_longest(self, start, also_without):
        fits = original(self, start, also_without)
        clock.now += planted_build_seconds(1 + len({tuple(h) for h in also_without} - {tuple(start)}))
        return fits

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(causal_tools_module, "time", clock)
        mp.setattr(fitted_models.Models, "without_longest", without_longest)
        yield clock


def never_played(blocks):
    """Development combinations beside the start, enough for every number of blocks left out, which no episode here plays."""
    return [h for h in development_starts(PROFILE, set(blocks)) if h != DEV_START][:max(TIMING_SIZES) - 1]


@pytest.fixture(scope="module")
def versions(tmp_path_factory, world):
    """The released evidence of one episode of the start after each of four readings: a reading of the start's block, a second one
    (the blocks left out unchanged), a reading of the repaired block (one block more) and a second one of it (unchanged again)."""
    cycles, blocks = world
    episode = FixtureEpisode(tmp_path_factory.mktemp("versions") / "episode", cycles, blocks, DEV_START, seed=0)
    out = []
    for group in ("tier1", "tier1"):
        episode.read(group)
        out.append(episode.released_path.read_text(encoding="utf-8"))
    order = episode.create_work_order("pump")
    while any(o["id"] == order for o in episode.open):
        episode.advance(1)
    for group in ("tier3", "tier1"):
        episode.read(group)
        out.append(episode.released_path.read_text(encoding="utf-8"))
    return out


def belief_spans(fits, folder, released_versions, served, sleeps):
    """The planted clock's time of CausalTools._belief_model on one CausalTools at each released version in turn, the first on a
    fresh CausalTools, which also times its reference build, with Models.served_blocks returning the blocks of served."""
    folder.mkdir(parents=True)
    path, spans = folder / "released.json", []
    with planted_clock(sleeps) as clock, pytest.MonkeyPatch.context() as mp:
        mp.setattr(fitted_models.Models, "served_blocks", lambda self, observations: list(served))
        tools = CausalTools(fits, path)
        for version in released_versions:
            path.write_text(version, encoding="utf-8")
            released = tools._released(note=False)
            started = clock.now
            tools._belief_model(released)
            spans.append(clock.now - started)
    return spans


def test_every_build_of_the_belief_takes_one_budget_whatever_the_blocks_left_out(tmp_path, world, models, versions):
    """On the planted clock, whatever the load of the machine: the first build of a CausalTools with its reference, and the next
    build at a new reading, each take one time for every number of blocks left out; the planted variant without the padding and
    the planted padding of the first build alone give next builds that follow the blocks left out."""
    fits, others = start_only(models, DEV_START), never_played(world[1])
    spans = {mode: [belief_spans(fits, tmp_path / f"{mode}_{k}", versions[:2], [DEV_START, *others[:k - 1]], mode) for k in TIMING_SIZES]
             for mode in ("every", "never", "first")}
    print({mode: [[round(x, 6) for x in pair] for pair in pairs] for mode, pairs in spans.items()})
    assert len({round(first, 9) for first, _ in spans["every"]}) == 1 and len({round(nxt, 9) for _, nxt in spans["every"]}) == 1
    assert len({round(nxt, 9) for _, nxt in spans["never"]}) == len(TIMING_SIZES)
    assert len({round(first, 9) for first, _ in spans["first"]}) == 1 and len({round(nxt, 9) for _, nxt in spans["first"]}) == len(TIMING_SIZES)


class LeftOutCache(CausalTools):
    """The planted leak of a persistent server: the fits of the belief are built again only when the blocks left out change, so
    a call is faster when a new reading leaves them unchanged."""

    def _belief_model(self, released):
        served = tuple(self.models.source.served_blocks(released.raw.get("observations", [])))
        if getattr(self, "_planted_served", None) == served and self._belief_backend is not None:
            return self._belief_backend
        self._planted_served, self._belief_key = served, None
        return super()._belief_model(released)


def persistent_rankings(fits, folder, released_versions, sleeps="every", tools_class=CausalTools, fresh=False):
    """[(raw answer, planted clock time)] of a ranking at each released version in one folder, by one CausalTools, or by a fresh
    one per call with fresh, as a server started for each call; the predictions each ranking stores carry over to the next."""
    folder.mkdir(parents=True)
    path, out = folder / "released.json", []
    with planted_clock(sleeps) as clock:
        tools = tools_class(fits, path)
        for version in released_versions:
            path.write_text(version, encoding="utf-8")
            tools = tools_class(fits, path) if fresh else tools
            started = clock.now
            result = tools.rank_root_causes(200, "summary")
            out.append((result.model_dump(mode="json"), clock.now - started))
    return out


def test_a_persistent_server_answers_as_one_per_call_and_pads_every_build(tmp_path, models, versions):
    """One CausalTools that serves four rankings, as the one server process of an episode of the model agents' rows does: its
    answers equal those of a fresh CausalTools per call, and every ranking after a new reading builds the belief's fits again
    and takes the same padded time, whether the reading left the blocks left out unchanged or added one. The planted cache keyed
    by the blocks left out and the planted padding of the first build alone are caught."""
    fits = start_only(models, DEV_START)
    one = persistent_rankings(fits, tmp_path / "one", versions)
    per_call = persistent_rankings(fits, tmp_path / "per_call", versions, fresh=True)
    cached = persistent_rankings(fits, tmp_path / "cached", versions, tools_class=LeftOutCache)
    first_only = persistent_rankings(fits, tmp_path / "first_only", versions, sleeps="first")
    spans = {name: [round(span, 9) for _, span in runs] for name, runs in (("one", one), ("cached", cached), ("first_only", first_only))}
    print(spans)
    assert [answer for answer, _ in one] == [answer for answer, _ in per_call]
    assert len(set(spans["one"][1:])) == 1 and spans["one"][1] > 0 and spans["one"][0] > spans["one"][1]
    assert len(set(spans["cached"][1:])) > 1 and len(set(spans["first_only"][1:])) > 1


def session_records(tmp, released_versions, plant):
    """The records of four rankings through open_sessions, one server process for the episode, at each released version in
    turn, with the recorder and the plant in the server, and whether the plant fired."""
    folder, records = tmp / "episode", tmp / "records"
    folder.mkdir(parents=True)
    records.mkdir()
    hook = tmp / "hook.py"
    hook.write_text(HOOK.replace("@RECORDS@", repr(str(records))).replace("@PLANT@", repr(plant)), encoding="utf-8")
    (folder / "released.json").write_text(released_versions[0], encoding="utf-8")
    launcher = write_launchers(folder, servers=(CAUSAL_SERVER,), left_out=DEV_START, models_path=fitted_models.models_path())[CAUSAL_SERVER]
    text = launcher.read_text(encoding="utf-8")
    launcher.write_text(text.replace(LAUNCH_LINE, f"exec(compile(open({str(hook)!r}).read(), {str(hook)!r}, 'exec'))\n{LAUNCH_LINE}"), encoding="utf-8")

    async def play_session():
        async with open_sessions({CAUSAL_SERVER: launcher}) as sessions:
            for version in released_versions:
                (folder / "released.json").write_text(version, encoding="utf-8")
                answer, is_error = await sessions.call(CAUSAL_SERVER, "rank_root_causes", {"top_k": 200, "branch_detail": "summary"})
                assert not is_error, answer

    asyncio.run(play_session())
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(records.glob("*.json"))], (records / "plant_fired").exists()


def test_one_server_per_episode_answers_as_one_per_call_and_builds_at_every_new_reading(tmp_path, models, versions):
    """Through open_sessions, as the model agents' rows start the servers: one process answers the four rankings as a fresh
    CausalTools per call answers them from what each read, and builds the belief's fits after every new reading, its reference
    with the first; the planted server that builds again only when the blocks left out change skips the builds of the readings
    that left them unchanged."""
    records, _ = session_records(tmp_path / "server", versions, "none")
    assert len(records) == len(versions) and len({r["pid"] for r in records}) == 1
    assert [r["builds"] for r in records] == [2, 1, 1, 1], [r["builds"] for r in records]
    again = [replay(r, tmp_path / "again" / f"{i:03d}", models, DEV_START) for i, r in enumerate(records)]
    assert [r["result"] for r in records] == [a["result"] for a in again]
    planted, fired = session_records(tmp_path / "planted", versions, "left_out_cache")
    assert fired and [r["builds"] for r in planted] != [2, 1, 1, 1], [r["builds"] for r in planted]


# the wall clock: the time of an estimate at the boundary of the tool, and of the build of the belief's fits on a primed server


def incremental_without_longest(self, start, also_without):
    """The planted build of the time test at the boundary of the tool: the fits of the belief built by leaving the blocks out one
    at a time, one build per block, so that the build takes longer the more blocks it leaves out."""
    also = [tuple(h) for h in also_without if tuple(h) != tuple(start)]
    fits = self.without(start)
    for k in range(1, len(also) + 1):
        fits = self.without(start, also[:k])
    return fits


def call_seconds(fits, released_path, served, variant):
    """Seconds of one estimate of monitor, the cheapest tool that builds the fits of the belief, by a fresh CausalTools on the
    same fits of the start, as a server started for each call meets it, with Models.served_blocks returning the blocks of served;
    variant incremental builds the fits of the belief one block at a time."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fitted_models.Models, "served_blocks", lambda self, observations: list(served))
        if variant == "incremental":
            mp.setattr(fitted_models.Models, "without_longest", incremental_without_longest)
        tools = CausalTools(fits, released_path)
        started = time.perf_counter()
        tools.estimate_intervention_effect("monitor")
        return time.perf_counter() - started, tools._pad_seconds


def build_seconds(tools, path, released_version, served, variant):
    """Seconds of the build of the belief's fits at the boundary of CausalTools._belief_model, where the padding sits, on a
    primed CausalTools whose last build read the other released version, so that it builds again and its reference is behind
    it; variant unpadded makes the sleep of the padding a no-op."""
    path.write_text(released_version, encoding="utf-8")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fitted_models.Models, "served_blocks", lambda self, observations: list(served))
        if variant == "unpadded":
            mp.setattr(causal_tools_module, "time", SimpleNamespace(perf_counter=time.perf_counter, sleep=lambda seconds: None))
        released = tools._released(note=False)
        started = time.perf_counter()
        tools._belief_model(released)
        return time.perf_counter() - started, tools._pad_seconds


@pytest.fixture(scope="module")
def timing(tmp_path_factory, world, models, versions):
    """({(variant, boundary): (blocks, sizes) seconds}, the same of the pad target): per boundary, TIMING_BLOCKS randomized
    complete blocks, each timing the server and the planted variant of that boundary at every number of blocks left out in
    TIMING_SIZES, in an order the seed shuffles. At the tool every estimate runs in a folder of its own on three readings of the
    start's block; at the build one primed CausalTools per variant alternates between two released versions."""
    cycles, blocks = world
    tmp = tmp_path_factory.mktemp("timing")
    episode = FixtureEpisode(tmp / "episode", cycles, blocks, DEV_START, seed=0)
    for group in ("tier1", "tier3", "tier1"):
        episode.read(group)
    released = episode.released_path.read_text(encoding="utf-8")
    others, fits, rng, seconds, pads, n = never_played(blocks), start_only(models, DEV_START), random.Random(1), {}, {}, 0

    def folder():
        nonlocal n
        n += 1
        path = tmp / f"{n:05d}"
        path.mkdir()
        (path / "released.json").write_text(released, encoding="utf-8")
        return path / "released.json"

    primed = {}
    for variant in ("server", "unpadded"):
        (tmp / variant).mkdir()
        primed[variant] = (CausalTools(fits, tmp / variant / "released.json"), tmp / variant / "released.json", [0])
        build_seconds(primed[variant][0], primed[variant][1], versions[0], [DEV_START], variant)
    call_seconds(fits, folder(), [DEV_START], "server")  # the first call of a process pays for its imports
    for boundary, planted_variant in (("call", "incremental"), ("build", "unpadded")):
        for variant in ("server", planted_variant):
            seconds[(variant, boundary)] = np.zeros((TIMING_BLOCKS[boundary], len(TIMING_SIZES)))
            pads[(variant, boundary)] = np.zeros((TIMING_BLOCKS[boundary], len(TIMING_SIZES)))
        for b in range(TIMING_BLOCKS[boundary]):
            cells = [(variant, j) for variant in ("server", planted_variant) for j in range(len(TIMING_SIZES))]
            rng.shuffle(cells)
            for variant, j in cells:
                served = [DEV_START, *others[:TIMING_SIZES[j] - 1]]
                if boundary == "call":
                    timed = call_seconds(fits, folder(), served, variant)
                else:
                    tools, path, last = primed[variant]
                    last[0] = 1 - last[0]
                    timed = build_seconds(tools, path, versions[last[0]], served, variant)
                seconds[(variant, boundary)][b, j], pads[(variant, boundary)][b, j] = timed
    return seconds, pads


def report(timing, variant, boundary):
    seconds, pads = timing[0][(variant, boundary)], timing[1][(variant, boundary)]
    p, slope = trend_p(seconds)
    print(f"{variant} at the {boundary}: p {p:.4g}, slope {slope * 1000:.3f} ms per block, median ms by blocks left out "
          f"{[round(float(m) * 1000, 2) for m in np.median(seconds, axis=0)]}, median pad target by blocks left out "
          f"{[round(float(m) * 1000, 2) for m in np.median(pads, axis=0)]} ms")
    return p, slope


def trend_p(seconds, seed=2):
    """(two sided p, slope in seconds per block) of the trend of the seconds over the number of blocks left out, within the
    randomized blocks: the statistic is the slope on the times centred per block, and the null draws permute the sizes within
    every block."""
    sizes = np.array(TIMING_SIZES, dtype=float) - np.mean(TIMING_SIZES)
    centred = seconds - seconds.mean(axis=1, keepdims=True)
    observed = float((centred * sizes).sum())
    rng = np.random.default_rng(seed)
    permuted = sizes[np.argsort(rng.random((PERMUTATIONS, *seconds.shape)), axis=2)]
    null = np.abs((centred[None] * permuted).sum(axis=(1, 2)))
    return (1 + int((null >= abs(observed)).sum())) / (PERMUTATIONS + 1), observed / (seconds.shape[0] * float((sizes ** 2).sum()))


def within_block_ranks(seconds):
    """The times of every randomized block replaced by their ranks within the block, 1 for the shortest."""
    return np.argsort(np.argsort(seconds, axis=1), axis=1).astype(float) + 1


def rank_trend_p(seconds):
    """The two sided p of the trend over the number of blocks left out on the ranks within every block, by the permutation test
    of trend_p, so that a few slow calls cannot carry the statistic."""
    return trend_p(within_block_ranks(seconds))[0]


def median_block_slope(seconds):
    """The median over the randomized blocks of the least squares slope of a block's times on the number of blocks left out, in
    seconds per block."""
    sizes = np.array(TIMING_SIZES, dtype=float) - np.mean(TIMING_SIZES)
    return float(np.median((seconds - seconds.mean(axis=1, keepdims=True)) @ sizes / float((sizes ** 2).sum())))


def rank_report(timing, variant, boundary):
    """(rank p, median slope) of the variant at the boundary, printed beside the mean statistics of report."""
    report(timing, variant, boundary)
    seconds = timing[0][(variant, boundary)]
    p, slope = rank_trend_p(seconds), median_block_slope(seconds)
    print(f"{variant} at the {boundary}: rank p {p:.4g}, median slope {slope * 1000:.3f} ms per block")
    return p, slope


def inconclusive_unless_caught(planted_p, plant):
    """Skips the timing test, with the reason, when the plant timed in the same blocks shows no trend at ALPHA: the load of the
    machine then leaves the run no power to see a leak, so it carries no verdict, which a failure would report as a leak."""
    if planted_p > ALPHA:
        pytest.skip(f"inconclusive: {plant} was not caught (rank p {planted_p:.4g} above ALPHA {ALPHA}), so this run carries no verdict")


def test_the_padded_build_does_not_follow_the_number_of_blocks_left_out(timing):
    """On the wall clock, at the boundary of the build where the padding sits, by the rank test within blocks: the planted
    unpadded build timed in the same blocks shows its trend at ALPHA, without which the clean verdict carries no information and
    the test is skipped as inconclusive; then the padded build shows none at ALPHA, and its median slope per block is at most
    half the planted one."""
    planted_p, planted_slope = rank_report(timing, "unpadded", "build")
    inconclusive_unless_caught(planted_p, "the planted unpadded build")
    p, slope = rank_report(timing, "server", "build")
    assert p > ALPHA and abs(slope) <= abs(planted_slope) / 2, (p, slope, planted_slope)


def test_the_time_of_a_call_does_not_follow_the_number_of_blocks_left_out(timing):
    """On the wall clock, at the boundary of the tool, by the rank test within blocks: the planted build one block at a time shows
    its trend at ALPHA, without which the clean verdict carries no information and the test is skipped as inconclusive; then the
    estimate shows none at ALPHA. The test sets no bound on the size of the slope at this boundary."""
    planted_p, planted_slope = rank_report(timing, "incremental", "call")
    inconclusive_unless_caught(planted_p, "the planted build one block at a time")
    p, slope = rank_report(timing, "server", "call")
    assert p > ALPHA, (p, slope)



# the process time of a call outside the padded window

def lazy_e6_timed_build(self, source, blocks):
    """The planted variant of the work outside the padded window: the build as the server made it before the E6 resampled tables
    were drawn inside it, so that the replay draws them after the padded build, and their cost, which follows the blocks the fits
    keep, falls outside the window."""
    started = causal_tools_module.time.perf_counter()
    backend = diagnostics_backend(EffectDiagnostics(source.without_longest(self.models.start, blocks), block=self.ranking_model != "joint"))
    return backend, causal_tools_module.time.perf_counter() - started


def outside_seconds(fits, released_path, served, variant, call):
    """Process seconds of call(tools), one tool call by a fresh CausalTools, outside CausalTools._belief_model, the padded window
    with its reference build, with Models.served_blocks returning the blocks of served. The sleep of the padding, which lies
    inside the window and takes no process time, is a no-op here to save wall time. Variant lazy_e6 builds as the server did
    before the E6 tables moved into the build."""
    inside, belief_model = [0.0], CausalTools._belief_model

    def timed_belief_model(self, released):
        started = time.process_time()
        try:
            return belief_model(self, released)
        finally:
            inside[0] += time.process_time() - started

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fitted_models.Models, "served_blocks", lambda self, observations: list(served))
        mp.setattr(CausalTools, "_belief_model", timed_belief_model)
        mp.setattr(causal_tools_module, "time", SimpleNamespace(perf_counter=time.perf_counter, sleep=lambda seconds: None))
        if variant == "lazy_e6":
            mp.setattr(CausalTools, "_timed_build", lazy_e6_timed_build)
        tools = CausalTools(fits, released_path)
        started = time.process_time()
        call(tools)
        return time.process_time() - started - inside[0]


def outside_cells(tmp, files, fits, others, rng, call):
    """{variant: (blocks, sizes) process seconds of call outside the padded window}: TIMING_BLOCKS["outside"] randomized complete
    blocks, each timing the server and the planted variant at every number of blocks left out in TIMING_SIZES, in an order rng
    shuffles, every call in a folder of its own that holds files {name: text}."""
    n = 0

    def folder():
        nonlocal n
        n += 1
        path = tmp / f"{n:05d}"
        path.mkdir()
        for name, text in files.items():
            (path / name).write_text(text, encoding="utf-8")
        return path / "released.json"

    outside_seconds(fits, folder(), [DEV_START], "server", call)  # the first call of a process pays for its imports
    seconds = {variant: np.zeros((TIMING_BLOCKS["outside"], len(TIMING_SIZES))) for variant in ("server", "lazy_e6")}
    for b in range(TIMING_BLOCKS["outside"]):
        cells = [(variant, j) for variant in seconds for j in range(len(TIMING_SIZES))]
        rng.shuffle(cells)
        for variant, j in cells:
            seconds[variant][b, j] = outside_seconds(fits, folder(), [DEV_START, *others[:TIMING_SIZES[j] - 1]], variant, call)
    return seconds


@pytest.fixture(scope="module")
def outside_timing(tmp_path_factory, world, models):
    """The process seconds of outside_cells for estimates of monitor, the seed 3 shuffling the cells, every estimate on three
    readings of the start's block."""
    cycles, blocks = world
    tmp = tmp_path_factory.mktemp("outside")
    episode = FixtureEpisode(tmp / "episode", cycles, blocks, DEV_START, seed=0)
    for group in ("tier1", "tier3", "tier1"):
        episode.read(group)
    files = {"released.json": episode.released_path.read_text(encoding="utf-8")}
    return outside_cells(tmp, files, start_only(models, DEV_START), never_played(blocks), random.Random(3),
                         lambda tools: tools.estimate_intervention_effect("monitor"))


def test_the_work_of_a_call_outside_the_padded_window_does_not_follow_the_number_of_blocks_left_out(outside_timing):
    """The process time of an estimate outside the padded window, by the rank test within blocks: the planted variant, whose
    replay draws the E6 tables after the padded build, shows its trend over the number of blocks left out at ALPHA, without which
    the clean verdict carries no information and the test is skipped as inconclusive; then the server shows none at ALPHA."""
    found = {}
    for variant in ("lazy_e6", "server"):
        found[variant] = rank_trend_p(outside_timing[variant])
        p, slope = trend_p(outside_timing[variant])
        medians = [round(float(m) * 1000, 2) for m in np.median(outside_timing[variant], axis=0)]
        print(f"{variant} outside the padded window: rank p {found[variant]:.4g}, p {p:.4g}, slope {slope * 1000:.3f} ms per block, "
              f"median process ms by blocks left out {medians}")
    inconclusive_unless_caught(found["lazy_e6"], "the planted lazy E6 draw")
    assert found["server"] > ALPHA, found["server"]


@pytest.fixture(scope="module")
def verify_outside_timing(tmp_path_factory, world, models):
    """The process seconds of outside_cells for verifications with a confirmation, the seed 4 shuffling the cells: every call
    verifies the prediction of hand_played from the released evidence and the predictions that its verification read."""
    tmp = tmp_path_factory.mktemp("verify_outside")
    record = hand_played(world, models, tmp / "hand").records[-1]
    assert record["tool"] == "verify_outcome" and record["result"]["confirmed"] is not None, record["result"]["status"]
    files = {"released.json": record["released"], "predictions.json": record["predictions_before"]}
    return outside_cells(tmp, files, start_only(models, DEV_START), never_played(world[1]), random.Random(4),
                         lambda tools: tools.verify_outcome(record["args"][0]))


def test_the_work_of_a_verification_outside_the_padded_window_is_measured(verify_outside_timing):
    """A measurement, not a check: the trend of the process time of a verification with a confirmation outside the padded window
    over the number of blocks left out, for the server and the planted variant whose replay draws the E6 tables after the padded
    build, by the rank test within blocks and by the mean statistic. It asserts nothing; the deterministic check of the state
    built after a build asserts."""
    for variant in ("lazy_e6", "server"):
        seconds = verify_outside_timing[variant]
        p, slope = trend_p(seconds)
        medians = [round(float(m) * 1000, 2) for m in np.median(seconds, axis=0)]
        print(f"{variant} outside the padded window of a verification: rank p {rank_trend_p(seconds):.4g}, p {p:.4g}, slope "
              f"{slope * 1000:.3f} ms per block, median process ms by blocks left out {medians}")


# ----- state of the belief's fits built after their build, during a verification -----

ATOMS = (type(None), bool, int, float, complex, str, bytes, np.generic)


def signature_and_children(node):
    """(signature, children) of an object of a graph: the type and size of a container, the attribute names and atomic attribute
    values of an object, the shape and type of an array; the children are what a container holds, the attributes of an object
    and the cells of a function's closure."""
    if isinstance(node, np.ndarray):
        return ("array", node.shape, str(node.dtype)), ()
    if isinstance(node, dict):
        return ("dict", len(node), tuple(sorted(map(repr, node)))), list(node.values())
    if isinstance(node, (list, tuple, set, frozenset)):
        return (type(node).__name__, len(node)), list(node)
    if isinstance(node, types.FunctionType):
        cells = [cell.cell_contents for cell in node.__closure__ or ()]
        return ("function", node.__qualname__, len(cells)), cells
    attributes = None if isinstance(node, (types.ModuleType, type)) else getattr(node, "__dict__", None)
    if attributes is None:
        return (type(node).__qualname__,), ()
    atoms = tuple(sorted((key, repr(value)) for key, value in attributes.items() if isinstance(value, ATOMS)))
    return (type(node).__qualname__, tuple(sorted(attributes)), atoms), list(attributes.values())


def graph(roots, skip):
    """({id: signature}, the objects) of every object reachable from roots, but not through an object whose id is in skip; the
    objects are returned so that they stay alive and no id is reused."""
    found, kept, stack = {}, [], list(roots)
    while stack:
        node = stack.pop()
        if isinstance(node, ATOMS) or id(node) in skip or id(node) in found:
            continue
        found[id(node)], children = signature_and_children(node)
        kept.append(node)
        stack.extend(children)
    return found, kept


def built_after_the_build(fits, released_path, served, variant, call, skip):
    """(the E6 tables drawn on the belief's fits outside CausalTools._timed_build, [(id, signature at the end of the build,
    signature at the end of the call)] of every object of a build's fits that changed or appeared in between) of call(tools) by a
    fresh CausalTools, with Models.served_blocks returning served and the sleep of the padding a no-op. skip holds the ids of
    the shared objects, the models and the fits without the start's block, which no build makes. Variant lazy_e6 builds as the
    server did before the E6 tables moved into the build."""
    builds, resamplers, inside, outside = [], set(), [False], [0]
    timed_build = lazy_e6_timed_build if variant == "lazy_e6" else CausalTools._timed_build
    table = Resampler.table

    def watched_build(self, source, blocks):
        inside[0] = True
        try:
            backend, elapsed = timed_build(self, source, blocks)
        finally:
            inside[0] = False
        found, kept = graph([backend], skip)
        builds.append((backend, found, kept))
        resamplers.update(id(node) for node in kept if isinstance(node, Resampler))
        return backend, elapsed

    def watched_table(self, r):
        if not inside[0] and id(self) in resamplers:
            outside[0] += 1
        return table(self, r)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fitted_models.Models, "served_blocks", lambda self, observations: list(served))
        mp.setattr(causal_tools_module, "time", SimpleNamespace(perf_counter=time.perf_counter, sleep=lambda seconds: None))
        mp.setattr(CausalTools, "_timed_build", watched_build)
        mp.setattr(Resampler, "table", watched_table)
        call(CausalTools(fits, released_path))
    changed = []
    for backend, before, _kept in builds:
        after, _ = graph([backend], skip)
        changed += [(key, before.get(key), after.get(key)) for key in set(before) | set(after) if before.get(key) != after.get(key)]
    return outside[0], changed


@pytest.mark.parametrize("case", ["hand", "confirms", "guard_refuses"])
def test_no_state_of_the_belief_fits_is_built_after_their_build_during_a_verification(world, models, tmp_path, case):
    """Each verification of the hand-played episode asked twice and of the first repairs of REPAIRS, by a fresh CausalTools at
    every number of blocks left out in TIMING_SIZES (the start and development blocks the episode never played): the planted
    variant, whose replay draws the E6 tables after the padded build, trips both checks at every size, without which a clean
    verdict carries no information and the test fails; then the server draws no E6 table of the belief's fits outside
    CausalTools._timed_build, and no object of those fits changes or appears between the end of its build and the end of the
    call."""
    play = verified_episode(case, world, models, tmp_path / "played")
    verifications = [r for r in play.records if r["tool"] == "verify_outcome"]
    assert len(verifications) == 2 and all(r["result"]["confirmed"] is not None for r in verifications)
    played = {play.start, *play.combination_of.values()}
    fits = start_only(models, play.start)
    others = [h for h in development_starts(PROFILE, set(world[1])) if h not in played][:max(TIMING_SIZES) - 1]
    skip = set(graph([models, fits], set())[0])
    found = {}
    for v, record in enumerate(verifications):
        for variant in ("lazy_e6", "server"):
            for k in TIMING_SIZES:
                folder = tmp_path / f"{v}_{variant}_{k}"
                folder.mkdir()
                (folder / "released.json").write_text(record["released"], encoding="utf-8")
                (folder / "predictions.json").write_text(record["predictions_before"], encoding="utf-8")
                found[(v, variant, k)] = built_after_the_build(fits, folder / "released.json", [play.start, *others[:k - 1]], variant,
                                                               lambda tools: tools.verify_outcome(record["args"][0]), skip)
                print(f"{case}, verification {v + 1}, {variant}, {k} blocks left out: {found[(v, variant, k)][0]} E6 tables drawn after a "
                      f"build, {len(found[(v, variant, k)][1])} objects changed or new")
    planted = [key for key in found if key[1] == "lazy_e6"]
    assert all(found[key][0] and found[key][1] for key in planted), ("the planted lazy E6 draw was not caught at every size, so the "
                                                                    "clean verdict is inconclusive", {key: found[key][0] for key in planted})
    assert all(found[key] == (0, []) for key in found if key[1] == "server"), {key: found[key] for key in found if key[1] == "server"}
