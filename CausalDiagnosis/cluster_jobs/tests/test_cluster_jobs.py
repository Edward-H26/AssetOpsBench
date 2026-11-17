"""The cluster scripts of CausalDiagnosis/cluster_jobs/: every script parses, the grid runner plays the reference runs that the design implies, plays
them alone as a CPU job on request and hands the checks of its runs to cluster_run_checks.py. The parts of run_grid.sbatch are run in
bash with stand-ins for python, sacct and the functions around them, so no model, no Slurm and no episode takes part."""
from CausalDiagnosis.utils.paths import CLUSTER_JOBS, GRIDS, PACKAGE, ROOT
import hashlib
import http.server
import importlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading

import pytest
from pathlib import Path

from CausalDiagnosis.experiments.table_definitions import experiment_tables as design
from CausalDiagnosis.training import gepa_prompt_optimization as gepa_online
from CausalDiagnosis.experiments.cluster_runs import grid_files, grid_job_lists as grid_lines, run_lines
from CausalDiagnosis.experiments.cluster_runs import pending_grid_lines as pending_lines
from CausalDiagnosis.experiments.cluster_runs import cluster_run_checks as run_checks, run_health_checks
from CausalDiagnosis.experiments.episode_runners import run_episodes
from CausalDiagnosis.experiments.table_definitions import training_variants as training
from CausalDiagnosis.utils import paths
from CausalDiagnosis.experiments.cluster_runs.tests.test_run_checks_overlap import _flush as overlap_flush, _killed as overlap_killed, _run as overlap_run

RUNNER = (CLUSTER_JOBS / "3_run" / "run_grid.sbatch").read_text(encoding="utf-8")
README = (CLUSTER_JOBS / "README.md").read_text(encoding="utf-8")
PASTE_WIDTH = 78  # the longest command line of the page, so that a pasted line never wraps
# the scripts of the chain that the run order replaced
REMOVED = ("agents_grid.sh", "anchored_then_gepa_then_grid.sh", "run_anchored.sbatch", "run_gepa.sbatch")
# the folder of the scripts, its numbered step folders and grids, the folders that the page lists
FOLDERS = [CLUSTER_JOBS, *sorted(p for p in CLUSTER_JOBS.iterdir() if p.is_dir() and re.fullmatch(r"[0-9]_\w+|grids", p.name))]
# the folder cluster_jobs of every framework, whose job_services.sh functions.sh sources and whose scripts submit.sh finds
FRAMEWORK_FOLDERS = sorted((PACKAGE / "frameworks").glob("*/cluster_jobs"))
# every script of those folders but the env.sh of one person, so that a clone collects the tests of its commit alone
SCRIPTS = sorted(p for folder in [*FOLDERS, *FRAMEWORK_FOLDERS] for p in folder.iterdir() if p.suffix in (".sh", ".sbatch") and p.name != "env.sh")
# the place of every script by its file name, relative to the folder of the scripts, so that a framework's script is reached
# from the folder of a tree as from this one
PLACE = {p.name: Path(os.path.relpath(p, CLUSTER_JOBS)) for p in SCRIPTS}
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _between(start, end):
    """The lines of run_grid.sbatch from the one that starts with start to the first later line that equals end."""
    lines = RUNNER.splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith(start))
    last = next(i for i in range(first, len(lines)) if lines[i] == end)
    return "\n".join(lines[first:last + 1])


def _bash(tmp_path, *parts):
    return subprocess.run(["bash", "-c", "\n".join(parts)], capture_output=True, text=True, cwd=tmp_path, env={"PATH": os.environ["PATH"]})


def _function(text, name):
    """The definition of the shell function name in a script, from its first line to the first later line that is a lone brace."""
    lines = text.splitlines()
    first = lines.index(f"{name}() {{")
    return "\n".join(lines[first:lines.index("}", first) + 1])


ENV_SCRIPT = (CLUSTER_JOBS / "env.example.sh").read_text(encoding="utf-8")
FUNCTIONS = (CLUSTER_JOBS / "functions.sh").read_text(encoding="utf-8")  # the functions of every job, which env.sh sources
# the served name of the model of the template, SERVED_NAME of env.sh, which run_lines.sh passes to pending_grid_lines.py
TEMPLATE_SERVED_NAME = re.search(r"^MODEL=\$\{MODEL:-(\S+)\}", ENV_SCRIPT, re.M).group(1).rsplit("/", 1)[-1]
# the scheduler of the runner, play_scheduled and the functions it calls, with its places empty and a short poll
SCHEDULER = ("SLOT_PIDS=(); SLOT_KEYS=(); SLOT_KINDS=(); SLOT_NAMES=(); SLOT_POLL=0.05",
             *(_function(RUNNER, name) for name in ("in_flight", "start_slot", "reap_slot", "service_answers", "service_down", "play_phase",
                                                    "play_scheduled", "clocked_lines", "gpu_snapshot")))
# the tracked serving file of record, which model_revision of env.sh reads, and its pins {Hugging Face id: commit}
SERVING_FILE = PACKAGE / "experiments" / "table_definitions" / "serving_of_record.json"
PINS = {block["hub_id"]: block["weights_commit"] for role, block in json.loads(SERVING_FILE.read_text(encoding="utf-8")).items() if role in ("agent", "judge")}
# a stand-in of run_ledger/: record-check prints FAKE_RECORD or the verdict it was written with, serving --check exits FAKE_SERVING
LEDGER_STAND_IN = """import os, sys
if sys.argv[1:2] == ["record-check"]:
    print(os.environ.get("FAKE_RECORD", "{verdict}"))
elif sys.argv[1:2] == ["serving"]:
    print("SERVING_OK agent" if os.environ.get("FAKE_SERVING", "0") == "0" else "SERVING_DIFFERS agent")
    sys.exit(int(os.environ.get("FAKE_SERVING", "0")))
"""


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_every_script_of_the_folder_parses(script):
    done = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def _runnable(module):
    """Whether python -m runs the module of the tree: a file module.py or a package with __main__.py."""
    path = ROOT / module.replace(".", "/")
    return path.with_suffix(".py").is_file() or (path / "__main__.py").is_file()


def test_every_module_that_a_script_runs_and_every_name_that_it_imports_resolve_in_the_tree():
    """Every module that a script of the folders runs with python -m is a module or a package with __main__.py of this tree, and
    every module that the Python of a script imports a name from holds that name, so a module that a move or a split of the tree
    left behind stops here and not in a job on the cluster."""
    run, imported = set(), set()
    for script in SCRIPTS:
        text = script.read_text(encoding="utf-8")
        run.update(re.findall(r"-m (CausalDiagnosis(?:\.\w+)+)", text))
        imported.update((module, name.strip()) for module, names in re.findall(r"\bfrom ([A-Za-z_][\w.]*) import ([\w, ]+)", text)
                        for name in names.split(","))
    assert run and sorted(module for module in run if not _runnable(module)) == []
    missing = []
    for module, name in sorted(imported):
        try:
            found = importlib.util.find_spec(module) is not None and hasattr(importlib.import_module(module), name)
        except ImportError:
            found = False
        if not found:
            missing.append(f"{module}.{name}")
    assert imported and missing == []


def test_the_reference_runs_take_their_presets_from_the_grid_file_and_their_agents_from_the_design():
    """No list of presets or of reference policies is written into the runner: both come from grid_job_lists.py, so a preset or a
    policy that experiment_tables.py gains is played without an edit here."""
    assert 'REFERENCE_CONFIGS=${GRID_REFERENCE_CONFIGS:-$(python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists presets "$GRID_PATH")}' in RUNNER
    assert "REFERENCE_AGENTS=$(python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists policies)" in RUNNER
    assert "WIRING_AGENTS=$(python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists wiring)" in RUNNER
    assert "for agent in $REFERENCE_AGENTS; do" in RUNNER and not re.search(r"for agent in [a-z_ ]+;", RUNNER)


def test_every_reference_agent_plays_every_preset_and_replicate_into_a_log_of_its_own(tmp_path):
    """One run per preset, reference agent and replicate, the clock mode of an entry after its colon, marked as a reference run
    for the ledger, each with its output in a log beside the run folders and its start and end in in_flight.tsv under the stem of
    that log. The agent of the wiring check plays the first replicate alone, since one run per preset proves the wiring. No preset
    charges the wall seconds here, so no filler starts."""
    python = 'python() { case "$*" in *" clocked "*) ;; *) echo "$HYD_GRID_LINE $*" ;; esac; }'
    done = _bash(tmp_path, python, re.search(r"^replicate_option\(\) \{.*\}$", RUNNER, re.M).group(0), *SCHEDULER,
                 *(_function(RUNNER, name) for name in ("wiring_skipped", "reference_name", "start_reference",
                                                        "wall_reference", "reference_rows")),
                 'OUT=. SEEDS="--seeds 3 --first-seed 0 --starts sampled" REPLICATES="1 2" GRID_PARALLEL=1000',
                 'REFERENCE_CONFIGS="base timed:instant" REFERENCE_AGENTS="oracle random scripted" WIRING_AGENTS=scripted', "reference_rows")
    assert done.returncode == 0 and done.stdout == "", done.stderr
    played = [(name, options, agent, r) for name, options in (("base", "base"), ("timed_instant", "timed --clock-mode instant"))
              for agent, replicates in (("oracle", (1, 2)), ("random", (1, 2)), ("scripted", (1,))) for r in replicates]
    flight = [line.split("\t") for line in (tmp_path / "in_flight.tsv").read_text(encoding="utf-8").splitlines()]
    assert sorted(row[1] for row in flight) == ["end"] * len(played) + ["start"] * len(played)
    assert all(len(row) == 7 and row[2:4] == ["reference", "run"] and row[6] == "0" for row in flight), flight
    assert sorted(row[4] for row in flight if row[1] == "start") == sorted(f"reference_{name}_{agent}_replicate_{r}" for name, _o, agent, r in played)
    assert not (tmp_path / "fillers_test").exists()
    logs = {p.name: p.read_text(encoding="utf-8").strip() for p in tmp_path.glob("reference_*.log")}
    runner = "reference -m CausalDiagnosis.experiments.episode_runners.run_episodes"
    assert logs == {f"reference_{name}_{agent}_replicate_{r}.log": f"{runner} --config {options} --agent {agent} --seeds 3 --first-seed 0 --starts sampled --replicate {r} --out ."
                    for name, options, agent, r in played}


def test_a_grid_line_of_the_wiring_check_plays_the_first_replicate_alone(tmp_path):
    """The runs of a grid file are its lines times the replicates of the job, a line of the wiring check once, on the first
    replicate of the test set, as pending_grid_lines.py counts it, and a line that names its replicate once."""
    units = "\n".join([_function(RUNNER, "wiring_skipped"), _function(RUNNER, "line_agent"), _between("UNITS=()", "done"),
                       'printf "%s\\n" "${UNITS[@]}"'])
    lines = 'LINES=("--config base --agent oracle" "--config base --servers aob --agent scripted --no-diagnostics" "--config base --agent causal --replicate 3")'
    done = _bash(tmp_path, lines, 'REPLICATES="1 2" WIRING_AGENTS=scripted', units)
    assert done.returncode == 0 and done.stdout.split("\n")[:-1] == ["0 1", "1 1", "2 none", "0 2"], done.stderr
    done = _bash(tmp_path, lines, 'REPLICATES=none WIRING_AGENTS=scripted', units)
    assert done.returncode == 0 and done.stdout.split("\n")[:-1] == ["0 none", "1 none", "2 none"], done.stderr


def test_the_reference_agents_alone_wait_for_no_endpoint_and_play_no_line(tmp_path):
    """GRID_REFERENCE=only is the CPU job of the reference agents: no endpoint is looked up, since no model plays, and no line
    of the grid runs. GRID_REFERENCE=0 plays the lines without the reference agents, and the default plays both."""
    endpoints = "\n".join([_function(RUNNER, "reference_line"), _function(RUNNER, "model_lines"),
                           _between("# the reference agents alone wait for no endpoint", "fi")])
    flow = "\n".join([*SCHEDULER, *(_function(RUNNER, name) for name in ("line_name", "start_line", "wall_line", "play_units", "play_grid")),
                      "play_grid"])
    stand_ins = ('endpoint_url() { echo LOOKUP >&2; echo http://node/v1; }', 'judge_args() { echo "--judge local"; }',
                 'REFERENCE_AGENTS="oracle do_nothing scripted"',
                 'reference_rows() { echo REFERENCE; }', 'run_unit() { echo "UNIT $1"; }', 'job_checks() { echo "CHECKS $1"; }',
                 'unit_statuses() { :; }', 'reference_statuses() { :; }', 'python() { echo "PY $*"; }', 'OUT=. ENDPOINTS=. SERVED_NAME=m GRID_PARALLEL=1000',
                 'LINES=("--config base" "--config priced") UNITS=("0 1" "1 1")')
    played = {}
    for mode in ("only", "0", "1"):
        done = _bash(tmp_path, *stand_ins, f"GRID_REFERENCE={mode}", endpoints, 'echo "URL [$BASE_URL] JUDGE [$JUDGE]"', flow)
        assert done.returncode == 0, done.stderr
        logs = sorted(p.read_text(encoding="utf-8").strip() for p in tmp_path.glob("line_*.log"))
        played[mode] = (done.stdout.splitlines(), "LOOKUP" in done.stderr, logs)
        for path in tmp_path.glob("line_*.log"):
            path.unlink()
    tail = ["CHECKS ."]
    assert played["only"] == (["URL [] JUDGE []", "REFERENCE", *tail], False, [])
    for mode, reference in (("0", []), ("1", ["REFERENCE"])):
        printed, looked_up, logs = played[mode]
        starts = [line for line in printed if line.startswith("### ") and "start grid run" in line]  # they carry the date
        assert [line for line in printed if line not in starts] == ["URL [http://node/v1] JUDGE [--judge local]", *reference, *tail]
        assert looked_up and len(starts) == 2 and logs == ["UNIT 0 1", "UNIT 1 1"]


def test_a_line_of_a_reference_agent_plays_without_the_model_so_its_pending_pairs_play_as_a_cpu_job(tmp_path):
    """The pending reference pairs that pending_grid_lines.py prints from grid_job_lists.py references PART play as a CPU job with
    GRID_REFERENCE=0: a line that names a reference agent runs without the model and the judge, a grid file of such lines alone
    looks up no endpoint, and a line of a model agent beside them still plays on the served model with its judge."""
    endpoints = "\n".join([_function(RUNNER, "reference_line"), _function(RUNNER, "model_lines"),
                           _between("# the reference agents alone wait for no endpoint", "fi")])
    stand_ins = ('endpoint_url() { echo LOOKUP >&2; echo http://node/v1; }', 'judge_args() { echo "--judge local"; }',
                 'REFERENCE_AGENTS="oracle do_nothing scripted"', "ENDPOINTS=. SERVED_NAME=m GRID_REFERENCE=0")
    reference = ('"--config priced --agent oracle --replicate 2" "--config real_time_token_clock --agent do_nothing --replicate 1"')
    looked_up = {}
    for name, lines in (("reference", reference), ("mixed", f'{reference} "--config priced --agent causal --no-diagnostics"'), ("none", "")):
        done = _bash(tmp_path, *stand_ins, f"LINES=({lines})", "set -u", endpoints, 'echo "URL [$BASE_URL] JUDGE [$JUDGE]"')
        assert done.returncode == 0, done.stderr
        looked_up[name] = (done.stdout.strip(), "LOOKUP" in done.stderr)
    assert looked_up == {"reference": ("URL [] JUDGE []", False), "mixed": ("URL [http://node/v1] JUDGE [--judge local]", True),
                         "none": ("URL [] JUDGE []", False)}
    driver = 'python() { echo "$HYD_GRID_LINE $*" > "$PWD/call_$HYD_GRID_LINE"; }'
    units = (driver, "hostname() { echo node7; }", 'REFERENCE_AGENTS="oracle do_nothing scripted"',
             re.search(r"^replicate_option\(\) \{.*\}$", RUNNER, re.M).group(0), _function(RUNNER, "reference_line"), _function(RUNNER, "run_unit"),
             f'LINES=({reference} "--config priced --agent causal --no-diagnostics")',
             'OUT=. SERVED_NAME=m BASE_URL=http://node/v1 JUDGE="--judge local" SEEDS="--seeds 1"')
    done = _bash(tmp_path, *units, 'run_unit "0 none"; run_unit "1 none"; run_unit "2 3"')
    assert done.returncode == 0, done.stderr
    runner = "-m CausalDiagnosis.experiments.episode_runners.run_episodes"
    assert [(tmp_path / f"call_{i}").read_text(encoding="utf-8").strip() for i in range(3)] == [
        f"0 {runner} --config priced --agent oracle --replicate 2 --seeds 1 --out .",
        f"1 {runner} --config real_time_token_clock --agent do_nothing --replicate 1 --seeds 1 --out .",
        f"2 {runner} --config priced --agent causal --no-diagnostics --model m --base-url http://node/v1 --judge local --seeds 1 --replicate 3 --out ."]


def test_a_selector_reaches_the_driver_as_one_word_through_grid_extra_and_a_grid_line(tmp_path):
    """--start-state and --scenario-ids are one word each, the levels and the ids joined by /, so the comma list of --export, as the
    page passes GRID_EXTRA, and the word splitting of run_grid.sbatch hand each to the driver whole, from GRID_EXTRA and from a line
    of the grid file alike, and the driver's parser reads it."""
    export = "ALL,RUN_AFTER_SERVE=$H/3_run/run_lines.sh,GRID_STARTS=seeds,GRID_EXTRA=--start-state=3/100/0/130@2"
    extra = dict(item.split("=", 1) for item in export.split(",") if "=" in item)["GRID_EXTRA"]
    driver = 'python() { shift 2; printf "%s\\n" "$@" > "$PWD/call_$HYD_GRID_LINE"; }'
    units = (driver, "hostname() { echo node7; }", 'REFERENCE_AGENTS="oracle"',
             re.search(r"^replicate_option\(\) \{.*\}$", RUNNER, re.M).group(0), _function(RUNNER, "reference_line"), _function(RUNNER, "run_unit"),
             'LINES=("--config base --agent causal" "--config anchored --agent oracle --scenario-ids 2001/2003")',
             'OUT=. SERVED_NAME=m BASE_URL=http://node/v1 JUDGE= SEEDS="--seeds 1 --starts seeds"')
    done = _bash(tmp_path, *units, f'GRID_EXTRA={extra} run_unit "0 none"; run_unit "1 none"')
    assert done.returncode == 0, done.stderr
    first, second = (run_episodes.build_parser().parse_args((tmp_path / f"call_{i}").read_text(encoding="utf-8").splitlines()) for i in range(2))
    assert (first.start_state, first.starts, first.scenario_ids) == ("3/100/0/130@2", "seeds", None)
    assert (second.scenario_ids, second.start_state, second.config) == ([2001, 2003], None, "anchored")


def test_the_checks_at_the_end_of_a_job_print_the_latency_keep_the_job_record_and_name_suspect_runs(tmp_path):
    """The job prints the latency of its model calls, keeps the record of its own job from
    Slurm's accounting beside its run folders, in the fields that cluster_run_checks.py reads the suspended time from, names the
    runs of a suspended or requeued job and runs the overlap check of its episodes on the wall clock. Outside Slurm no record
    is asked for."""
    stand_ins = ('python() { echo "PY $*"; }', 'sacct() { echo "SACCT $*" >&2; echo "77|RUNNING|2026-01-01T00:00:00|Unknown|00:10:00|00:01:40|node001"; }',
                 _function(FUNCTIONS, "write_job_record"), _between("job_checks() {", "}"))
    done = _bash(tmp_path, *stand_ins, "SLURM_JOB_ID=77 SLURM_RESTART_COUNT=2", "job_checks .")
    assert done.returncode == 0 and done.stdout.splitlines() == [
        "PY -m CausalDiagnosis.experiments.cluster_runs.cluster_run_checks latency .",
        "JOB_RECORD restarts 2: 77|RUNNING|2026-01-01T00:00:00|Unknown|00:10:00|00:01:40|node001 ",
        "PY -m CausalDiagnosis.experiments.cluster_runs.cluster_run_checks suspect .",
        "PY -m CausalDiagnosis.experiments.cluster_runs.cluster_run_checks overlap ."]
    assert done.stderr.strip() == "SACCT -j 77 -D -X -n -P -o JobID,State,Start,End,Elapsed,Suspended,NodeList"
    assert run_health_checks.suspended_seconds(tmp_path) == {"77": 100.0}
    (tmp_path / run_health_checks.JOB_RECORD).unlink()
    done = _bash(tmp_path, *stand_ins, "job_checks .")
    assert done.stdout.splitlines() == ["PY -m CausalDiagnosis.experiments.cluster_runs.cluster_run_checks latency .", "PY -m CausalDiagnosis.experiments.cluster_runs.cluster_run_checks suspect .",
                                        "PY -m CausalDiagnosis.experiments.cluster_runs.cluster_run_checks overlap ."]
    assert done.stderr == "" and not (tmp_path / run_health_checks.JOB_RECORD).exists()


def test_the_checks_at_the_end_of_a_job_name_a_run_alive_in_an_episode_on_the_wall_clock_and_an_unclean_flush(tmp_path):
    """job_checks runs the overlap check of cluster_run_checks.py over the run folders of the job, the check of this tree, so the log names
    a run alive during an episode of a preset with the wall clock and an episode whose flush of the model server's cache is not
    shown clean, each followed by OVERLAP_EXIT 1. The control: the runs of the job one after the other on a clean flush print no
    overlap and no exit line. The runs lie a day apart, so a time zone that the check reads otherwise moves no verdict."""
    module = f'PYTHONPATH="{os.pathsep.join(p for p in sys.path if p)}" "{sys.executable}" -m {run_checks.__name__}'
    stand_ins = (f'python() {{ case "$*" in *" overlap "*) {module} overlap "${{@: -1}}" ;; *) echo "PY $*" ;; esac; }}',
                 "write_job_record() { :; }", _between("job_checks() {", "}"))
    t0, day = 1_800_000_000, 86_400
    overlap_run(tmp_path, "wall", t0, t0 + 100, run_health_checks.WALL_CLOCK, episodes=[(t0 + 10, t0 + 50)])
    overlap_run(tmp_path, "later", t0 + day, t0 + day + 30)
    done = _bash(tmp_path, *stand_ins, "job_checks .")
    lines = done.stdout.splitlines()
    assert done.returncode == 0 and "OVERLAP_EXIT" not in done.stdout, done.stdout + done.stderr
    assert lines[-1] == "OVERLAP_EPISODES 0 overlaps, 0 episodes without a recorded interval, 0 episodes without a clean flush", lines
    overlap_killed(tmp_path, "filler", t0 - day, t0 + day)
    lines = _bash(tmp_path, *stand_ins, "job_checks .").stdout.splitlines()
    assert any(line.startswith("OVERLAP wall/seed000000 ") and " overlaps filler " in line for line in lines), lines
    assert lines[-1] == "OVERLAP_EXIT 1", lines
    shutil.rmtree(tmp_path / "filler")
    overlap_flush(tmp_path, first_request_cached_tokens=12)
    lines = _bash(tmp_path, *stand_ins, "job_checks .").stdout.splitlines()
    assert "OVERLAP_FLUSH wall/seed000000: its first model request reused 12 cached tokens, not 0" in lines, lines
    assert lines[-1] == "OVERLAP_EXIT 1", lines


def test_a_failed_overlap_check_ends_the_job_with_status_7_and_a_failed_run_with_5(tmp_path):
    """A run alive in an episode on the wall clock or an unclean flush voids the wall-clock runs that the overlap check names, so a
    job that plays to its end then ends with status 7 and without CD_GRID_DONE, through the tail of the script; a failed run ends
    it with 5 as before, the overlap line still in the log. The control: a clean check ends the job with 0 and CD_GRID_DONE."""
    tail = RUNNER[RUNNER.index("# a job that the time limit ends keeps its record"):]
    stand_ins = ('python() { case "$*" in *" overlap "*) return "$FAKE_OVERLAP" ;; *) echo "PY $*" ;; esac; }',
                 "write_job_record() { :; }", "gpu_snapshot() { :; }", "reference_rows() { :; }", "play_units() { :; }",
                 'unit_statuses() { return "$FAKE_UNITS"; }', _function(RUNNER, "job_checks"), _function(RUNNER, "play_grid"))
    for overlap, units, status in ((0, 0, 0), (1, 0, 7), (1, 5, 5)):
        done = _bash(tmp_path, *stand_ins, f"OUT=. GRID_REFERENCE=0 FAKE_OVERLAP={overlap} FAKE_UNITS={units}", tail)
        assert done.returncode == status, (overlap, units, done.returncode, done.stdout, done.stderr)
        assert ("OVERLAP_EXIT 1" in done.stdout) == bool(overlap) and ("CD_GRID_DONE ." in done.stdout) == (status == 0), done.stdout


def test_run_lines_plays_a_grid_file_and_leaves_the_loops_of_gepa_and_the_reference_agents_to_their_own_jobs():
    """run_lines.sh plays the lines of a grid file against the served model and nothing else: no loop of GEPA, which needs an
    agent and a fold and has run_gepa_online.sh, no reference agent unless GRID_REFERENCE=1, since they play as a CPU job, and
    no judge, which scores the runs after the grid. The scripts of the chain that the run order replaced are gone."""
    lines = (CLUSTER_JOBS / "3_run" / "run_lines.sh").read_text(encoding="utf-8")
    assert gepa_online.__name__ not in lines and "GRID_REFERENCE=${GRID_REFERENCE:-0}" in lines
    tables = (CLUSTER_JOBS / "0_setup" / "precompute_tables.sbatch").read_text(encoding="utf-8")
    grid_tables = _function(FUNCTIONS, "grid_tables")
    assert "grid_tables compute || exit 4\ngrid_tables check || exit 4\n" in tables
    assert 'for f in "$folder"/grid_*.txt; do grids+=(--grid "$f"); done' in grid_tables and "precompute_grid_tables" in grid_tables


def test_the_tables_of_the_fillers_are_computed_before_the_platform_record(tmp_path):
    """A filler plays its line on the development starts, whose fault rule keys a label table of its own, so the tables job
    computes the tables of every grid line on the development starts beside those of their own start set, and the platform
    record holds them; a run that wrote one into the data would differ from the tree that the record check accepted. The
    grid job asks for both sets before its first line, so a missing one stops it there."""
    (tmp_path / "grid_a.txt").write_text("--config base\n", encoding="utf-8")
    grids_here = 'tree_location() { [ "$1" = GRIDS ] && echo "$PWD"; }'  # the paths module of the tree names this folder
    writer = 'python() { echo "PY $CD_TABLES_READ_ONLY $*"; }'  # the call and whether it may write the tables
    done = _bash(tmp_path, writer, grids_here, _function(FUNCTIONS, "grid_tables"), f"FORK={tmp_path}", "grid_tables compute")
    script = "-m CausalDiagnosis.experiments.cluster_runs.precompute_grid_tables"
    grid_a = tmp_path / "grid_a.txt"
    assert done.returncode == 0 and done.stdout.splitlines() == [f"PY 0 {script} --grid-presets --grid {grid_a}",
                                                               f"PY 0 {script} --grid-presets --starts development --grid {grid_a}"], done.stderr
    # the check of the same tables writes nothing, and a launch of its own passes its grid file, start set and run options
    done = _bash(tmp_path, writer, grids_here, _function(FUNCTIONS, "grid_tables"), f"FORK={tmp_path}",
                 "GRID_FILE=grid_a.txt GRID_STARTS=development GRID_EXTRA=--start-cap=2 grid_tables check")
    assert done.returncode == 0 and done.stdout.splitlines() == [
        f"PY 1 {script} --check --grid-presets --grid {grid_a} --starts development --start-cap=2",
        f"PY 1 {script} --check --grid-presets --starts development --grid {grid_a} --start-cap=2"], done.stderr
    done = _bash(tmp_path, 'python() { case "$*" in *development*) return 1 ;; esac; }', grids_here, _function(FUNCTIONS, "grid_tables"),
                 f"FORK={tmp_path}", "grid_tables compute")
    assert done.returncode == 1  # a failed pass fails the tables job
    done = _bash(tmp_path, 'python() { echo "PY $*"; }', "tree_location() { return 2; }", _function(FUNCTIONS, "grid_tables"), "grid_tables")
    assert done.returncode == 4 and "PY" not in done.stdout  # a tree whose paths module does not answer computes no table
    assert [name for name in REMOVED if name in PLACE] == []


# ----- CausalDiagnosis/cluster_jobs/README.md, the page for whoever runs the jobs -----

def _commands():
    """Every line of the bash blocks of the page."""
    return [line for block in re.findall(r"```bash\n(.*?)```", README, re.S) for line in block.splitlines()]


def test_every_command_of_the_page_is_short_enough_to_paste():
    assert [line for line in _commands() if len(line) > PASTE_WIDTH] == []


def test_every_variable_of_a_command_is_set_on_an_earlier_line_of_the_page():
    """Whoever runs the jobs pastes the blocks of the page from the top, so every variable that a command reads is set on an earlier line
    of the page or comes from the login shell."""
    seen, unset = {"PWD", "PATH"}, []
    for line in _commands():
        unset += [f"{name}: {line}" for name in re.findall(r"(?<!\\)\$([A-Z][A-Z0-9_]*)", line) if name not in seen]
        seen.update(re.findall(r"^(?:export )?([A-Z][A-Z0-9_]*)=", line))
    assert unset == []


def test_the_page_lists_every_file_of_the_folder_and_its_commands_name_files_that_exist():
    """The table of files names every file of the folder, of its step folders and of grids by its path in the folder, and every
    path under $H, every script of a submit line and every module of a command exists."""
    names = sorted(p.relative_to(CLUSTER_JOBS).as_posix() for folder in FOLDERS for p in folder.iterdir()
                   if p.is_file() and p.name not in ("env.sh", "__init__.py") and not p.name.startswith("local_"))
    assert [name for name in names if f"| `{name}` |" not in README] == []
    named = {name for line in _commands() for name in re.findall(r"\$H/([A-Za-z0-9_./]+)", line)}
    assert named and sorted(name for name in named if name != "env.sh" and not (CLUSTER_JOBS / name).exists()) == []
    submitted = {line.split()[2] for line in _commands() if line.startswith("bash $H/submit.sh ")}
    assert submitted and sorted(name for name in submitted if name not in PLACE) == []
    modules = {name for line in _commands() for name in re.findall(r"\bCausalDiagnosis(?:\.\w+)+", line)}
    assert modules and sorted(name for name in modules if not _runnable(name)) == []


def test_the_page_states_the_counts_that_the_design_implies():
    """The structure counts of the page are written by a script from grid_job_lists.py, so a design that moves shows here until the
    page is built again: the run lines of every grid file in the table of files, and the lines, replicates and runs of every part
    of grid_job_lists.py in the section on the jobs of a cut. The page types no count of episodes, since the starts of one run are counted from the data file."""
    for name in grid_files.FILES:
        assert f"| `grids/{name}` | the {len(pending_lines.grid_lines(GRIDS / name))} run lines of " in README, name
    assert len(design.AGENT_ROWS) == len(run_lines.agents())
    for name in run_lines.PARTS:
        lines, replicates = len(run_lines.part(name)), run_lines.replicates(name)
        assert f"| `{name}` | {lines} | {replicates} | {lines * replicates} |" in README, name


def test_the_run_order_holds_every_step_in_its_order_with_the_script_of_each():
    """Section 3 gives the registered order from SETUP_OK in the order it is played, with no step by hand before it: the reference
    rows of every preset as a CPU job, the part main, the reproduction of record, the part companion, the part environment_token,
    the part environment, the judge over the registered folders, the rescoring and the offline table of the MCP, the layer study as
    the parts layers and layers_token on two cards at once, the top rung of the part ladder with GRID_PENDING=1, then the part
    ablations and the part misdiagnosis_price. Every job of it goes through submit.sh, so no line names a partition or a node."""
    order = README[README.index("\n### The registered order\n"):README.index("\n## 4. After a run\n")]
    steps = ["GRID_REFERENCE=only", "GRID_PART=main", "AOB_MODE=recovered", "RUN_AFTER_SERVE=$A/run_reproduction.sh", "GRID_PART=companion\n",
             "GRID_PART=environment_token\n", "GRID_PART=environment\n", "--export=$E,JUDGE_FOLDERS=<grid folders>\n", "python -m $RS <run folders>",
             "offline_table", "GRID_PART=layers\n", "GRID_PART=layers_token\n", "E=$P,GRID_PENDING=1\n", "--export=$E,GRID_PART=ladder\n",
             "GRID_PART=ablations\n", "GRID_PART=misdiagnosis_price\n"]
    places = [order.index(step) for step in steps]
    assert places == sorted(places)
    lines = [line for line in _commands() if line.startswith("bash $H/submit.sh")]
    assert lines and not [line for line in lines if re.search(r"(^| )(-p|-w|--partition|--nodelist|--exclude)( |=)", line)]


def _page_launches(section):
    """The submit lines of a part of section 3 of the page as (script, exports), the shell variables of its blocks resolved in their order
    and N, the runs in flight that a submit line names, set to 4."""
    body = README[README.index(f"\n### {section}\n") + 1:]
    body = body[:re.search(r"\n##+ ", body).start()]
    names, launches = {"N": "4"}, []
    for block in re.findall(r"```bash\n(.*?)```", body, re.S):
        for line in block.splitlines():
            line = re.sub(r"\$([A-Z]+)\b", lambda m: names.get(m.group(1), m.group(0)), line)
            if (assigned := re.fullmatch(r"([A-Z]+)=(\S+)", line)) and assigned.group(1) != "N":
                names[assigned.group(1)] = assigned.group(2)
            elif line.startswith("bash $H/submit.sh "):
                export = next((word[len("--export="):] for word in line.split() if word.startswith("--export=")), "")
                launches.append((line.split()[2], dict(item.split("=", 1) for item in export.split(",") if "=" in item)))
    return launches


def _launch_runs(script, exports):
    """(arguments, settings of run_episodes/) of every run that a launch plays, on the first replicate of the test set: the
    lines of a part as run_grid.sbatch composes them behind run_lines.sh, the reference runs of the CPU job of the baseline
    policies, and the loops of the job of GEPA as run_gepa_online.sh and gepa_online.driver_settings compose them."""
    after, starts = exports.get("RUN_AFTER_SERVE", ""), exports.get("GRID_STARTS", "sampled")
    seeds = ["--seeds", "3", "--first-seed", "0", "--starts", starts]
    replicate = ["--replicate", "1"] if starts == "sampled" else []
    if script == "serve_model.sbatch" and after.endswith("/run_lines.sh"):
        for _comment, line in run_lines.part(exports["GRID_PART"]):
            words = line.split()
            agent = [] if "--agent" in words else ["--agent", "plan_execute"]
            reference = words[words.index("--agent") + 1] in run_lines.REFERENCE_AGENTS if "--agent" in words else False
            model = [] if reference else ["--model", "m", "--base-url", "http://node/v1"]
            argv = [*words, *agent, *model, *seeds, *([] if "--replicate" in words else replicate), "--out", "x", *exports.get("GRID_EXTRA", "").split()]
            yield argv, run_episodes.build_parser().parse_args(argv)
    elif script == "run_grid.sbatch" and exports.get("GRID_REFERENCE") == "only":
        for entry in grid_files.presets(GRIDS / "grid_agents.txt"):
            preset, _colon, mode = entry.partition(":")
            for agent in run_lines.REFERENCE_AGENTS:
                argv = ["--config", preset, *(["--clock-mode", mode] if mode else []), "--agent", agent, *seeds, *replicate, "--out", "x"]
                yield argv, run_episodes.build_parser().parse_args(argv)
    elif script == "serve_model.sbatch" and after.endswith("/run_gepa_online.sh"):
        for agent in ("plan_execute", "agent_as_tool", "causal"):
            for fold in training.held_out_folds.fold_ids():
                for card in ("verifier", "score_only"):
                    argv = ["--model", "m", "--base-url", "http://node/v1", "--reflection-model", "j", "--reflection-base-url", "http://judge/v1",
                            "--agent", agent, "--training-variant", "as_released", "--training-fold", str(fold), "--card", card,
                            "--config", "deterioration", "--servers", "aob", "--store", "couchdb", "--out", "x", "--resume",
                            *(["--training-split", exports["GEPA_SPLIT"]] if exports.get("GEPA_SPLIT") else [])]
                    yield argv, gepa_online.driver_settings(gepa_online.build_parser().parse_args(argv))


def test_every_registered_row_registers_under_the_launch_that_the_page_documents(monkeypatch):
    """The wiring check of the registered order, with no model and one replicate: every run of the test set or of the training
    starts that a submit line of the registered order plays, and every run of the training path of the two further tables as the
    registered order plays it on the registered split, without GRID_EXTRA,
    GEPA_SPLIT and the folder of the trial, passes registration() with the tree, the files the driver loaded, the data, the open
    values and the endpoints taken as of record, every run on a preset with a clock at the one run in flight that the wall phase
    of its job hands it and every other run at the N of its launch, as the job hands them down. A part on a preset of no table,
    the healthy control of the part main among them, or an option of a trial in a registered launch fails here, before any job."""
    # set where registration() reads them, in the modules of the package run_episodes/, not on the package
    for module, name, value in ((run_episodes.start_check, "tree_reasons", lambda: []), (run_episodes.agent_kinds, "MODEL_AGENTS", frozenset()),
                                (run_episodes.start_check, "stamped_inputs", lambda a, config: [])):
        monkeypatch.setattr(module, name, value)
    # the rules that a tree of the tests cannot pass stand as of record: rule 3 asks git about the loaded files, which a copy
    # outside a repository cannot answer, rule 5 reads the data of the clone, and rule 6 the copy of the registered file that the
    # tests read, which leaves two values open (tests/conftest.py)
    monkeypatch.setattr(run_episodes.ledger.code_identity, "untracked_loaded", lambda paths: [])
    monkeypatch.setattr(run_episodes.ledger.code_identity, "data_reasons", lambda *args, **kwargs: [])
    monkeypatch.setattr(run_episodes.ledger.code_identity, "open_reasons", lambda: [])
    trial = ("GRID_EXTRA", "GEPA_SPLIT", "HYD_TRAINING")
    launches = [*_page_launches("The registered order"),
                *((script, {key: value for key, value in exports.items() if key not in trial})
                  for script, exports in _page_launches("The two further tables on development combinations"))]
    # the pilot, the gate and the trial of the ladder play the development starts by design and register nothing
    launches = [(script, exports) for script, exports in launches if exports.get("GRID_STARTS") != "development"]
    played, refused, kinds = 0, {}, set()
    for script, exports in launches:
        for argv, settings in _launch_runs(script, exports):
            config_name, config, _prices = run_episodes.configure(settings)
            # the runs in flight that the job hands the run: one in the wall phase, the N of the launch elsewhere
            handed = "1" if config.clock_mode == run_lines.WALL_CLOCK else exports.get("GRID_PARALLEL")
            if handed:
                monkeypatch.setenv("GRID_PARALLEL", handed)
            else:
                monkeypatch.delenv("GRID_PARALLEL", raising=False)
            if reasons := run_episodes.start_check.registration(settings, config, config_name, None):
                refused[" ".join(argv)] = reasons
            played += 1
            kinds.add(exports.get("GRID_PART") or exports.get("GRID_REFERENCE") or exports["RUN_AFTER_SERVE"].rsplit("/", 1)[-1])
    assert {"only", "main", "companion", "environment_token", "layers", "layers_token", "ladder", "ablations", "sampling", "training",
            "run_gepa_online.sh"} <= kinds, kinds
    assert played and not refused, (played, dict(list(refused.items())[:5]), len(refused))


def test_every_command_of_the_page_goes_through_submit_sh():
    commands = _commands()
    assert commands and [line for line in commands if line.startswith("sbatch")] == []
    assert [line for line in commands if ".sbatch" in line and not line.startswith("bash $H/submit.sh ")] == []


def test_the_loops_of_gepa_take_their_folds_from_the_registered_split():
    """run_gepa_online.sh types no fold list: without GEPA_FOLDS it loops over the folds that grid_job_lists.py prints from the
    registered split, so an asset with another number of folds plays every one of them."""
    script = (CLUSTER_JOBS / "3_run" / "run_gepa_online.sh").read_text(encoding="utf-8")
    assert "FOLDS=${GEPA_FOLDS:-$(python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists folds)}" in script and "for FOLD in $FOLDS; do" in script
    assert "GEPA_FOLDS:-1" not in script
    lines = script.splitlines()
    after = lines[lines.index("FOLDS=${GEPA_FOLDS:-$(python -m CausalDiagnosis.experiments.cluster_runs.grid_job_lists folds)} || exit 2") + 1]
    assert after.startswith('[ -n "$FOLDS" ] || {') and after.endswith("exit 2; }")  # an empty list starts no loop without a word


# ----- the settings, the submit line and the clone (the scripts run in bash with stand-ins, no Slurm and no model) -----

SETTINGS_SENTENCE = "set ROOT in env.sh to a folder of your own on the scratch filesystem"
LOCK = "lock of the test tree\n"
# the fetch command of the data of record: it records how it was called and passes its check unless FAKE_DATA_CHECK names a status
FETCH = """import os, sys
from pathlib import Path
with open(Path(sys.argv[0]).with_suffix(".calls"), "a") as f:
    f.write(os.getcwd() + " " + " ".join(sys.argv[1:]) + "\\n")
sys.exit(int(os.environ.get("FAKE_DATA_CHECK", "0")))
"""
# the modules of the default configuration that data_file imports, and the module of the one fetch command of the data of record
DATA_STAND_INS = {"CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config": "class EpisodeConfig:\n    pass\n",
                  "CausalDiagnosis.environment.Causal_Diagnosis_Environment.environment_factory":
                  "def default_facts_path(config):\n    return '/data/of/record.jsonl'\n"}
FETCH_MODULE = "CausalDiagnosis.datasets.hyd_1.download_and_build"
# the ledger that a job asks for the record check and the serving check, and the digest tool of its opening check
LEDGER_MODULE = "CausalDiagnosis.experiments.run_verification.run_ledger"
DIGESTS_MODULE = "CausalDiagnosis.experiments.run_verification.invariant_digests"


def _jobs(tree):
    """The folder of the scripts in a tree."""
    return tree / CLUSTER_JOBS.relative_to(ROOT)


def _in(tree, name):
    """The script name in a tree, in its step folder."""
    return _jobs(tree) / PLACE[name]


def _module_file(tree, module, text=None):
    """The file of a module of a tree that a script runs with python -m, written with text as a stand-in, together with the
    package files above it, when text is given."""
    path = tree / (module.replace(".", "/") + ".py")
    if text is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        for package in list(path.relative_to(tree).parents)[:-1]:
            (tree / package / "__init__.py").touch()
        path.write_text(text, encoding="utf-8")
    return path



def _tree(tmp_path, scripts=(), freeze=False):
    """A tree for the scripts of this folder: the folder of the scripts with env.sh (the template as it is), functions.sh and
    submit.sh, by which a script of a step folder finds the folder, the named scripts in their step folders, the lock, the
    paths module of this tree, the stand-ins of the data and of the fetch command, the package servers of src, the manifest of
    an upload, an environment of the lock whose python is this test's own, a root on a temporary folder and a folder of
    stand-ins first on PATH. Returns the tree, the root, that folder and the environment."""
    tree, root, fake = tmp_path / "tree", tmp_path / "root", tmp_path / "fakebin"
    _jobs(tree).mkdir(parents=True)
    shutil.copy(CLUSTER_JOBS / "env.example.sh", _jobs(tree) / "env.sh")
    for name in ("functions.sh", "submit.sh", *scripts):
        _in(tree, name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(CLUSTER_JOBS / PLACE[name], _in(tree, name))
    (tree / "uv.lock").write_text(LOCK, encoding="utf-8")
    (tree / ".python-version").write_text("3.12\n", encoding="utf-8")
    (tree / "UPLOAD_MANIFEST.txt").write_text("manifest of the test tree\n", encoding="utf-8")
    _module_file(tree, paths.__name__, Path(paths.__file__).read_text(encoding="utf-8"))
    for module, text in {**DATA_STAND_INS, FETCH_MODULE: FETCH}.items():
        _module_file(tree, module, text)
    (tree / "src" / "servers").mkdir(parents=True)
    (tree / "src" / "servers" / "__init__.py").write_text("", encoding="utf-8")
    (tree / SERVING_FILE.relative_to(ROOT)).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SERVING_FILE, tree / SERVING_FILE.relative_to(ROOT))
    if freeze:
        _record(tree, "RECORD")
    env_bin = root / "envs" / hashlib.sha256(LOCK.encode()).hexdigest()[:12] / "bin"
    env_bin.mkdir(parents=True)
    _stand_in(env_bin, "python", f'exec "{sys.executable}" "$@"')
    (env_bin.parent / ".cd_synced").write_text(hashlib.sha256(LOCK.encode()).hexdigest() + "\n", encoding="utf-8")
    fake.mkdir()
    env = {"PATH": f"{fake}:{os.environ['PATH']}", "USER": os.environ.get("USER", "tester"), "HOME": str(tmp_path), "ROOT": str(root),
           "HPC_DIR": str(_jobs(tree))}
    return tree, root, fake, env


def _services(tree):
    """The job_services.sh of a stand-in framework of the tree, which functions.sh sources as it sources that of every
    framework; a test writes into it the stand-ins of the services and the steps that the framework adds."""
    folder = tree / PACKAGE.relative_to(ROOT) / "frameworks" / "stand_in" / "cluster_jobs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "job_services.sh"


def _record(tree, verdict):
    """run_ledger/ of the tree as a stand-in whose record check gives the verdict, RECORD (a run of record) or DEVELOPMENT."""
    text = LEDGER_STAND_IN.replace("{verdict}", verdict + (" " + "0" * 40 if verdict == "RECORD" else ""))
    _module_file(tree, LEDGER_MODULE, text)


def _stand_in(folder, name, body):
    path = folder / name
    path.write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _run(tmp_path, env, *parts):
    return subprocess.run(["bash", "-c", "\n".join(parts)], capture_output=True, text=True, cwd=tmp_path, env=env)


def _sourced(tree, *parts):
    """The settings of the tree sourced, then the parts."""
    return (f'source "{_jobs(tree) / "env.sh"}"', *parts)


def test_the_settings_file_refuses_a_missing_root_and_a_root_in_the_home_folder(tmp_path):
    tree, root, _fake, env = _tree(tmp_path)
    shutil.rmtree(root)
    without = {name: value for name, value in env.items() if name != "ROOT"}
    done = _run(tmp_path, without, *_sourced(tree, "echo SOURCED"))
    assert done.returncode == 2 and SETTINGS_SENTENCE in done.stderr and "SOURCED" not in done.stdout
    home = tmp_path / "homefs"
    done = _run(tmp_path, {**env, "HOME_FS": str(home), "ROOT": str(home / "runs")}, *_sourced(tree, "echo SOURCED"))
    assert done.returncode == 2 and SETTINGS_SENTENCE in done.stderr and not (home / "runs").exists()
    done = _run(tmp_path, env, *_sourced(tree, "echo SOURCED"))
    assert done.returncode == 0 and done.stdout.split() == ["SOURCED"], done.stderr
    assert stat.S_IMODE(root.stat().st_mode) == 0o700


def test_the_runs_folder_follows_the_kind_of_the_tree(tmp_path):
    tree, root, _fake, env = _tree(tmp_path)
    tree_id = hashlib.sha256((tree / "UPLOAD_MANIFEST.txt").read_bytes()).hexdigest()[:12]

    def runs(kind):
        done = _run(tmp_path, {**env, "RUN_KIND": kind}, *_sourced(tree, 'runs_root; echo "$RUNS"'))
        return done.returncode, done.stdout.strip()

    assert runs("") == (0, f"{root}/runs/development/{tree_id}")
    assert runs("development") == (0, f"{root}/runs/development/{tree_id}")
    assert runs("smoke") == (0, f"{root}/runs/smoke")
    assert runs("record")[0] == 2
    _record(tree, "RECORD")
    assert runs("") == (0, f"{root}/runs/record/{tree_id}")
    assert runs("record") == (0, f"{root}/runs/record/{tree_id}")
    assert runs("development") == (0, f"{root}/runs/development/{tree_id}")


def test_the_python_of_a_job_imports_the_code_of_its_own_tree(tmp_path):
    tree, root, _fake, env = _tree(tmp_path)
    done = _run(tmp_path, env, *_sourced(tree, "fork_python", 'echo "$PYTHONPATH"', "command -v python"))
    assert done.returncode == 0 and done.stdout.split()[0] == f"{tree}:{tree / 'src'}", done.stderr
    assert done.stdout.split()[1].startswith(str(root / "envs"))
    elsewhere = tmp_path / "elsewhere"
    python = next((root / "envs").glob("*/bin/python"))
    _stand_in(python.parent, "python", f'echo "{elsewhere / "CausalDiagnosis"} {elsewhere / "src" / "servers"}"')
    done = _run(tmp_path, env, *_sourced(tree, "fork_python", "echo IMPORTED"))
    assert done.returncode == 2 and "IMPORTED" not in done.stdout
    assert f"not from {tree / 'CausalDiagnosis'} and {tree / 'src' / 'servers'}" in done.stderr
    shutil.rmtree(python.parent.parent)
    done = _run(tmp_path, env, *_sourced(tree, "fork_python", "echo IMPORTED"))
    assert done.returncode == 2 and "setup_env.sbatch" in done.stderr


def test_a_model_folder_that_does_not_exist_stops_at_once(tmp_path):
    tree, root, fake, env = _tree(tmp_path)
    _stand_in(fake, "sleep", f'echo "sleep $*" >> "{tmp_path}/sleeps"')
    models = tmp_path / "models"
    done = _run(tmp_path, {**env, "MODELS": str(models), "SLURM_JOB_NAME": "cd-serve"}, *_sourced(tree, "model_dir org/absent"))
    assert done.returncode == 0 and done.stdout.strip() == str(models / "absent") and not (tmp_path / "sleeps").exists()
    (models / "loading").mkdir(parents=True)
    done = _run(tmp_path, {**env, "MODELS": str(models), "SLURM_JOB_NAME": "cd-serve"}, *_sourced(tree, "model_dir org/loading"))
    assert done.stdout.strip() == str(models / "loading") and (tmp_path / "sleeps").exists()  # a download in progress is waited for


def test_the_data_check_runs_the_check_of_the_fetch_command(tmp_path):
    """The data of record lies in the data folder of the clone with its pins in the tracked CHECKSUMS.txt, so the data check of
    a job is the check of the one fetch command, run in the tree with its Python, and the settings file names no data file."""
    tree, _root, _fake, env = _tree(tmp_path)
    done = _run(tmp_path, env, *_sourced(tree, "fork_python", "check_data", "echo CHECKED", 'echo "$(data_file)"'))
    assert done.returncode == 0 and done.stdout.split() == ["CHECKED", "/data/of/record.jsonl"], done.stderr
    cwd, *args = _module_file(tree, FETCH_MODULE).with_suffix(".calls").read_text(encoding="utf-8").split()
    assert Path(cwd).resolve() == tree.resolve() and args == ["--check"]
    done = _run(tmp_path, {**env, "FAKE_DATA_CHECK": "1"}, *_sourced(tree, "fork_python", "check_data", "echo CHECKED"))
    assert done.returncode == 2 and "CHECKED" not in done.stdout and "fetch_data.sbatch" in done.stderr
    assert all("HYD_FACTS_PATH" not in text and "DATA_FILES" not in text for text in (ENV_SCRIPT, FUNCTIONS))


# the fetch of the scenario file: it writes its call into the record of the data stand-in, so the order of the calls shows
SCENARIOS = """import os, sys
from pathlib import Path
with open(Path(sys.argv[0]).resolve().parents[1] / "download_and_build.calls", "a") as f:
    f.write(os.getcwd() + " scenarios " + " ".join(sys.argv[1:]) + "\\n")
"""


def test_the_data_job_builds_the_data_then_fetches_the_scenarios_then_checks_the_data(tmp_path):
    """The data job runs in the tree the one fetch and build command with the archive in DATA, then the fetches that the
    job_services.sh of each framework lists (for AssetOpsBench the scenario file of the anchored preset), then the data
    check, and a failed fetch stops the job before anything else runs."""
    tree, root, _fake, env = _tree(tmp_path, ("fetch_data.sbatch",))
    _module_file(tree, "CausalDiagnosis.datasets.hyd_1.stand_in_framework.download_scenarios", SCENARIOS)
    _services(tree).write_text('DATA_FETCHES+=("python -m CausalDiagnosis.datasets.hyd_1.stand_in_framework.download_scenarios")\n', encoding="utf-8")
    calls = _module_file(tree, FETCH_MODULE).with_suffix(".calls")
    done = subprocess.run(["bash", str(_in(tree, "fetch_data.sbatch"))], capture_output=True, text=True, cwd=tmp_path, env=env)
    assert done.returncode == 0 and done.stdout.split()[-1] == "CD_DATA_DONE", done.stderr
    lines = [line.split() for line in calls.read_text(encoding="utf-8").splitlines()]
    assert {Path(line[0]).resolve() for line in lines} == {tree.resolve()}
    assert [line[1:] for line in lines] == [["--download", "--dest", str(root / "data")], ["scenarios"], ["--check"]]
    calls.unlink()
    done = subprocess.run(["bash", str(_in(tree, "fetch_data.sbatch"))], capture_output=True, text=True, cwd=tmp_path,
                          env={**env, "FAKE_DATA_CHECK": "1"})
    assert done.returncode == 3 and "CD_DATA_DONE" not in done.stdout
    assert [line.split()[1:] for line in calls.read_text(encoding="utf-8").splitlines()] == [["--download", "--dest", str(root / "data")]]


def _sbatch_stand_in(fake, tmp_path):
    """sbatch that records its arguments, one per line, and HPC_DIR of its environment, and prints a job id as --parsable does."""
    return _stand_in(fake, "sbatch", f'printf "%s\\n" "$@" > "{tmp_path}/sbatch_args"; echo "$HPC_DIR" > "{tmp_path}/sbatch_hpc_dir"; echo "4242;cluster"')


def test_submit_sh_builds_the_sbatch_line_from_the_settings(tmp_path):
    scripts = ("submit.sh", "precompute_tables.sbatch", "serve_model.sbatch", "train_lora.sbatch")
    tree, root, fake, env = _tree(tmp_path, scripts)
    _sbatch_stand_in(fake, tmp_path)
    settings = {**env, "ACCOUNT": "acct", "PARTITION_CPU": "cpu", "PARTITION_GPU": "gpu", "PARTITION_GPU_RESUMABLE": "gpu,spare",
                "EXCLUDE_NODES": "bad1"}
    submit = str(_in(tree, "submit.sh"))

    def submitted(*argv):
        done = subprocess.run(["bash", submit, *argv], capture_output=True, text=True, cwd=tmp_path, env=settings)
        return done, (tmp_path / "sbatch_args").read_text(encoding="utf-8").splitlines() if done.returncode == 0 else None

    done, args = submitted("precompute_tables.sbatch", "--mem=2G", "--", "first", "second")
    head = ["--parsable", "-A", "acct", "-p", "cpu", "-o", f"{root}/logs/%x-%j.out", "--exclude=bad1"]
    assert done.returncode == 0 and args == [*head, "--mem=2G", str(_in(tree, "precompute_tables.sbatch")), "first", "second"]
    assert done.stdout.strip() == f"SUBMITTED precompute_tables.sbatch 4242 {root}/logs/cd-tables-4242.out"
    assert (tmp_path / "sbatch_hpc_dir").read_text(encoding="utf-8").strip() == str(_jobs(tree))
    done, args = submitted(str(_in(tree, "serve_model.sbatch")), "-J", "cd-judge")
    assert args[:6] == ["--parsable", "-A", "acct", "-p", "gpu", "-o"] and args[-3:] == ["-J", "cd-judge", str(_in(tree, "serve_model.sbatch"))]
    assert done.stdout.strip().endswith("/logs/cd-judge-4242.out")
    done, args = submitted("train_lora.sbatch")
    assert args[3:5] == ["-p", "gpu,spare"]
    outside = tmp_path / "outside.sbatch"
    outside.write_text("#!/bin/bash\n", encoding="utf-8")
    for argv in ([str(outside)], ["missing.sbatch"], ["env.sh.txt"]):
        assert submitted(*argv)[0].returncode == 2, argv
    for name in ("ACCOUNT", "PARTITION_CPU"):
        done = subprocess.run(["bash", submit, "precompute_tables.sbatch"], capture_output=True, text=True, cwd=tmp_path,
                              env={**settings, name: ""})
        assert done.returncode == 2 and f"{name} is empty" in done.stderr  # no value of a cluster stands in a tracked file
    (_jobs(tree) / "precompute_tables.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")
    done, _ = submitted("precompute_tables.sbatch")
    assert done.returncode == 2 and "more than one script precompute_tables.sbatch" in done.stderr  # a name names one script
    (_jobs(tree) / "precompute_tables.sbatch").unlink()
    (_jobs(tree) / "env.sh").write_text(ENV_SCRIPT[:ENV_SCRIPT.rindex("\nsource ")] + "\n", encoding="utf-8")  # made before functions.sh
    done, _ = submitted("precompute_tables.sbatch")
    assert done.returncode == 2 and "predates functions.sh" in done.stderr
    (_jobs(tree) / "env.sh").unlink()
    done, _ = submitted("precompute_tables.sbatch")
    assert done.returncode == 2 and "env.example.sh" in done.stderr


def test_submit_sh_keeps_a_registered_job_on_the_nodes_of_record(tmp_path):
    tree, _root, fake, env = _tree(tmp_path, ("submit.sh", "serve_model.sbatch", "precompute_tables.sbatch"), freeze=True)
    _sbatch_stand_in(fake, tmp_path)
    _stand_in(fake, "sinfo", 'printf "%s\\n" "card1 gpu:h100:2" "card2 gpu:h100:2(S:0-1)" "card3 gpu:h100:4" "other1 gpu:l40s:8" "card1 gpu:h100:2"')
    submit = str(_in(tree, "submit.sh"))

    def submitted(script, **values):
        settings = {**env, "ACCOUNT": "acct", "PARTITION_CPU": "cpu", "PARTITION_GPU": "gpu", "EXCLUDE_NODES": "",
                    "NODES_OF_RECORD": "card1 card3", **values}
        done = subprocess.run(["bash", submit, script], capture_output=True, text=True, cwd=tmp_path, env=settings)
        args = (tmp_path / "sbatch_args").read_text(encoding="utf-8").splitlines() if done.returncode == 0 else []
        (tmp_path / "sbatch_args").unlink(missing_ok=True)
        return done.returncode, [arg for arg in args if arg.startswith(("--exclude", "-w", "--nodelist"))]

    assert submitted("serve_model.sbatch") == (0, ["--exclude=card2"])
    assert submitted("serve_model.sbatch", EXCLUDE_NODES="bad1") == (0, ["--exclude=bad1,card2"])
    assert submitted("serve_model.sbatch", RUN_KIND="development") == (0, [])
    assert submitted("serve_model.sbatch", RUN_KIND="record", NODES_OF_RECORD="card1 card2 card3") == (0, [])
    assert submitted("precompute_tables.sbatch") == (0, [])  # a job without a card plays no registered run
    assert submitted("serve_model.sbatch", NODES_OF_RECORD="") == (0, [])  # the nodes of record are optional


def test_clone_commit_refuses_a_folder_that_exists_and_a_tree_that_differs_from_its_commit(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("needs git")
    tree, root, _fake, env = _tree(tmp_path, ("clone_commit.sh",))
    origin = tmp_path / "origin"
    _jobs(origin).mkdir(parents=True)
    (_jobs(origin) / ".gitignore").write_text("env.sh\n", encoding="utf-8")
    (origin / "README.md").write_text("a small repository\n", encoding="utf-8")
    git = {**env, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.org", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org",
           "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    for argv in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "one"]):
        subprocess.run(["git", "-C", str(origin), *argv], check=True, capture_output=True, env=git)
    commit = subprocess.run(["git", "-C", str(origin), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    settings = {**git, "FORK_URL": str(origin)}
    clone = [str(_in(tree, "clone_commit.sh"))]
    done = subprocess.run(["bash", *clone, commit], capture_output=True, text=True, cwd=tmp_path, env=settings)
    target = root / "clones" / commit[:12]
    assert done.returncode == 0 and done.stdout.strip() == f"H={_jobs(target)}", done.stderr
    assert (_jobs(target) / "env.sh").read_bytes() == (_jobs(tree) / "env.sh").read_bytes()
    done = subprocess.run(["bash", *clone, commit], capture_output=True, text=True, cwd=tmp_path, env=settings)
    assert done.returncode == 4 and "never updated in place" in done.stderr
    done = subprocess.run(["bash", *clone, commit[:12]], capture_output=True, text=True, cwd=tmp_path, env=settings)
    assert done.returncode == 2
    unknown = "0" * 40
    done = subprocess.run(["bash", *clone, unknown], capture_output=True, text=True, cwd=tmp_path, env=settings)
    assert done.returncode == 3 and (root / "clones" / unknown[:12]).is_dir()  # a failed check removes nothing


# ----- the serving job, run whole in bash with stand-ins for apptainer, the card, curl and the node -----

TEMPLATE = "{{ messages }}"
APPTAINER = """echo "$*" >> "$FAKE_LOG/apptainer_calls"
case "$1" in
  build) mkdir -p "$3" ;;
  exec)
    case " $* " in
      *" import sglang"*) echo 0.5.19 ;;
      *" --model-path "*)
        printf "%s\\n" "$@" > "$FAKE_LOG/server_args"; env > "$FAKE_LOG/server_env"
        trap 'echo TERM >> "$FAKE_LOG/server_signals"; exit 0' TERM
        while :; do sleep 1; done ;;
    esac ;;
esac"""
AFTER = """echo "$API_KEY" > "$FAKE_LOG/script_key"
python -c 'import os, sys; print(oct(os.stat(sys.argv[1]).st_mode & 0o777))' "${ENDPOINT_FILE%.txt}.key" > "$FAKE_LOG/key_mode"
cp "$ENDPOINT_FILE" "$FAKE_LOG/endpoint.txt"
exit "${AFTER_STATUS:-0}" """


def _serving_tree(tmp_path, rewrite=0):
    """A tree, a model folder, an image and the stand-ins of a serving job. Returns the tree and the environment of the job."""
    tree, root, fake, env = _tree(tmp_path, ("serve_model.sbatch",))
    log = tmp_path / "log"
    log.mkdir()
    model = tmp_path / "models" / "granite"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer_config.json").write_text('{"chat_template": "%s"}' % TEMPLATE, encoding="utf-8")
    (model / "HUB_COMMIT.txt").write_text("a" * 40 + "\n", encoding="utf-8")
    (tmp_path / "image.sif").write_bytes(b"image")
    _module_file(tree, "CausalDiagnosis.agents.utils.model_adapters",
                 f"import sys\nif {rewrite} == 0:\n    open(sys.argv[2], 'w').write('rewritten')\nsys.exit({rewrite})\n")
    _stand_in(fake, "apptainer", APPTAINER)
    _stand_in(fake, "module", "exit 0")
    _stand_in(fake, "nvidia-smi", 'echo "0, card, 1, driver"')
    _stand_in(fake, "df", 'case " $* " in *" -Pk "*) echo "Filesystem 1K-blocks Used Available Capacity Mounted"; echo "local 1 1 ${FAKE_FREE_KB:-999999999} 1% /local" ;; *) echo "local free" ;; esac')
    _stand_in(fake, "curl", 'echo "$*" >> "$FAKE_LOG/curl_calls"; case "$*" in *"/v1/models"*) echo "{\"data\": []}" ;; '
                            '*"/server_info"*) cat "${FAKE_SERVER_INFO:?}" ;; esac; exit 0')
    _stand_in(fake, "hostname", 'case "${1:-}" in -i) echo "10.1.2.3 10.1.2.4" ;; *) echo node7 ;; esac')
    _module_file(tree, "CausalDiagnosis.experiments.cluster_runs.engine_record",
                 (PACKAGE / "experiments" / "cluster_runs" / "engine_record.py").read_text(encoding="utf-8"))
    _stand_in(fake, "squeue", "exit 0")
    _stand_in(fake, "sleep", 'exec /bin/sleep 0.1')
    after = _stand_in(tmp_path, "after.sh", AFTER)
    job = str(10000 + os.getpid() % 50000)
    env = {**env, "FAKE_LOG": str(log), "MODEL": str(model), "SIF": str(tmp_path / "image.sif"), "SLURM_JOB_ID": job,
           "SLURM_JOB_NAME": "cd-serve", "SGLANG_PORT": "30123", "RUN_AFTER_SERVE": str(after),
           "FAKE_SERVER_INFO": str(PACKAGE / "experiments" / "cluster_runs" / "tests" / "fixtures" / "server_info_v0519.json")}
    return tree, env


def _serve(tree, env, timeout=60):
    return subprocess.run(["bash", str(_in(tree, "serve_model.sbatch"))], capture_output=True, text=True, cwd=tree.parent,
                          env=env, timeout=timeout)


def _logged(env, name):
    path = Path(env["FAKE_LOG"]) / name
    return path.read_text(encoding="utf-8") if path.exists() else None


def test_the_server_listens_on_one_address_and_takes_its_key_from_the_environment(tmp_path):
    for reach, address in (("", "127.0.0.1"), ("nodes", "10.1.2.3")):
        tree, env = _serving_tree(tmp_path / (reach or "loopback"))
        done = _serve(tree, {**env, "SERVE_REACH": reach})
        assert done.returncode == 0 and "KEY_NOT_IN_PS ok" in done.stdout and "CD_SERVE_DONE" in done.stdout, done.stderr
        args = _logged(env, "server_args").splitlines()
        assert args[args.index("--host") + 1] == address and "0.0.0.0" not in args
        key = _logged(env, "script_key").strip()
        assert len(key) == 64 and all(key not in arg for arg in args)
        assert f"CD_ENDPOINT_KEY={key}" in _logged(env, "server_env").splitlines()
        assert key not in _logged(env, "curl_calls") and key not in _logged(env, "apptainer_calls")
        assert _logged(env, "key_mode").strip() == "0o600" and not (Path(env["ROOT"]) / "endpoints" / f"{env['SLURM_JOB_ID']}.key").exists()
        endpoint = _logged(env, "endpoint.txt")
        assert endpoint.splitlines()[0] == f"http://{address}:30123/v1" and key not in endpoint
        assert key not in done.stdout and key not in done.stderr


def test_the_endpoint_file_records_the_engine_from_the_allow_list_of_server_info_and_the_flags_in_public_form(tmp_path):
    tree, env = _serving_tree(tmp_path)
    done = _serve(tree, env)
    assert done.returncode == 0 and "CD_SERVE_DONE" in done.stdout, done.stderr
    lines = _logged(env, "endpoint.txt").splitlines()
    assert "engine_version 0.5.19" in lines and "dtype bfloat16 dtype_launch bfloat16 quantization none kv_cache_dtype auto" in lines
    launch = json.loads(next(line for line in lines if line.startswith("launch_args ")).split(" ", 1)[1])
    assert launch[launch.index("--model-path") + 1].endswith("@" + "a" * 40) and launch[launch.index("--dtype") + 1] == "bfloat16"
    assert "--host" not in launch and "--port" not in launch and "--api-key" not in launch
    assert "ENGINE_PRECISION dtype bfloat16 dtype_launch bfloat16 quantization none kv_cache_dtype auto" in done.stdout.splitlines()
    fixture = json.loads(Path(env["FAKE_SERVER_INFO"]).read_text(encoding="utf-8"))
    for value in (fixture["api_key"], fixture["admin_api_key"], fixture["host"], fixture["model_path"]):
        assert value not in "\n".join(lines) and value not in done.stdout and value not in done.stderr
    assert str(tmp_path) not in next(line for line in lines if line.startswith("launch_args "))


def test_a_server_that_names_no_precision_stops_the_job_before_the_endpoint_file_and_its_key_is_shown_nowhere(tmp_path):
    tree, env = _serving_tree(tmp_path)
    fixture = json.loads(Path(env["FAKE_SERVER_INFO"]).read_text(encoding="utf-8"))
    info = tmp_path / "server_info.json"
    info.write_text(json.dumps({key: value for key, value in fixture.items() if key != "dtype"}), encoding="utf-8")
    done = _serve(tree, {**env, "FAKE_SERVER_INFO": str(info)})
    assert done.returncode == 3 and "ENGINE_RECORD_FAILED the /server_info answer names no dtype" in done.stderr, done.stderr
    assert not (Path(env["ROOT"]) / "endpoints" / f"{env['SLURM_JOB_ID']}.txt").exists() and _logged(env, "endpoint.txt") is None
    assert all(fixture[key] not in done.stdout + done.stderr for key in ("api_key", "admin_api_key", "host"))


def test_a_serving_job_without_a_script_and_without_a_reader_is_refused(tmp_path):
    tree, env = _serving_tree(tmp_path)
    env = {name: value for name, value in env.items() if name != "RUN_AFTER_SERVE"}
    done = _serve(tree, env)
    assert done.returncode == 2 and "serve nobody" in done.stderr and _logged(env, "apptainer_calls") is None


def test_a_serving_job_ends_with_the_status_of_its_script_and_stops_its_server(tmp_path):
    tree, env = _serving_tree(tmp_path / "script")
    done = _serve(tree, {**env, "AFTER_STATUS": "7"})
    assert done.returncode == 7 and "RUN_AFTER_SERVE finished with 7" in done.stdout and "CD_SERVE_DONE" in done.stdout, done.stderr
    assert _logged(env, "server_signals") == "TERM\n"
    tree, env = _serving_tree(tmp_path / "reader")
    env = {**{name: value for name, value in env.items() if name != "RUN_AFTER_SERVE"}, "SERVE_REACH": "nodes"}
    job = subprocess.Popen(["bash", str(_in(tree, "serve_model.sbatch"))], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                           cwd=tree.parent, env=env)
    endpoints = Path(env["ROOT"]) / "endpoints"
    for _ in range(300):
        if (endpoints / f"{env['SLURM_JOB_ID']}.txt").exists() or job.poll() is not None:
            break
        subprocess.run(["/bin/sleep", "0.1"])
    (endpoints / f"{env['SLURM_JOB_ID']}.stop").touch()
    out, err = job.communicate(timeout=60)
    assert job.returncode == 0 and "the stop file appeared" in out and "CD_SERVE_DONE" in out, err
    assert _logged(env, "server_signals") == "TERM\n" and not (endpoints / f"{env['SLURM_JOB_ID']}.stop").exists()
    # the warning signal of the time limit: the script gets it, ends with its own status, and the server stops
    tree, env = _serving_tree(tmp_path / "warning")
    script = _stand_in(tmp_path / "warning", "warned.sh",
                       'trap \'echo TERM > "$FAKE_LOG/script_term"; exit 9\' TERM\n: > "$FAKE_LOG/script_started"\nwhile :; do sleep 1; done')
    job = subprocess.Popen(["bash", str(_in(tree, "serve_model.sbatch"))], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                           cwd=tree.parent, env={**env, "RUN_AFTER_SERVE": str(script), "SCRIPT_GRACE": "5"})
    for _ in range(300):
        if (Path(env["FAKE_LOG"]) / "script_started").exists() or job.poll() is not None:
            break
        subprocess.run(["/bin/sleep", "0.1"])
    job.send_signal(15)
    out, err = job.communicate(timeout=60)
    assert job.returncode == 9 and "the time limit is near" in out and "RUN_AFTER_SERVE finished with 9" in out, err
    assert _logged(env, "script_term") == "TERM\n" and _logged(env, "server_signals") == "TERM\n"


# a card for the tests of the card of record: its name and memory come from FAKE_GPU and FAKE_MIB, and its sampler writes a
# header, records its call and its stop, or fails at once with FAKE_SAMPLER_FAIL
CARD = """case "$*" in
  *--help-query-gpu*) printf '"timestamp"\\n"utilization.gpu"\\n"memory.used"\\n"clocks_throttle_reasons.active"\\n' ;;
  *" -l 5"*) if [ -n "${FAKE_SAMPLER_FAIL:-}" ]; then echo "no permission to sample"; exit 9; fi
    echo "$*" >> "$FAKE_LOG/sampler_calls"; echo "timestamp, utilization.gpu [%]"
    trap 'echo STOP >> "$FAKE_LOG/sampler_calls"; exit 0' TERM; while :; do /bin/sleep 0.1; done ;;
  *--query-gpu=name,memory.total,uuid,driver_version*) echo "${FAKE_GPU:-NVIDIA TEST NVL}, ${FAKE_MIB:-1000}, GPU-1, 1.2.3" ;;
  *--query-gpu=name,memory.total*) echo "${FAKE_GPU:-NVIDIA TEST NVL}, ${FAKE_MIB:-1000}" ;;
  *--query-gpu*) echo "0, ${FAKE_GPU:-NVIDIA TEST NVL}, ${FAKE_MIB:-1000} MiB, 1.2.3" ;;
  "") echo "| NVIDIA-SMI 1.2.3    Driver Version: 1.2.3    CUDA Version: 9.9 |" ;;
esac"""


def test_a_registered_job_needs_the_card_of_record_and_compares_its_name_and_memory(tmp_path):
    """The settings of the card of record are empty in the template, a registered job stops with status 2 without them, and a
    card is the card of record when both its name, as nvidia-smi prints it, and its memory in MiB equal the settings."""
    tree, _root, fake, env = _tree(tmp_path)
    _stand_in(fake, "nvidia-smi", CARD)
    assert "GPU_OF_RECORD=${GPU_OF_RECORD:-}" in ENV_SCRIPT and "GPU_MIB_OF_RECORD=${GPU_MIB_OF_RECORD:-}" in ENV_SCRIPT
    done = _run(tmp_path, env, *_sourced(tree, "require_gpu_of_record", "echo SET"))
    assert done.returncode == 2 and "GPU_OF_RECORD is empty" in done.stderr and "SET" not in done.stdout
    record = {**env, "GPU_OF_RECORD": "NVIDIA TEST NVL", "GPU_MIB_OF_RECORD": "1000"}
    done = _run(tmp_path, record, *_sourced(tree, "require_gpu_of_record", 'echo "$(card_record) $(cuda_version)"',
                                            'card_of_record "NVIDIA TEST NVL" 1000 && echo SAME',
                                            'card_of_record "NVIDIA TEST PCIe" 2000 || echo OTHER', 'card_of_record "NVIDIA TEST NVL" 2000 || echo OTHER'))
    assert done.returncode == 0 and done.stdout.split("\n")[:4] == ["NVIDIA TEST NVL|1000|GPU-1|1.2.3 9.9", "SAME", "OTHER", "OTHER"]


def test_a_job_with_a_card_checks_it_whenever_env_sh_names_the_card_of_record(tmp_path):
    """check_card of env.sh reads the card of the job and card_line writes it as the line of an endpoint file. The card is checked
    against the card of record when the caller asks for it or when env.sh names the card: on another card the job prints
    GPU_NOT_OF_RECORD with its host and card and ends with status 2, and an asking job without both settings, or a job with one
    of them, ends with status 2. A job, registered or not, whose env.sh names no card passes on any card."""
    tree, _root, fake, env = _tree(tmp_path)
    _stand_in(fake, "nvidia-smi", CARD)
    _stand_in(fake, "hostname", "echo node7")
    other = {**env, "FAKE_GPU": "NVIDIA TEST PCIe", "FAKE_MIB": "2000"}
    record = {"GPU_OF_RECORD": "NVIDIA TEST NVL", "GPU_MIB_OF_RECORD": "1000"}
    line = "gpu_name NVIDIA_TEST_PCIe gpu_mib 2000 gpu_uuid GPU-1 driver_version 1.2.3 cuda_version 9.9"
    cases = {"development, no card named": (other, "development", "", 0, f"PASSED {line}"),
             "development, the card named": ({**other, **record}, "development", "", 2, "GPU_NOT_OF_RECORD node7 NVIDIA TEST PCIe 2000"),
             "development, on the card named": ({**env, **record}, "development", "", 0, "PASSED gpu_name NVIDIA_TEST_NVL gpu_mib 1000 "),
             "record, no card named": (other, "record", "", 0, f"PASSED {line}"),
             "asked, no card named": (other, "development", "required", 2, "GPU_OF_RECORD is empty"),
             "one setting alone": ({**other, "GPU_OF_RECORD": "NVIDIA TEST NVL"}, "development", "", 2, "GPU_MIB_OF_RECORD is empty")}
    for name, (environment, kind, ask, status, printed) in cases.items():
        done = _run(tmp_path, {**environment, "RUNS_KIND": kind}, *_sourced(tree, f"check_card {ask}", 'echo "PASSED $(card_line)"'))
        assert done.returncode == status and printed in done.stdout + done.stderr, (name, done.stdout, done.stderr)
        assert ("PASSED" in done.stdout) == (status == 0), name


def test_a_serving_job_records_its_card_and_samples_it_and_a_registered_one_needs_the_card_of_record(tmp_path):
    """Every serving job writes the line of its card into its endpoint file and samples the card for its life with the fields
    that the driver lists, a sampler that cannot run prints GPU_SAMPLER_FAILED and the server serves on, and a registered job
    on another card prints GPU_NOT_OF_RECORD and stops with status 2 before the sandbox, the server and an endpoint file. With no
    card of record in env.sh a registered job serves on any card, and with one of the two settings alone it stops."""
    tree, env = _serving_tree(tmp_path / "development")
    _stand_in(Path(env["PATH"].split(":")[0]), "nvidia-smi", CARD)
    done = _serve(tree, {**env, "FAKE_GPU": "NVIDIA TEST PCIe", "FAKE_MIB": "2000"})
    assert done.returncode == 0 and "CD_SERVE_DONE" in done.stdout and "GPU_SAMPLER_FAILED" not in done.stdout, done.stderr
    sampler = Path(env["ROOT"]) / "logs" / f"gpu-{env['SLURM_JOB_ID']}.csv"
    assert (f"gpu_name NVIDIA_TEST_PCIe gpu_mib 2000 gpu_uuid GPU-1 driver_version 1.2.3 cuda_version 9.9 gpu_sampler {sampler}"
            == _logged(env, "endpoint.txt").splitlines()[3])
    calls = _logged(env, "sampler_calls").splitlines()
    assert calls == ["--query-gpu=timestamp,utilization.gpu,memory.used,clocks_throttle_reasons.active --format=csv -l 5", "STOP"]
    assert sampler.read_text(encoding="utf-8").startswith("timestamp, utilization.gpu")
    tree, env = _serving_tree(tmp_path / "failed")
    _stand_in(Path(env["PATH"].split(":")[0]), "nvidia-smi", CARD)
    done = _serve(tree, {**env, "FAKE_SAMPLER_FAIL": "1"})
    assert done.returncode == 0 and "GPU_SAMPLER_FAILED: no permission to sample" in done.stdout and _logged(env, "server_args"), done.stderr
    tree, env = _serving_tree(tmp_path / "registered")
    _stand_in(Path(env["PATH"].split(":")[0]), "nvidia-smi", CARD)
    _record(tree, "RECORD")
    record = {**env, "GPU_OF_RECORD": "NVIDIA TEST NVL", "GPU_MIB_OF_RECORD": "1000"}
    done = _serve(tree, {**record, "FAKE_GPU": "NVIDIA TEST PCIe", "FAKE_MIB": "2000"})
    assert done.returncode == 2 and "GPU_NOT_OF_RECORD node7 NVIDIA TEST PCIe 2000" in done.stdout, done.stderr
    assert _logged(env, "apptainer_calls") is None and not list((Path(env["ROOT"]) / "endpoints").glob("*.txt"))
    done = _serve(tree, {**env, "FAKE_GPU": "NVIDIA TEST PCIe", "FAKE_MIB": "2000"})
    assert done.returncode == 0 and "gpu_name NVIDIA_TEST_PCIe gpu_mib 2000 " in _logged(env, "endpoint.txt"), done.stderr
    assert _serve(tree, {**env, "GPU_MIB_OF_RECORD": "1000"}).returncode == 2
    done = _serve(tree, record)
    assert done.returncode == 0 and "gpu_name NVIDIA_TEST_NVL gpu_mib 1000" in _logged(env, "endpoint.txt"), done.stderr
    # once env.sh names the card of record, a development job on another card stops as well
    tree, env = _serving_tree(tmp_path / "named")
    _stand_in(Path(env["PATH"].split(":")[0]), "nvidia-smi", CARD)
    named = {**env, "GPU_OF_RECORD": "NVIDIA TEST NVL", "GPU_MIB_OF_RECORD": "1000", "FAKE_GPU": "NVIDIA TEST PCIe", "FAKE_MIB": "2000"}
    done = _serve(tree, named)
    assert done.returncode == 2 and "GPU_NOT_OF_RECORD node7 NVIDIA TEST PCIe 2000" in done.stdout and _logged(env, "apptainer_calls") is None


def test_the_image_is_unpacked_once_after_a_check_of_the_free_space(tmp_path):
    tree, env = _serving_tree(tmp_path / "room")
    done = _serve(tree, env)
    assert done.returncode == 0, done.stderr
    calls = _logged(env, "apptainer_calls").splitlines()
    builds = [call for call in calls if call.startswith("build ")]
    assert len(builds) == 1 and builds[0].startswith("build --sandbox ") and builds[0].endswith(env["SIF"])
    sandbox = builds[0].split()[2]
    execs = [call for call in calls if call.startswith("exec ")]
    assert len(execs) == 2 and all(f" {sandbox} " in call and env["SIF"] not in call for call in execs)
    tree, env = _serving_tree(tmp_path / "full")
    done = _serve(tree, {**env, "FAKE_FREE_KB": "1"})
    assert done.returncode == 3 and "FREE_FACTOR" not in done.stdout and _logged(env, "apptainer_calls") is None


def test_the_endpoint_file_names_the_lora_adapters_that_the_server_holds(tmp_path):
    """The endpoint file of a serving job names the LoRA adapters its server holds beside the model, 0 without SERVE_ADAPTERS and one
    per adapter folder under HYD_TRAINING with it, the serving setup that run.json records for every run on the endpoint."""
    tree, env = _serving_tree(tmp_path / "plain")
    done = _serve(tree, env)
    assert done.returncode == 0, done.stderr
    assert "lora_adapters 0" in _logged(env, "endpoint.txt").splitlines() and "--enable-lora" not in _logged(env, "server_args").splitlines()
    tree, env = _serving_tree(tmp_path / "adapters")
    adapters = tmp_path / "adapters" / "training" / "adapters"
    for name in ("causal_ft_1", "causal_ft_2"):
        (adapters / name).mkdir(parents=True)
        (adapters / name / "adapter_config.json").write_text("{}", encoding="utf-8")
    done = _serve(tree, {**env, "SERVE_ADAPTERS": "1", "HYD_TRAINING": str(adapters.parent)})
    assert done.returncode == 0, done.stderr
    assert "lora_adapters 2" in _logged(env, "endpoint.txt").splitlines() and "--enable-lora" in _logged(env, "server_args").splitlines()


def test_a_failed_template_rewrite_stops_the_job_and_the_endpoint_file_names_the_template(tmp_path):
    tree, env = _serving_tree(tmp_path / "failed", rewrite=1)
    done = _serve(tree, env)
    assert done.returncode == 3 and f"the chat template of {env['MODEL']} could not be rewritten" in done.stderr
    assert _logged(env, "server_args") is None
    own = hashlib.sha256(TEMPLATE.encode()).hexdigest()
    for case, rewrite, tool_response, expected in (("kept", 3, "1", f"chat_template model template_sha256 {own}"),
                                                    ("plain", 0, "", f"chat_template model template_sha256 {own}"),
                                                    ("rewritten", 0, "1", "chat_template tool_response template_sha256 "
                                                     + hashlib.sha256(b"rewritten").hexdigest())):
        tree, env = _serving_tree(tmp_path / case, rewrite=rewrite)
        done = _serve(tree, {**env, "MODEL_ADAPTER": tool_response})
        assert done.returncode == 0, (case, done.stderr)
        assert _logged(env, "endpoint.txt").splitlines()[-1] == expected, case
        assert ("--chat-template" in _logged(env, "server_args").splitlines()) == (case == "rewritten"), case


# ----- the lines of a serving job, the grid runner and the job record -----

RUNNER_STAND_IN = """echo "RUNNER $GRID_FILE $GRID_STARTS" >> "$FAKE_LOG/runner_calls"
case "$GRID_FILE" in /*) cp "$GRID_FILE" "$FAKE_LOG/played" ;; *) cp "$HPC_DIR/grids/$GRID_FILE" "$FAKE_LOG/played" ;; esac
exit "${RUNNER_STATUS:-0}" """


def _lines_tree(tmp_path, freeze=False):
    """A tree for run_lines.sh with stand-ins for the runner, for CouchDB, for pending_grid_lines.py, for curl and for the node."""
    tree, root, fake, env = _tree(tmp_path, ("run_lines.sh",), freeze=freeze)
    log = tmp_path / "log"
    log.mkdir()
    _stand_in(_in(tree, "run_grid.sbatch").parent, "run_grid.sbatch", RUNNER_STAND_IN)
    _services(tree).write_text('start_couchdb() { echo "COUCHDB $1" >> "$FAKE_LOG/couchdb"; COUCH_PID=; }\n', encoding="utf-8")
    _module_file(tree, pending_lines.__name__,
                 "import sys\nopen(sys.argv[0] + '.args', 'w').write(' '.join(sys.argv[1:]))\nprint('--config base --agent causal --replicate 2')\n")
    (_jobs(tree) / "grids").mkdir()
    (_jobs(tree) / "grids" / "local_one.txt").write_text("# one line\n--config base --agent causal\n", encoding="utf-8")
    _stand_in(fake, "hostname", 'case "${1:-}" in -s) echo node7 ;; *) echo node7.cluster ;; esac')
    # curl answers the model list of an endpoint, as an endpoint does once its serving job has written the endpoint file, which
    # the opening checks wait for (endpoint_url of env.sh); the timed call of the checks still meets the dead port of the file
    _stand_in(fake, "curl", 'case "$*" in *"/v1/models"*) exit 0 ;; *) exit 7 ;; esac')
    tree_id = hashlib.sha256((tree / "UPLOAD_MANIFEST.txt").read_bytes()).hexdigest()[:12]
    runs = root / "runs" / ("record" if freeze else "development") / tree_id
    # the opening checks of a job with a served model: the digest tool and the digest label of the clone, and run_ledger/
    _module_file(tree, DIGESTS_MODULE, "import os, sys\nsys.exit(int(os.environ.get('FAKE_DIGESTS', '0')))\n")
    (root / "digests").mkdir(parents=True, exist_ok=True)
    (root / "digests" / f"{tree_id}.json").write_text("{}", encoding="utf-8")
    if not freeze:
        _record(tree, "DEVELOPMENT")
    return tree, runs, {**env, "FAKE_LOG": str(log), "GRID_FILE": "local_one.txt", "SLURM_JOB_ID": "55"}


def _lines(tree, env):
    return subprocess.run(["bash", str(_in(tree, "run_lines.sh"))], capture_output=True, text=True, cwd=tree.parent, env=env, timeout=60)


def test_an_empty_grid_file_ends_with_nothing_pending(tmp_path):
    tree, _runs, env = _lines_tree(tmp_path)
    (_jobs(tree) / "grids" / "local_empty.txt").write_text("# a grid file whose lines have all run\n\n", encoding="utf-8")
    done = _lines(tree, {**env, "GRID_FILE": "local_empty.txt"})
    assert done.returncode == 0 and "nothing pending in local_empty.txt" in done.stdout and "CD_RUN_LINES_DONE" in done.stdout, done.stderr
    assert _logged(env, "runner_calls") is None and _logged(env, "couchdb") is None
    assert _lines(tree, {**env, "GRID_FILE": "local_absent.txt"}).returncode == 2


def test_a_requeued_job_plays_the_pending_pairs_only(tmp_path):
    tree, runs, env = _lines_tree(tmp_path)
    (runs / "grid_55").mkdir(parents=True)
    done = _lines(tree, {**env, "SLURM_RESTART_COUNT": "1"})
    assert done.returncode == 0, done.stderr
    assert (runs / "grid_55_attempt1").is_dir() and _logged(env, "played") == "--config base --agent causal --replicate 2\n"
    asked = Path(f"{_module_file(tree, pending_lines.__name__)}.args").read_text(encoding="utf-8")
    assert asked == (f"--grid {_jobs(tree) / 'grids' / 'local_one.txt'} --runs {runs / 'grid_55_attempt1'} --starts sampled "
                     f"--model {TEMPLATE_SERVED_NAME}")
    done = _lines(tree, env)
    assert _logged(env, "played") == "# one line\n--config base --agent causal\n"  # a first start plays the whole file


def test_a_requeued_job_asks_for_the_pairs_of_the_model_it_serves(tmp_path):
    """The grid lines name no model, run_grid.sbatch adds the served name after each line, so run_lines.sh passes it to
    pending_grid_lines.py, which matches a finished run of a model agent on that name."""
    tree, runs, env = _lines_tree(tmp_path)
    (runs / "grid_55").mkdir(parents=True)
    done = _lines(tree, {**env, "SLURM_RESTART_COUNT": "1", "MODEL": "/models/served-model"})
    assert done.returncode == 0, done.stderr
    asked = Path(f"{_module_file(tree, pending_lines.__name__)}.args").read_text(encoding="utf-8")
    assert asked.endswith("--starts sampled --model served-model"), asked


def test_with_grid_pending_a_job_plays_only_the_pairs_that_no_finished_run_of_its_runs_root_covers(tmp_path):
    """GRID_PENDING=1, for a part whose lines another part has played or a part submitted again after a job that the time limit
    ended: the job plays only the pairs that pending_grid_lines.py finds without a finished run in any grid folder of RUNS, so no
    pair is played twice into one RUNS root. The folders of the loops of GEPA hold no grid line and are not read, and without
    the switch a first start plays the whole file."""
    tree, runs, env = _lines_tree(tmp_path)
    for name in ("grid_40", "grid_41_attempt1", "gepa_online_42"):
        (runs / name).mkdir(parents=True)
    done = _lines(tree, {**env, "GRID_PENDING": "1"})
    assert done.returncode == 0, done.stderr
    assert _logged(env, "played") == "--config base --agent causal --replicate 2\n"
    asked = Path(f"{_module_file(tree, pending_lines.__name__)}.args").read_text(encoding="utf-8")
    assert asked == (f"--grid {_jobs(tree) / 'grids' / 'local_one.txt'} --runs {runs / 'grid_40'} {runs / 'grid_41_attempt1'} "
                     f"--starts sampled --model {TEMPLATE_SERVED_NAME}")
    done = _lines(tree, {**env, "GRID_PENDING": "1", "GRID_STARTS": "development"})
    asked = Path(f"{_module_file(tree, pending_lines.__name__)}.args").read_text(encoding="utf-8")
    assert done.returncode == 0 and asked.endswith(f"{runs / 'grid_41_attempt1'} --starts development --model {TEMPLATE_SERVED_NAME}"), done.stderr
    done = _lines(tree, env)
    assert done.returncode == 0 and _logged(env, "played") == "# one line\n--config base --agent causal\n"


def test_with_grid_pending_a_selector_in_grid_extra_stops_the_job_before_its_first_line(tmp_path):
    """GRID_PENDING=1 matches the finished runs to the lines of the grid file, which hold no word of GRID_EXTRA, so a selector in
    GRID_EXTRA would leave every line pending and play it again: the job stops before its first line and says that a selector
    belongs on the grid line, for either selector in either form, while without GRID_PENDING the same GRID_EXTRA plays the file."""
    tree, runs, env = _lines_tree(tmp_path)
    (runs / "grid_40").mkdir(parents=True)
    for extra in ("--start-state=3/100/0/130@2", "--store couchdb --scenario-ids 2001"):
        done = _lines(tree, {**env, "GRID_PENDING": "1", "GRID_EXTRA": extra})
        assert done.returncode == 2 and "so a selector belongs on the grid line and not in GRID_EXTRA" in done.stderr, done.stderr
        assert _logged(env, "runner_calls") is None and not Path(f"{_module_file(tree, pending_lines.__name__)}.args").exists(), extra
    done = _lines(tree, {**env, "GRID_EXTRA": "--start-state=3/100/0/130@2"})
    assert done.returncode == 0 and _logged(env, "played") == "# one line\n--config base --agent causal\n", done.stderr


def test_run_lines_ends_with_the_status_of_the_runner(tmp_path):
    tree, _runs, env = _lines_tree(tmp_path)
    done = _lines(tree, {**env, "RUNNER_STATUS": "5"})
    assert done.returncode == 5 and "RUN_GRID_EXIT 5" in done.stdout and "CD_RUN_LINES_DONE" not in done.stdout
    done = _lines(tree, env)
    assert done.returncode == 0 and "RUN_GRID_EXIT 0" in done.stdout and "CD_RUN_LINES_DONE" in done.stdout, done.stderr


def test_a_registered_job_runs_on_a_node_of_record_and_the_card_of_record_only(tmp_path):
    """Once env.sh names them, a registered line plays on a node of NODES_OF_RECORD and on the card of record alone, as the
    endpoint file of its serving job names the card, and stops with status 2 on another node or another card; without them it
    plays on any node and card. GRID_PARALLEL is no setting of record: the wall-clock runs play alone in a phase of their own."""
    tree, _runs, env = _lines_tree(tmp_path, freeze=True)
    endpoint = tmp_path / "endpoint.txt"
    endpoint.write_text("http://127.0.0.1:1/v1\njob 55 node node7 model m served_name m\nstarted now\n"
                        "gpu_name NVIDIA_TEST_NVL gpu_mib 1000 gpu_uuid GPU-1 driver_version 1.2.3 cuda_version 9.9 gpu_sampler s.csv\n",
                        encoding="utf-8")
    env = {**env, "ENDPOINT_FILE": str(endpoint), "GPU_OF_RECORD": "NVIDIA TEST NVL", "GPU_MIB_OF_RECORD": "1000"}
    assert _lines(tree, {**env, "NODES_OF_RECORD": "node3 node7"}).returncode == 0
    assert _lines(tree, {**env, "NODES_OF_RECORD": ""}).returncode == 0
    for nodes in ("node3",):
        done = _lines(tree, {**env, "NODES_OF_RECORD": nodes})
        assert done.returncode == 2 and "node of record" in done.stderr, nodes
    for settings in ({"GPU_MIB_OF_RECORD": "2000"}, {"GPU_OF_RECORD": "NVIDIA TEST PCIe"}, {"GPU_OF_RECORD": ""}):
        done = _lines(tree, {**env, "NODES_OF_RECORD": "node7", **settings})
        assert done.returncode == 2 and ("not the card of record" in done.stderr or "GPU_OF_RECORD is empty" in done.stderr), settings
    assert _lines(tree, {**env, "NODES_OF_RECORD": "", "RUN_KIND": "development", "GPU_OF_RECORD": ""}).returncode == 0


GRID_LINES_STAND_IN = """import sys
open(sys.argv[0] + '.args', 'a').write(' '.join(sys.argv[1:]) + '\\n')
print('# ' + ' '.join(sys.argv[1:]))
print('--config base --agent causal')
"""


def test_a_part_or_a_row_plays_the_grid_file_that_grid_lines_prints(tmp_path):
    tree, runs, env = _lines_tree(tmp_path)
    _module_file(tree, grid_lines.__name__, GRID_LINES_STAND_IN)
    env.pop("GRID_FILE")
    for name, value in (("GRID_PART", "main"), ("GRID_ROW", "causal+mcp")):
        done = _lines(tree, {**env, name: value})
        assert done.returncode == 0 and "CD_RUN_LINES_DONE" in done.stdout, done.stderr
        assert _logged(env, "played") == f"# {name[5:].lower()} {value}\n--config base --agent causal\n"
    asked = Path(f"{_module_file(tree, grid_lines.__name__)}.args").read_text(encoding="utf-8")
    assert asked == "part main\nrow causal+mcp\n"
    (runs / "grid_55").mkdir(parents=True)
    done = _lines(tree, {**env, "GRID_PART": "main", "SLURM_RESTART_COUNT": "1"})
    assert done.returncode == 0, done.stderr
    pending = Path(f"{_module_file(tree, pending_lines.__name__)}.args").read_text(encoding="utf-8")
    assert pending.startswith("--grid /") and "/grid_main.txt --runs " in pending
    for names in ({}, {"GRID_PART": "main", "GRID_ROW": "causal"}, {"GRID_PART": "main", "GRID_FILE": "local_one.txt"}):
        done = _lines(tree, {**env, **names})
        assert done.returncode == 2 and "one of GRID_FILE" in done.stderr, names


UNIT_STAND_INS = ('python() { case "$*" in *episode_runners.run_episodes*) exit_of "$HYD_GRID_LINE" ;; *) echo "PY $*" ;; esac; }',
                  'exit_of() { case "$1" in 0) return 0 ;; 1) return 75 ;; *) return 3 ;; esac; }', "hostname() { echo node7; }")


def _units(*stand_ins):
    """The functions of the runner that play the units, with the settings of a grid of three lines and one replicate."""
    return (*stand_ins, re.search(r"^replicate_option\(\) \{.*\}$", RUNNER, re.M).group(0), *SCHEDULER,
            *(_function(RUNNER, name) for name in ("reference_line", "run_unit", "line_name", "start_line", "wall_line",
                                                   "play_units", "unit_statuses")),
            'LINES=("--config base" "--config priced" "--config timed") UNITS=("0 1" "1 1" "2 1")',
            'OUT=. SERVED_NAME=m BASE_URL=http://node/v1 JUDGE= SEEDS="--seeds 1" GRID_PARALLEL=1000 REFERENCE_AGENTS=oracle')


def test_a_failed_unit_fails_the_job_and_a_stopped_unit_is_named_apart(tmp_path):
    stand_ins = (*UNIT_STAND_INS, 'job_checks() { echo "CHECKS $1"; }', 'reference_rows() { echo REFERENCE; }')
    tail = RUNNER[RUNNER.index("play_grid; GRID_STATUS=$?"):]
    done = _bash(tmp_path, *_units(*stand_ins), _function(RUNNER, "play_grid"), "GRID_REFERENCE=0", tail)
    assert done.returncode == 5 and "CD_GRID_DONE" not in done.stdout, done.stderr
    assert [line for line in done.stdout.splitlines() if line.startswith("UNIT_")] == ["UNIT_STOPPED 1 1", "UNIT_FAILED 2 1 3"]
    assert sorted(p.read_text(encoding="utf-8").strip() for p in tmp_path.glob("line_*.log.status")) == ["0", "3", "75"]
    done = _bash(tmp_path, *_units(*stand_ins), _function(RUNNER, "play_grid"), "GRID_REFERENCE=0", 'exit_of() { return 0; }', tail)
    assert done.returncode == 0 and done.stdout.splitlines()[-1] == "CD_GRID_DONE .", done.stderr


def test_a_failed_reference_run_fails_the_job(tmp_path):
    """A reference run writes its exit status beside its log as a unit does, and the job reads it whenever the reference rows
    played, under GRID_REFERENCE=only too: a crashed reference run fails the job with status 5 and is named, a stopped one is named
    apart."""
    driver = ('python() { case "$*" in *episode_runners.run_episodes*" --agent random "*) return 1 ;; *episode_runners.run_episodes*" --agent greedy "*) return 75 ;; '
              '*" clocked "*) return ;; *) return 0 ;; esac; }')
    reference = (*SCHEDULER, re.search(r"^replicate_option\(\) \{.*\}$", RUNNER, re.M).group(0), _function(RUNNER, "wiring_skipped"),
                 *(_function(RUNNER, name) for name in ("reference_name", "start_reference", "wall_reference",
                                                        "reference_rows", "reference_statuses", "play_grid")),
                 'job_checks() { :; }', 'play_units() { :; }', 'unit_statuses() { return 0; }',
                 'OUT=. SEEDS="--seeds 1" GRID_PARALLEL=4 REFERENCE_CONFIGS="base priced" REPLICATES=1 WIRING_AGENTS=scripted')
    for mode in ("only", "1"):
        done = _bash(tmp_path, *reference, driver, 'REFERENCE_AGENTS="oracle random greedy"', f"GRID_REFERENCE={mode}", "play_grid")
        assert done.returncode == 5, (mode, done.stderr)
        assert sorted(line for line in done.stdout.splitlines() if line.startswith("REFERENCE_")) == [
            "REFERENCE_FAILED base random 1 1", "REFERENCE_FAILED priced random 1 1", "REFERENCE_STOPPED base greedy 1",
            "REFERENCE_STOPPED priced greedy 1"], mode
    done = _bash(tmp_path, *reference, driver, 'REFERENCE_AGENTS="oracle"', "GRID_REFERENCE=only", "play_grid")
    assert done.returncode == 0 and "REFERENCE_" not in done.stdout, done.stderr


def test_the_unit_loop_plays_every_unit_with_the_runs_in_flight_of_the_job(tmp_path):
    """The scheduler polls the jobs of the shell in place of wait -n, so it runs on any bash: six units with two runs in flight."""
    driver = ('python() { case "$*" in *" clocked "*) return ;; esac; mkdir -p running; : > "running/$HYD_GRID_LINE"; '
              'ls running | wc -l >> "$PWD/count_in_flight"; sleep 0.3; rm -f "running/$HYD_GRID_LINE"; }')
    units = 'LINES=(a b c d e f) UNITS=("0 1" "1 1" "2 1" "3 1" "4 1" "5 1")'
    parts = [*_units('exit_of() { return 0; }', "hostname() { echo node7; }"), units, "GRID_PARALLEL=2", driver, "play_units"]
    done = _bash(tmp_path, *parts)
    assert done.returncode == 0, done.stderr
    assert len(list(tmp_path.glob("line_*_replicate_1.log"))) == 6 and len(list(tmp_path.glob("line_*.log.status"))) == 6
    assert max(int(n) for n in (tmp_path / "count_in_flight").read_text(encoding="utf-8").split()) <= 2


def test_the_wall_clock_runs_play_one_at_a_time_after_every_other_run(tmp_path):
    """A job plays in two phases: every run whose configuration charges no wall seconds first, GRID_PARALLEL at
    once, then, after the last of them has ended, every wall-clock run alone, one after another, each named by a WALL_EPISODE
    line; no filler runs."""
    driver = ('python() { case "$*" in *" clocked "*) printf -- "--config real_time\\n--config wall_two\\n" ;; '
              '*episode_runners.run_episodes*) echo "start $HYD_GRID_LINE" >> "$PWD/events"; sleep 0.2; echo "end $HYD_GRID_LINE" >> "$PWD/events" ;; esac; }')
    units = 'LINES=("--config base" "--config priced" "--config real_time" "--config wall_two") UNITS=("0 1" "1 1" "2 1" "3 1")'
    done = _bash(tmp_path, *_units(driver, "hostname() { echo node7; }"), units, "GRID_PATH=g.txt GRID_PARALLEL=3", "play_units")
    assert done.returncode == 0, done.stderr
    events = (tmp_path / "events").read_text(encoding="utf-8").split("\n")[:-1]
    first_wall = events.index("start 2")
    assert {"end 0", "end 1"} <= set(events[:first_wall]) and events[first_wall:] == ["start 2", "end 2", "start 3", "end 3"], events
    assert "PHASE_FREE start 2 runs, 3 in flight" in done.stderr and "PHASE_WALL start 2 runs, 1 in flight" in done.stderr
    assert [line for line in done.stderr.splitlines() if line.startswith("WALL_EPISODE")] == ["WALL_EPISODE line_2_replicate_1",
                                                                                              "WALL_EPISODE line_3_replicate_1"]
    assert not list(tmp_path.glob("fillers*")) and "filler" not in (tmp_path / "in_flight.tsv").read_text(encoding="utf-8")


def test_the_wall_phase_hands_its_runs_one_run_in_flight_and_the_other_phase_the_n_of_the_job(tmp_path, monkeypatch):
    """The wall phase hands each of its runs GRID_PARALLEL=1, the runs alive in it, which the start check of a preset with a clock
    compares with design.RUNS_IN_FLIGHT, so a wall-clock line of a job at GRID_PARALLEL=4 is of record. The control: the runs of
    the phase without the wall clock keep the 4 of the job, read as a process started by the run reads its environment."""
    driver = ('python() { case "$*" in *" clocked "*) printf -- "--config real_time\\n" ;; '
              '*run_episodes*) bash -c \'echo "$HYD_GRID_LINE ${GRID_PARALLEL:-unset}"\' >> "$PWD/handed" ;; esac; }')
    units = 'LINES=("--config base" "--config priced" "--config real_time") UNITS=("0 1" "1 1" "2 1")'
    done = _bash(tmp_path, *_units(driver, "hostname() { echo node7; }"), units, "GRID_PATH=g.txt; export GRID_PARALLEL=4", "play_units")
    assert done.returncode == 0, done.stderr
    handed = dict(line.split() for line in (tmp_path / "handed").read_text(encoding="utf-8").splitlines())
    assert handed == {"0": "4", "1": "4", "2": "1"}, handed
    for line, preset in (("2", "real_time"), ("0", "base")):
        monkeypatch.setenv("GRID_PARALLEL", handed[line])
        _name, config, _prices = run_episodes.configure(run_episodes.build_parser().parse_args(["--config", preset, "--agent", "causal"]))
        wall = config.clock_mode == run_lines.WALL_CLOCK
        assert wall == (preset == "real_time") and (run_episodes.runs_in_flight() == design.RUNS_IN_FLIGHT) == wall, (line, handed)


def test_a_unit_stopped_on_harness_errors_ends_the_job_when_the_server_no_longer_answers(tmp_path):
    """After a model run stops with status 75 the job probes the endpoint once: a server that does not answer ends the job with
    SERVICE_DOWN and status 6 before the next run starts, so a dead server holds the card no longer; one that answers lets the job
    play on."""
    driver = ('python() { case "$*" in *" clocked "*) ;; "- "*) echo probe >> "$PWD/calls"; return "${FAKE_HEALTH:-1}" ;; '
              '*episode_runners.run_episodes*) [ "$HYD_GRID_LINE" != 1 ] || return 75 ;; esac; }')
    for health, status in (("1", 6), ("0", 0)):
        folder = tmp_path / health
        folder.mkdir()
        done = _bash(folder, *_units(driver, "hostname() { echo node7; }", 'job_checks() { echo "CHECKS $1"; }'),
                     f"FAKE_HEALTH={health} GRID_PARALLEL=1", "play_units")
        assert done.returncode == status, (health, done.stdout, done.stderr)
        assert (folder / "calls").read_text(encoding="utf-8") == "probe\n", health
        down = "SERVICE_DOWN http://node/v1 after line_1_replicate_1 stopped with status 75"
        assert (down in done.stdout) == (status == 6) and (folder / "line_2_replicate_1.log").exists() == (status == 0), health


class _ModelList(http.server.BaseHTTPRequestHandler):
    """A stand-in endpoint that answers GET /v1/models with an empty model list and logs nothing."""

    def do_GET(self):
        self.send_response(200 if self.path == "/v1/models" else 404)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"data": []}')

    def log_message(self, *args):
        pass


def test_the_probe_after_a_stopped_run_imports_its_health_check_from_the_tree_and_tells_an_answering_endpoint_from_a_dead_one():
    """service_answers runs as the grid job runs it, the Python of the job at the root of the tree with the import path of
    fork_python, against a stand-in that answers its model list, which gives status 0, and against a port where nothing listens,
    which gives 1, each with its ENDPOINT_HEALTH line. The tests of the scheduler stand in for python, so a health check that the
    tree lacks fails here alone."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ModelList)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {"PATH": os.pathsep.join([str(Path(sys.executable).parent), os.environ["PATH"]]),
           "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "src")])}
    try:
        for url, status, line in ((f"http://127.0.0.1:{server.server_port}/v1", 0, "ENDPOINT_HEALTH ok True status 200 "),
                                  ("http://127.0.0.1:1/v1", 1, "ENDPOINT_HEALTH ok False status None ")):
            done = subprocess.run(["bash", "-c", "\n".join([_function(RUNNER, "service_answers"), f"BASE_URL={url} service_answers"])],
                                  capture_output=True, text=True, cwd=ROOT, env=env, timeout=60)
            assert done.returncode == status and done.stdout.startswith(line), (url, done.stdout, done.stderr)
    finally:
        server.shutdown()
        server.server_close()


def test_the_kind_of_a_job_follows_the_record_check_of_its_tree_and_its_start_set(tmp_path):
    """No file of the tree decides the kind of a job: a job is of record when run_ledger/ record-check finds the tree a run of record
    and the job plays the start set of a table, development otherwise; RUN_KIND=record stops on any other verdict or start
    set, and the verdict stands in the log."""
    tree, root, _fake, env = _tree(tmp_path)
    tree_id = hashlib.sha256((tree / "UPLOAD_MANIFEST.txt").read_bytes()).hexdigest()[:12]

    def runs(kind, starts="sampled"):
        done = _run(tmp_path, {**env, "RUN_KIND": kind, "GRID_STARTS": starts}, *_sourced(tree, 'runs_root; echo "$RUNS"'))
        return done.returncode, done.stdout.strip(), done.stderr

    _record(tree, "RECORD")
    assert runs("")[:2] == (0, f"{root}/runs/record/{tree_id}") and "RECORD " in runs("")[2]
    assert runs("", "development")[:2] == (0, f"{root}/runs/development/{tree_id}")
    assert runs("record", "training")[0] == 2
    assert runs("development")[:2] == (0, f"{root}/runs/development/{tree_id}")
    _record(tree, "DEVELOPMENT")
    assert runs("")[:2] == (0, f"{root}/runs/development/{tree_id}")
    code, _out, err = runs("record")
    assert code == 2 and "says DEVELOPMENT" in err, err


def test_every_job_with_a_served_model_opens_with_three_checks(tmp_path):
    """A job whose serving job exports ENDPOINT_FILE checks, before its first line, the digests of a scripted episode against the
    label of its clone (a difference or no label stops it), the served model against the serving file of record (a difference
    stops a registered job and is printed for a development job) and prints the seconds of one timed model call; a run that
    stops plays no line."""
    tree, _runs, env = _lines_tree(tmp_path, freeze=True)
    endpoint = tmp_path / "endpoint.txt"
    endpoint.write_text("http://127.0.0.1:1/v1\njob 55 node node7 model m served_name m\n", encoding="utf-8")
    env = {**env, "ENDPOINT_FILE": str(endpoint)}
    label = Path(env["ROOT"]) / "digests" / f"{hashlib.sha256((tree / 'UPLOAD_MANIFEST.txt').read_bytes()).hexdigest()[:12]}.json"
    played = Path(env["FAKE_LOG"]) / "played"

    def lines(**settings):
        played.unlink(missing_ok=True)
        return _lines(tree, {**env, **settings})

    done = lines()
    assert done.returncode == 0 and "OPENING_DIGESTS_OK" in done.stdout and "SERVING_OK agent" in done.stdout, done.stderr
    assert "OPENING_CALL failed" in done.stdout and played.exists()
    done = lines(FAKE_DIGESTS="1")
    assert done.returncode == 2 and "OPENING_DIGESTS_DIFFER" in done.stderr and not played.exists()
    done = lines(FAKE_SERVING="1")
    assert done.returncode == 2 and "SERVING_DIFFERS" in done.stderr and not played.exists()
    assert lines(FAKE_SERVING="1", RUN_KIND="development").returncode == 0 and played.exists()
    label.unlink()
    done = lines()
    assert done.returncode == 2 and "OPENING_DIGESTS_MISSING" in done.stderr and not played.exists()
    done = lines(ENDPOINT_FILE="")
    assert done.returncode == 0 and "OPENING_" not in done.stdout + done.stderr


def test_a_card_that_serves_the_judge_plays_no_wall_clock_line(tmp_path):
    """A job that serves the judge on its card (CO_SERVE_JUDGE) refuses a grid file with a wall-clock line, since the judge's
    requests would share the card with an episode whose wall seconds count."""
    tree, _runs, env = _lines_tree(tmp_path)
    stand_in = _module_file(tree, grid_lines.__name__, "")
    stand_in.write_text("import sys\nif sys.argv[1] == 'clocked':\n    print('--config base --agent causal')\n", encoding="utf-8")
    done = _lines(tree, {**env, "CO_SERVE_JUDGE": "1"})
    assert done.returncode == 2 and "WALL_CLOCK_WITH_JUDGE" in done.stderr and _logged(env, "played") is None
    stand_in.write_text("import sys\n", encoding="utf-8")
    assert _lines(tree, {**env, "CO_SERVE_JUDGE": "1"}).returncode == 0


def test_a_download_of_a_held_pin_fetches_nothing_and_every_exit_0_names_its_end(tmp_path):
    """setup.sh status reads CD_DOWNLOAD_DONE, so every exit 0 of the download job prints it: a second download of a folder that
    holds the pin and its hashed files fetches nothing, and so do a pinned shared copy and a model given as a folder."""
    tree, _root, fake, env = _tree(tmp_path, ("download_model.sbatch",))
    log = tmp_path / "log"
    log.mkdir()
    _stand_in(fake, "uvx", DOWNLOAD)
    _stand_in(fake, "curl", HUB_LISTING)
    agent = "ibm-granite/granite-3.3-8b-instruct"
    models = tmp_path / "models"
    settings = {**env, "FAKE_LOG": str(log), "MODELS": str(models), "SHARED": "", "MODEL": agent, "FAKE_COMMIT": PINS[agent]}

    def download(**extra):
        return subprocess.run(["bash", str(_in(tree, "download_model.sbatch"))], capture_output=True, text=True, cwd=tmp_path,
                              env={**settings, **extra}, timeout=60)

    assert download().returncode == 0 and _logged(settings, "uvx_calls")
    (log / "uvx_calls").unlink()
    done = download()
    assert done.returncode == 0 and "CD_DOWNLOAD_DONE" in done.stdout and _logged(settings, "uvx_calls") is None, done.stderr
    shared = tmp_path / "shared" / "models" / "granite-3.3-8b-instruct"
    shared.mkdir(parents=True)
    (shared / "config.json").write_text("{}", encoding="utf-8")
    (shared / "HUB_COMMIT.txt").write_text(f"{PINS[agent]}\n", encoding="utf-8")
    done = download(MODELS=str(tmp_path / "empty"), SHARED=str(tmp_path / "shared"))
    assert done.returncode == 0 and f"CD_DOWNLOAD_DONE {shared}" in done.stdout and _logged(settings, "uvx_calls") is None
    done = download(MODEL=str(shared))
    assert done.returncode == 0 and "CD_DOWNLOAD_DONE" in done.stdout and _logged(settings, "uvx_calls") is None


def test_the_page_leads_from_the_clone_to_the_parts_with_no_step_by_hand_between():
    """The path of the page: section 0 clones and runs setup.sh and setup.sh status, the registered order of section 3 starts at
    SETUP_OK, no command of the path names the removed freeze step or a script off the path, and every step off the path stands in the closing section
    Optional tools, where it stays runnable."""
    setup = README[README.index("\n## 0. Setup\n"):README.index("\n## 1. Checks\n")]
    assert "bash $H/0_setup/clone_commit.sh <full hash of the pushed commit>\n" in setup
    assert "bash $H/0_setup/setup.sh\nbash $H/0_setup/setup.sh status\n" in setup and "SETUP_OK" in setup
    path = README[:README.index("\n## Optional tools\n")]
    assert "SETUP_OK" in path[path.index("\n### The registered order\n"):]
    steps = path[path.index("\n## Before the first job\n"):path.index("\n## Files of this folder\n")]
    commands = "".join(re.findall(r"```bash\n(.*?)```", steps, re.S))
    for word in ("freeze", "frozen", "MODEL_REVISIONS"):
        assert word not in steps, word
    for word in ("run_suite.sbatch", "platform_record.sh", "check_nodes.sh label", "GRID_PART=pilot"):
        assert word not in commands, word
    optional = README[README.index("\n## Optional tools\n"):]
    for step in ("run_suite.sbatch", "JUDGE_RECORD_ONLY=1", "platform_record.sh", "latency_probe", "GRID_PART=pilot", "GRID_PART=gate",
                 "check_nodes.sh label", "check_nodes.sh submit", "check_nodes.sh read"):
        assert step in optional, step
    assert "freeze --" not in README and " freeze " not in README.replace("`", " ")


def test_the_job_record_is_kept_when_the_limit_ends_the_job(tmp_path):
    stand_ins = ('python() { case "$*" in *episode_runners.run_episodes*) : > "$PWD/started"; sleep 30 ;; *) echo "PY $*" ;; esac; }',
                 'sacct() { echo "77|RUNNING|2026-01-01T00:00:00|Unknown|00:10:00|00:00:00|node7"; }', "hostname() { echo node7; }")
    tail = RUNNER[RUNNER.index("# a job that the time limit ends keeps its record"):]
    script = "\n".join([*_units(*stand_ins), _function(FUNCTIONS, "write_job_record"), _function(RUNNER, "job_checks"),
                        _function(RUNNER, "play_grid"), 'reference_rows() { :; }', "GRID_REFERENCE=0 SLURM_JOB_ID=77", tail])
    job = subprocess.Popen(["bash", "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmp_path,
                           env={"PATH": os.environ["PATH"]}, start_new_session=True)
    try:
        for _ in range(200):
            if (tmp_path / "started").exists():
                break
            subprocess.run(["/bin/sleep", "0.05"])
        (tmp_path / "job_record.txt").unlink()  # the record of the start, written again by the trap
        job.send_signal(15)
        out, err = job.communicate(timeout=30)
    finally:
        os.killpg(job.pid, 9) if job.poll() is None else None
        subprocess.run(["pkill", "-9", "-g", str(job.pid)], capture_output=True)
    assert job.returncode == 143, err
    assert (tmp_path / "job_record.txt").read_text(encoding="utf-8").startswith("77|RUNNING|")
    assert "PY -m CausalDiagnosis.experiments.cluster_runs.cluster_run_checks latency ." in out and "PY -m CausalDiagnosis.experiments.cluster_runs.cluster_run_checks suspect ." in out


def test_the_grid_job_builds_no_table():
    assert "aggregate.py" not in RUNNER


def test_the_runner_checks_the_tables_of_its_grid_and_computes_none(tmp_path):
    """A grid job checks the tables of its lines and reference presets, with the clock mode of each preset, on its start set
    with GRID_EXTRA and on the development starts of its fillers, and computes none: the tables job is the one process that
    writes them on a clone. A missing table stops the job before its first line with the line that names the tables job."""
    settings = 'GRID_PATH=grid.txt REFERENCE_CONFIGS="base timed:instant" GRID_STARTS=sampled GRID_EXTRA=--start-cap=2'
    done = _bash(tmp_path, 'python() { echo "PY $*"; }', _function(RUNNER, "tables_ready"), settings, "tables_ready")
    script = "PY -m CausalDiagnosis.experiments.cluster_runs.precompute_grid_tables"
    assert done.returncode == 0 and done.stdout.splitlines() == [
        f"{script} --check --grid grid.txt --config base --config timed:instant --starts sampled --start-cap=2",
        f"{script} --check --starts development --grid grid.txt --config base --config timed:instant --start-cap=2"], done.stderr
    missing = 'python() { echo "PY $*"; case "$*" in *--check*) return 3 ;; esac; }'
    done = _bash(tmp_path, missing, _function(RUNNER, "tables_ready"), settings, "tables_ready")
    assert done.returncode == 4 and "TABLES_MISSING" in done.stderr and "precompute_tables.sbatch" in done.stderr
    assert [line for line in done.stdout.splitlines() if "--check" not in line] == []  # no call that computes a table


def test_every_job_reads_the_tables_and_the_tables_job_alone_writes_them():
    """fork_python, which every job runs, sets CD_TABLES_READ_ONLY=1 and PYTHONDONTWRITEBYTECODE=1, so that no job but the tables
    job writes a table or byte code into the clone that other jobs run, and the page names the tables job before the pilot."""
    fork_python = _function(FUNCTIONS, "fork_python")
    assert "export PYTHONDONTWRITEBYTECODE=1 CD_TABLES_READ_ONLY=1\n" in fork_python
    assert "export CD_TABLES_READ_ONLY=$read_only" in _function(FUNCTIONS, "grid_tables")
    cap, tables, pilot = (README.index(text) for text in ("python -m $C cap\n", "precompute_tables.sbatch --export=$PC\n", "GRID_PART=pilot\n"))
    assert cap < tables < pilot and "PC=ALL,GRID_STARTS=development,GRID_EXTRA=--start-cap=<cap>\n" in README


def test_the_reference_job_starts_couchdb_when_its_runs_name_that_store(tmp_path):
    starter = 'start_couchdb() { echo "START $1"; sleep 30 & COUCH_PID=$!; disown "$COUCH_PID"; echo "$COUCH_PID" > "$PWD/couch_pid"; }'
    trap = next(line for line in RUNNER.splitlines() if line.startswith("trap ") and "COUCH_PID" in line)
    stand_ins = ('node_local_dir() { echo "$PWD/local"; }', starter, _function(RUNNER, "couchdb_if_needed"), f"HPC_DIR={tmp_path}", "COUCH_PID=")
    done = _bash(tmp_path, *stand_ins, trap, 'GRID_EXTRA="--servers aob --store couchdb"', "couchdb_if_needed",
                 "SECONDS=0; wait; echo WAITED $SECONDS")
    started, waited = done.stdout.splitlines()
    assert done.returncode == 0 and started.startswith(f"START {tmp_path}/local/grid_couchdb_") and waited == "WAITED 0"
    pid = int((tmp_path / "couch_pid").read_text(encoding="utf-8"))
    for _ in range(100):  # the trap of the job stops the server by its pid when the job exits
        if subprocess.run(["kill", "-0", str(pid)], capture_output=True).returncode != 0:
            break
        subprocess.run(["/bin/sleep", "0.02"])
    else:
        subprocess.run(["kill", "-9", str(pid)], capture_output=True)
        pytest.fail("the server of CouchDB outlived the job")
    done = _bash(tmp_path, *stand_ins, 'GRID_EXTRA="--servers aob --store couchdb" COUCHDB_URL=http://127.0.0.1:5984', "couchdb_if_needed")
    assert done.returncode == 0 and done.stdout == ""
    done = _bash(tmp_path, *stand_ins, 'GRID_EXTRA="--servers aob --store memory"', "couchdb_if_needed")
    assert done.returncode == 0 and done.stdout == ""
    couchdb = (PACKAGE / "frameworks" / "AssetOpsBench" / "cluster_jobs" / "start_couchdb.sh").read_text(encoding="utf-8")
    assert 'disown "$COUCH_PID"' in couchdb  # the job's own wait for its runs never waits for CouchDB


def test_the_job_record_script_writes_the_record_of_each_named_folder(tmp_path):
    """It writes the accounting record of the job of each folder and copies the logs of that job from LOGS into job_logs of the
    folder, the Slurm log and for a serving job the logs of its servers and the file of its GPU sampler, gpu-<job id>.csv, so
    the logs travel with the runs; the log of another job stays out, and a job with no log under LOGS is named."""
    tree, root, fake, env = _tree(tmp_path, ("job_record.sh",))
    _stand_in(fake, "sacct", 'echo "$2|COMPLETED|2026-01-01T00:00:00|2026-01-01T01:00:00|01:00:00|00:00:00|node7"')
    folders = [root / "runs" / name for name in ("grid_101", "gepa_online_202", "grid_303_attempt1")]
    for folder in folders:
        folder.mkdir(parents=True)
    logs = {"grid_101": ["cd-serve-101.out", "gpu-101.csv", "sglang-101.log", "sglang-judge-101.log"],
            "gepa_online_202": ["cd-serve-202.out"], "grid_303_attempt1": ["cd-grid-303.out"]}
    (root / "logs").mkdir()
    for name in [log for names in logs.values() for log in names] + ["cd-serve-1010.out", "sglang-2101.log", "gpu-1101.csv"]:
        (root / "logs" / name).write_text(f"log {name}\n", encoding="utf-8")
    done = subprocess.run(["bash", str(_in(tree, "job_record.sh")), *map(str, folders)], capture_output=True, text=True,
                          cwd=tmp_path, env=env)
    assert done.returncode == 0, done.stderr
    assert [(folder / "job_record.txt").read_text(encoding="utf-8").split("|")[0] for folder in folders] == ["101", "202", "303"]
    assert len([line for line in done.stdout.splitlines() if line.startswith("JOB_RECORD ")]) == 3
    assert {folder.name: sorted(p.name for p in (folder / "job_logs").iterdir()) for folder in folders} == logs
    assert (folders[0] / "job_logs" / "sglang-101.log").read_text(encoding="utf-8") == "log sglang-101.log\n"
    lost = root / "runs" / "grid_404"
    lost.mkdir()
    done = subprocess.run(["bash", str(_in(tree, "job_record.sh")), str(lost)], capture_output=True, text=True, cwd=tmp_path, env=env)
    assert done.returncode == 2 and "no log of job 404" in done.stderr and (lost / "job_record.txt").exists()
    other = root / "runs" / "elsewhere"
    other.mkdir()
    assert subprocess.run(["bash", str(_in(tree, "job_record.sh")), str(other)], capture_output=True, text=True, cwd=tmp_path,
                          env=env).returncode == 2


# ----- the settings line and the directives of every script, the environment, the judge, the CPU jobs and the checks -----

DIRECTIVE = re.compile(r"^#SBATCH\s+(-A|-p|-w|-o|--account|--partition|--nodelist|--exclude|--output)\b", re.M)
JOB_SCRIPTS = ("check_endpoint.sbatch", "check_metering_layer.sh", "check_tables.sh", "cpu_job.sbatch", "download_model.sbatch", "fetch_data.sbatch",
               "platform_record.sh", "precompute_tables.sbatch", "pull_image.sbatch", "run_reproduction.sh", "run_gepa_online.sh", "run_grid.sbatch",
               "run_judge.sh", "run_lines.sh", "run_suite.sbatch", "serve_model.sbatch", "setup_env.sbatch", "setup_train_env.sbatch",
               "train_lora.sbatch")
SETTINGS_LINE = 'source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"'


def test_no_script_names_an_account_a_partition_a_node_or_a_log_file_in_a_directive():
    assert [(script.name, match.group(0)) for script in SCRIPTS for match in DIRECTIVE.finditer(script.read_text(encoding="utf-8"))] == []


def test_the_lines_job_tells_a_file_of_comments_from_a_file_with_a_pending_line_under_pipefail(tmp_path):
    """run_lines.sh runs under bash with set -uo pipefail, and its test of a file with nothing pending reads the file with one
    awk: a file of comments and blank lines plays nothing, a file with one pending line plays, and a long file with many
    pending lines plays too, which a grep -q at the end of a pipe could read as nothing pending once it closed the pipe."""
    script = (CLUSTER_JOBS / PLACE["run_lines.sh"]).read_text(encoding="utf-8")
    assert script.splitlines()[script.splitlines().index("set -uo pipefail") + 1].startswith('source "${HPC_DIR:?')
    check = next(line for line in script.splitlines() if line.startswith("if ! awk "))
    files = {"comments": "# a comment\n\n   \n#--config base\n", "one": "# a comment\n--config base --agent causal\n",
             "long": "--config base --agent causal\n" * 200000}
    for name, text in files.items():
        (tmp_path / f"{name}.txt").write_text(text, encoding="utf-8")
    read = {name: subprocess.run(["bash", "-c", f'set -uo pipefail; PLAY_PATH={tmp_path}/{name}.txt\n{check}\n  echo EMPTY\nelse\n  echo PENDING\nfi'],
                                 capture_output=True, text=True).stdout.strip() for name in files}
    assert read == {"comments": "EMPTY", "one": "PENDING", "long": "PENDING"}


def test_every_job_script_reads_its_settings_through_hpc_dir():
    texts = {name: (CLUSTER_JOBS / PLACE[name]).read_text(encoding="utf-8") for name in JOB_SCRIPTS}
    assert [name for name, text in texts.items() if SETTINGS_LINE not in text or "HPC_DIR:-" in text] == []


def test_the_environment_job_syncs_the_lock_frozen_and_no_job_installs_a_package():
    setup = (CLUSTER_JOBS / "0_setup" / "setup_env.sbatch").read_text(encoding="utf-8")
    assert "uv sync --frozen" in setup and "pip install" not in setup
    assert "check_data" not in setup  # the environment alone: the data of record is built after it, by fetch_data.sbatch
    install = re.compile(r"pip install|ensure_package")
    assert [name for name in ("run_lines.sh", "run_gepa_online.sh", "check_metering_layer.sh", "run_reproduction.sh", "start_couchdb.sh")
            if install.search((CLUSTER_JOBS / PLACE[name]).read_text(encoding="utf-8"))] == []


REJUDGE = """import json, os, sys
from pathlib import Path
calls = Path(sys.argv[0]).with_suffix(".calls")
number = len(calls.read_text().splitlines()) + 1 if calls.exists() else 1
calls.open("a").write(" ".join(sys.argv[1:]) + "\\n")
calls.with_suffix(".endpoint").open("a").write(os.environ.get("HYD_JUDGE_ENDPOINT_FILE", "none") + "\\n")
run = Path(sys.argv[1]) / "run_a"
mode = open(sys.argv[0] + ".mode").read()
if "--crash" in mode:
    sys.exit(1)
records = {1: {"ep1": None, "ep2": "the endpoint did not answer"}, 3: {"ep2": None} if "--fixed" in mode else {}}
for episode, error in records.get(number, {}).items():
    (run / episode / "hidden").mkdir(parents=True, exist_ok=True)
    (run / episode / "hidden" / "judge.json").write_text(json.dumps({"calls": [{"error": error}], "pass": True}))
print(f"run_a: judged {len(records.get(number, {})) if number == 1 else 0} episode(s)")
"""
# the count of judge_finished_runs.py, which the stand-in answers alone: the episodes of the runs of agent causal, judged with a record, failed
# with an error in it, left with trajectory.json and not judged without; a run whose run.json says refused is refused
REJUDGE_COUNT = """
def count(folders, names):
    left, judged, failed, calls, not_judged, refused = [], [], [], [], [], []
    for meta in sorted(p for folder in folders for p in Path(folder).glob("*/run.json")):
        run = json.loads(meta.read_text())
        if run.get("agent") != "causal":
            continue
        episodes = [e for e in sorted(meta.parent.iterdir()) if (e / "released.json").is_file()]
        if run.get("refused"):
            refused += episodes
            continue
        for e in episodes:
            record = e / "hidden" / "judge.json"
            if record.is_file():
                judged.append(e)
                calls += json.loads(record.read_text())["calls"]
                failed += [e] if any(c.get("error") for c in json.loads(record.read_text())["calls"]) else []
            else:
                (left if (e / "trajectory.json").is_file() else not_judged).append(e)
    if names:
        print("".join(f"JUDGE_NOT_JUDGED_EPISODE {e}\\n" for e in not_judged) + "".join(f"JUDGE_FAILED_EPISODE {e}\\n" for e in failed), end="")
    else:
        print(len(left), len(judged), len(failed), len(calls), sum(1 for c in calls if c.get("error")), len(not_judged), len(refused))
if "--count" in sys.argv or "--names" in sys.argv:
    count([a for a in sys.argv[1:] if not a.startswith("--") and Path(a).is_dir()], "--names" in sys.argv)
    sys.exit(0)
"""
JUDGE = """import json, os, sys
from pathlib import Path
if "--thinking" in sys.argv:
    print("JUDGE_THINKING " + os.environ.get("FAKE_THINKING", "0"))
    sys.exit(0)
for episode in sys.argv[1:]:
    if not episode.startswith("--") and Path(episode).is_dir():
        (Path(episode) / "hidden").mkdir(exist_ok=True)
        (Path(episode) / "hidden" / "judge.json").write_text(json.dumps({"calls": [{"error": None}], "pass": True}))
"""


REJUDGE_MODULE = "CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges.judge_finished_runs"  # the module of judge_finished_runs.py, which run_judge.sh runs


def _judge_tree(tmp_path, mode=""):
    tree, root, fake, env = _tree(tmp_path, ("run_judge.sh",))
    rejudge = _module_file(tree, REJUDGE_MODULE, REJUDGE.replace("from pathlib import Path\n", "from pathlib import Path\n" + REJUDGE_COUNT, 1))
    Path(f"{rejudge}.mode").write_text(mode, encoding="utf-8")
    _module_file(tree, "CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges.episode_judge", JUDGE)
    grid = root / "runs" / "development" / "grid_7"
    for run, agent in (("run_a", "causal"), ("run_b", "oracle")):
        (grid / run).mkdir(parents=True)
        (grid / run / "run.json").write_text(json.dumps({"agent": agent}), encoding="utf-8")
        for episode in ("ep1", "ep2"):
            (grid / run / episode).mkdir()
            (grid / run / episode / "released.json").write_text("{}", encoding="utf-8")
            (grid / run / episode / "trajectory.json").write_text("[]", encoding="utf-8")
    endpoint = tmp_path / "endpoint.txt"
    endpoint.write_text("http://127.0.0.1:30123/v1\njob 7 node node7 model judge served_name judge\n", encoding="utf-8")
    _stand_in(fake, "curl", 'echo "{\\"data\\": []}"')
    env = {**env, "MODEL": "/models/judge", "JUDGE_MODEL": "/models/judge", "BASE_URL": "http://127.0.0.1:30123/v1",
           "ENDPOINT_FILE": str(endpoint), "SLURM_JOB_ID": "7", "API_KEY": "k" * 64}
    return tree, grid, env


def _judge(tree, env):
    return subprocess.run(["bash", str(_in(tree, "run_judge.sh"))], capture_output=True, text=True, cwd=tree.parent, env=env, timeout=60)


def test_a_judge_job_of_record_stops_when_its_judge_is_not_the_one_of_the_serving_file(tmp_path):
    """The made-up endpoint file names no weights commit, engine or launch flags, so the serving check of the judge differs: with
    RUN_KIND=record the job stops with status 2 before any pass of judge_finished_runs.py, and any other job names the difference on one line
    and judges on."""
    tree, grid, env = _judge_tree(tmp_path / "record")
    done = _judge(tree, {**env, "JUDGE_FOLDERS": str(grid), "RUN_KIND": "record"})
    assert done.returncode == 2 and "SERVING_DIFFERS the judge is not the one of the serving file of record" in done.stderr, done.stderr
    assert not _module_file(tree, REJUDGE_MODULE).with_suffix(".calls").exists()
    tree, grid, env = _judge_tree(tmp_path / "development")
    done = _judge(tree, {**env, "JUDGE_FOLDERS": str(grid), "RUN_KIND": "development"})
    assert sum(line.startswith("SERVING_DIFFERS judge, a job of kind development judges on") for line in done.stdout.splitlines()) == 1
    assert "JUDGE_PASS 1" in done.stdout, done.stderr


def test_the_judge_job_repeats_until_nothing_is_left_and_counts_failed_calls(tmp_path):
    for mode, failed in (("", 1), ("--fixed", 0)):
        tree, grid, env = _judge_tree(tmp_path / (mode or "plain"), mode)
        done = _judge(tree, {**env, "JUDGE_FOLDERS": str(grid), "JUDGE_WORKERS": "3", "JUDGE_PROTOCOL": "single"})
        assert done.returncode == failed and f"JUDGE_FAILED_CALLS {failed}" in done.stdout and "CD_JUDGE_DONE" in done.stdout, done.stderr
        assert "JUDGE_LEFT 0" in done.stdout
        calls = _module_file(tree, REJUDGE_MODULE).with_suffix(".calls").read_text(encoding="utf-8").splitlines()
        assert len(calls) == 3 and calls[-1].endswith("--rejudge-failed") and not any("--rejudge-failed" in call for call in calls[:-1])
        assert all(f"{grid} --judge-model judge --judge-endpoint http://127.0.0.1:30123/v1 --protocol single --workers 3" in call for call in calls)
        assert "JUDGE_PLAN episodes 2 " in done.stdout and "JUDGE_PASS 1 judged 2" in done.stdout and "JUDGE_PASS 2 judged 0" in done.stdout
    assert (Path(env["ROOT"]) / "platform").is_dir() and any((Path(env["ROOT"]) / "platform").glob("judge_endpoint_*.txt"))
    # a second submission of judged folders: its first pass judges nothing, and the job ends after it
    done = _judge(tree, {**env, "JUDGE_FOLDERS": str(grid)})
    assert done.returncode == 0 and "JUDGE_PASS 1 judged 0" in done.stdout and "JUDGE_PASS 2" not in done.stdout, done.stderr
    tree, grid, env = _judge_tree(tmp_path / "crash", "--crash")
    done = _judge(tree, {**env, "JUDGE_FOLDERS": str(grid)})
    assert done.returncode == 1 and "judge_finished_runs.py failed" in done.stderr and "JUDGE_FAILED_CALLS" not in done.stdout  # a pass that fails fails the job
    # an episode that a pass could judge but did not stays without a record, which a table scores false, so it fails the job as a
    # failed call does; an episode that no pass can judge, here one whose run left no trajectory.json, is counted and named
    # apart and leaves the status alone
    tree, grid, env = _judge_tree(tmp_path / "left", "--fixed")
    for episode, trajectory in (("ep3", True), ("ep4", False)):
        (grid / "run_a" / episode).mkdir()
        (grid / "run_a" / episode / "released.json").write_text("{}", encoding="utf-8")
        if trajectory:
            (grid / "run_a" / episode / "trajectory.json").write_text("[]", encoding="utf-8")
    done = _judge(tree, {**env, "JUDGE_FOLDERS": str(grid)})
    assert done.returncode == 1 and "JUDGE_LEFT 1" in done.stdout and "JUDGE_FAILED_CALLS 0" in done.stdout and "CD_JUDGE_DONE" in done.stdout
    assert "JUDGE_PLAN episodes 3 " in done.stdout and "JUDGE_NOT_JUDGED 1" in done.stdout
    assert f"JUDGE_NOT_JUDGED_EPISODE {grid / 'run_a' / 'ep4'}" in done.stdout.splitlines()
    shutil.rmtree(grid / "run_a" / "ep3")
    done = _judge(tree, {**env, "JUDGE_FOLDERS": str(grid)})
    assert done.returncode == 0 and "JUDGE_LEFT 0" in done.stdout and "JUDGE_NOT_JUDGED 1" in done.stdout, done.stdout


def test_the_judge_job_leaves_a_refused_run_out_of_what_is_left(tmp_path):
    """What is left to judge is what judge_finished_runs.py selects: the episodes of a run that the pass refuses (its agent model is the judge,
    or a registered run whose judge is not the judge of the serving file) are counted and named apart and leave the status alone,
    since no pass can judge them."""
    tree, grid, env = _judge_tree(tmp_path, "--fixed")
    (grid / "run_c" / "ep1").mkdir(parents=True)
    (grid / "run_c" / "run.json").write_text(json.dumps({"agent": "causal", "refused": True}), encoding="utf-8")
    (grid / "run_c" / "ep1" / "released.json").write_text("{}", encoding="utf-8")
    (grid / "run_c" / "ep1" / "trajectory.json").write_text("[]", encoding="utf-8")
    done = _judge(tree, {**env, "JUDGE_FOLDERS": str(grid)})
    assert done.returncode == 0 and "JUDGE_LEFT 0" in done.stdout and "JUDGE_REFUSED 1" in done.stdout, done.stdout
    calls = _module_file(tree, REJUDGE_MODULE).with_suffix(".calls").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 3  # the passes alone: the counts judge nothing


def test_the_judge_job_holds_no_truncation_pass_and_every_judging_reads_the_endpoint_file(tmp_path):
    """The judge reads the whole input and cuts only past the context of the served judge, so the judge job runs no pass that
    finds lengths: JUDGE_TRUNCATION is no switch of it. judge_finished_runs.py reads the endpoint file of the judge through
    HYD_JUDGE_ENDPOINT_FILE, so each record names the judge it came from."""
    assert "TRUNCATION" not in (CLUSTER_JOBS / "3_run" / "run_judge.sh").read_text(encoding="utf-8") and "JUDGE_TRUNCATION" not in README
    tree, grid, env = _judge_tree(tmp_path)
    done = _judge(tree, {**env, "JUDGE_FOLDERS": str(grid), "JUDGE_PROTOCOL": "single"})
    endpoint = _module_file(tree, REJUDGE_MODULE).with_suffix(".endpoint").read_text(encoding="utf-8").splitlines()
    assert endpoint and set(endpoint) == {env["ENDPOINT_FILE"]}, done.stderr


def test_the_first_judge_job_keeps_the_endpoint_record_alone(tmp_path):
    """With JUDGE_RECORD_ONLY=1 the judge job writes the endpoint record of the judge into PLATFORM, which the platform record
    needs, and judges nothing."""
    tree, grid, env = _judge_tree(tmp_path)
    done = _judge(tree, {**env, "JUDGE_RECORD_ONLY": "1"})
    assert done.returncode == 0 and "JUDGE_RECORD_ONLY" in done.stdout and "CD_JUDGE_DONE" in done.stdout, done.stderr
    assert any((Path(env["ROOT"]) / "platform").glob("judge_endpoint_*.txt"))
    assert not _module_file(tree, REJUDGE_MODULE).with_suffix(".calls").exists() and not list(grid.glob("*/*/hidden/judge.json"))


def test_the_judge_smoke_changes_no_run_folder(tmp_path):
    tree, grid, env = _judge_tree(tmp_path)

    def hashes():
        return {p.relative_to(grid): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(grid.rglob("*")) if p.is_file()}

    before = hashes()
    done = _judge(tree, {**env, "JUDGE_SMOKE": str(grid)})
    smoke = next(line for line in done.stdout.splitlines() if line.startswith("JUDGE_SMOKE "))
    assert done.returncode == 0 and smoke.split()[1:8] == ["episodes", "2", "calls", "2", "failed", "0", "seconds"], done.stderr
    assert hashes() == before
    smoke = Path(env["ROOT"]) / "runs" / "smoke" / "judge_7"
    assert sorted(p.relative_to(smoke).as_posix() for p in smoke.glob("*/*/hidden/judge.json")) == ["run_a/ep1/hidden/judge.json",
                                                                                                    "run_b/ep1/hidden/judge.json"]


GEPA_ONLINE = """import sys
from pathlib import Path
if sys.argv[1:2] != ["export"]:
    Path(sys.argv[0]).with_suffix(".args").open("a").write(" ".join(sys.argv[1:]) + "\\n")
"""


def test_a_loop_job_reads_the_endpoint_of_the_reflection_job_by_its_id(tmp_path):
    tree, root, fake, env = _tree(tmp_path, ("run_gepa_online.sh",))
    _services(tree).write_text("start_couchdb() { COUCH_PID=; }\n", encoding="utf-8")
    _module_file(tree, gepa_online.__name__, GEPA_ONLINE)
    # the clock check of the loop job, a stand-in of grid_job_lists.py that names no line with the wall clock
    _module_file(tree, grid_lines.__name__, "")
    _stand_in(fake, "curl", "exit 0")
    _stand_in(fake, "squeue", "echo RUNNING")
    endpoints = root / "endpoints"
    endpoints.mkdir(parents=True)
    for job, url in (("100", "http://10.0.0.1:31000/v1"), ("200", "http://10.0.0.2:32000/v1")):
        (endpoints / f"{job}.txt").write_text(f"{url}\njob {job} node n{job} model judge served_name judge\n", encoding="utf-8")
    (endpoints / "judge.txt").symlink_to(endpoints / "200.txt")
    settings = {**env, "JUDGE_MODEL": "/models/judge", "BASE_URL": "http://127.0.0.1:30123/v1", "GEPA_FOLDS": "1", "GEPA_AGENTS": "causal",
                "GEPA_CARDS": "verifier", "SLURM_JOB_ID": "300"}
    script = str(_in(tree, "run_gepa_online.sh"))
    done = subprocess.run(["bash", script], capture_output=True, text=True, cwd=tmp_path, env=settings, timeout=60)
    assert done.returncode == 2 and "REFLECTION_JOB=<job id>" in done.stderr
    done = subprocess.run(["bash", script], capture_output=True, text=True, cwd=tmp_path, env={**settings, "REFLECTION_JOB": "100"}, timeout=60)
    assert done.returncode == 0 and "CD_GEPA_ONLINE_DONE" in done.stdout, done.stderr
    args = _module_file(tree, gepa_online.__name__).with_suffix(".args").read_text(encoding="utf-8").split()
    assert args[args.index("--reflection-base-url") + 1] == "http://10.0.0.1:31000/v1"


CURL = """code=${FAKE_OPEN:-401}
case " $* " in *" -K "*) code=200 ;; esac
case " $* " in *" -w "*) printf "%s" "$code" ;; esac"""


def test_the_endpoint_check_tells_a_closed_endpoint_from_an_open_one(tmp_path):
    tree, root, fake, env = _tree(tmp_path, ("check_endpoint.sbatch",))
    _stand_in(fake, "curl", CURL)
    _stand_in(fake, "squeue", "echo RUNNING")
    _stand_in(fake, "hostname", 'case "${1:-}" in -s) echo node7 ;; *) echo node7.cluster ;; esac')
    endpoints = root / "endpoints"
    endpoints.mkdir(parents=True)
    (endpoints / "900.key").write_text("k" * 64 + "\n", encoding="utf-8")

    def check(node, **values):
        (endpoints / "900.txt").write_text(f"http://10.1.2.3:30123/v1\njob 900 node {node} model m served_name m\n", encoding="utf-8")
        done = subprocess.run(["bash", str(_in(tree, "check_endpoint.sbatch"))], capture_output=True, text=True, cwd=tmp_path,
                              env={**env, "SERVE_JOB": "900", **values}, timeout=60)
        stopped = (endpoints / "900.stop").exists()
        (endpoints / "900.stop").unlink(missing_ok=True)
        return done, stopped

    done, stopped = check("node3")
    assert done.returncode == 0 and "HTTP_BETWEEN_NODES ok" in done.stdout and "ENDPOINT_CLOSED ok" in done.stdout and stopped, done.stderr
    assert "CD_ENDPOINT_DONE" in done.stdout and "k" * 64 not in done.stdout
    done, stopped = check("node3", FAKE_OPEN="200")
    assert done.returncode == 1 and "ENDPOINT_OPEN" in done.stdout and stopped
    done, _ = check("node7")
    assert "HTTP_BETWEEN_NODES same node" in done.stdout


def test_the_cpu_job_runs_its_command_and_ends_with_its_status(tmp_path):
    tree, _root, _fake, env = _tree(tmp_path, ("cpu_job.sbatch",))
    _services(tree).write_text('start_couchdb() { echo START_COUCHDB; COUCH_PID=; }\n', encoding="utf-8")
    job = [str(_in(tree, "cpu_job.sbatch")), "bash", "-c", 'echo "COMMAND in $PWD"; exit 6']
    done = subprocess.run(["bash", *job], capture_output=True, text=True, cwd=tmp_path, env=env, timeout=60)
    assert done.returncode == 6 and "CD_CPU_JOB_DONE 6" in done.stdout and f"COMMAND in {tree}" in done.stdout, done.stderr
    assert "START_COUCHDB" not in done.stdout
    done = subprocess.run(["bash", *job], capture_output=True, text=True, cwd=tmp_path, env={**env, "CPU_COUCHDB": "1"}, timeout=60)
    lines = done.stdout.splitlines()
    assert done.returncode == 6 and lines.index("START_COUCHDB") < lines.index(f"COMMAND in {tree}")
    # with CPU_TABLES=1 the job first checks the tables of every grid file of the folder, as the tables job leaves them
    grids = _jobs(tree) / "grids"
    grids.mkdir()
    for name in ("grid_a.txt", "grid_b.txt"):
        (grids / name).write_text("--config base --agent causal\n", encoding="utf-8")
    _module_file(tree, "CausalDiagnosis.experiments.cluster_runs.precompute_grid_tables",
                 "import os, sys\nprint('TABLES ' + os.getcwd() + ' ' + ' '.join(sys.argv[1:]))\n")
    done = subprocess.run(["bash", *job], capture_output=True, text=True, cwd=tmp_path, env={**env, "CPU_TABLES": "1"}, timeout=60)
    lines = done.stdout.splitlines()
    tables = f"TABLES {tree} --check --grid-presets --grid {grids / 'grid_a.txt'} --grid {grids / 'grid_b.txt'}"
    assert done.returncode == 6 and tables in lines and lines.index(tables) < lines.index(f"COMMAND in {tree}"), done.stdout


def test_the_training_job_checks_its_card_and_hands_it_to_the_trainer(tmp_path):
    """The training job checks its card as a serving job does, before the trainer starts: in a registered tree on another
    card than the card of record it prints GPU_NOT_OF_RECORD and ends with status 2 without a step, and on the card of record,
    or in a development tree whose env.sh names no card, it hands the line of its card to train_lora.py in HYD_TRAINING_CARD,
    which the trainer writes into the training manifest beside the adapter."""
    tree, root, fake, env = _tree(tmp_path, ("train_lora.sbatch",))
    log = tmp_path / "log"
    log.mkdir()
    venv = root / "train_venv" / "bin"
    venv.mkdir(parents=True)
    _stand_in(venv, "python", f'echo "TRAIN $HYD_TRAINING_CARD" >> "{log}/train"')
    _stand_in(fake, "nvidia-smi", CARD)
    _stand_in(fake, "hostname", "echo node7")
    model = root / "models" / "agent"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    job = ["bash", str(_in(tree, "train_lora.sbatch"))]
    smoke = {**env, "MODEL": str(model), "SMOKE": "1", "FAKE_GPU": "NVIDIA TEST PCIe", "FAKE_MIB": "2000"}
    record = {"GPU_OF_RECORD": "NVIDIA TEST NVL", "GPU_MIB_OF_RECORD": "1000"}
    done = subprocess.run(job, capture_output=True, text=True, cwd=tmp_path, env=smoke, timeout=60)
    assert done.returncode == 0 and "CD_TRAIN_DONE smoke" in done.stdout, done.stderr
    line = "gpu_name NVIDIA_TEST_PCIe gpu_mib 2000 gpu_uuid GPU-1 driver_version 1.2.3 cuda_version 9.9"
    assert (log / "train").read_text(encoding="utf-8") == f"TRAIN {line}\n"
    (log / "train").unlink()
    _record(tree, "RECORD")
    done = subprocess.run(job, capture_output=True, text=True, cwd=tmp_path, env={**smoke, **record}, timeout=60)
    assert done.returncode == 2 and "GPU_NOT_OF_RECORD node7 NVIDIA TEST PCIe 2000" in done.stdout, done.stderr
    assert not (log / "train").exists()
    done = subprocess.run(job, capture_output=True, text=True, cwd=tmp_path, env={**smoke, **record, "FAKE_GPU": "NVIDIA TEST NVL", "FAKE_MIB": "1000"},
                          timeout=60)
    assert done.returncode == 0 and (log / "train").read_text(encoding="utf-8").startswith("TRAIN gpu_name NVIDIA_TEST_NVL gpu_mib 1000 "), done.stderr


def test_a_cpu_job_on_a_node_with_a_card_checks_the_card_of_record_on_request(tmp_path):
    """With CPU_CARD_OF_RECORD=1 the CPU job checks that the card of its node is the card of record, which env.sh must name, and
    stops with GPU_NOT_OF_RECORD and status 2 before its command on another card: the job of the label of record of the digests,
    which check_nodes.sh label submits so."""
    tree, _root, fake, env = _tree(tmp_path, ("cpu_job.sbatch",))
    _stand_in(fake, "nvidia-smi", CARD)
    _stand_in(fake, "hostname", "echo node7")
    job = ["bash", str(_in(tree, "cpu_job.sbatch")), "bash", "-c", "echo COMMAND"]
    record = {"GPU_OF_RECORD": "NVIDIA TEST NVL", "GPU_MIB_OF_RECORD": "1000", "CPU_CARD_OF_RECORD": "1"}
    for extra, status, printed in (({**record, "FAKE_GPU": "NVIDIA TEST PCIe", "FAKE_MIB": "2000"}, 2, "GPU_NOT_OF_RECORD node7 NVIDIA TEST PCIe 2000"),
                                   ({"CPU_CARD_OF_RECORD": "1"}, 2, "GPU_OF_RECORD is empty"), (record, 0, "CD_CPU_JOB_DONE 0")):
        done = subprocess.run(job, capture_output=True, text=True, cwd=tmp_path, env={**env, **extra}, timeout=60)
        assert done.returncode == status and printed in done.stdout + done.stderr, (extra, done.stdout, done.stderr)
        assert ("COMMAND" in done.stdout.splitlines()) == (status == 0), extra


PLATFORM_LEDGER = """import sys
out = sys.argv[sys.argv.index("--out") + 1]
open(out, "w").write("platform record")
import json, os
names = ("HYD_MODEL", "HYD_HUB_ID", "HYD_REVISION", "HYD_JUDGE_MODEL", "HYD_JUDGE_HUB_ID", "HYD_JUDGE_REVISION")
open(out + ".names", "w").write(json.dumps({name: os.environ.get(name) for name in names}))
"""


def test_the_platform_record_needs_the_card_of_record_on_both_endpoints(tmp_path):
    """The platform record names the card of the agent and of the judge, so the record needs both settings of
    the card in env.sh and stops with GPU_NOT_OF_RECORD and status 2, before any record, when the endpoint file of its serving
    job or the judge endpoint kept for the tree names another card, and with status 2 when no judge endpoint is kept. It hands
    run_ledger/ the models of env.sh to check the two endpoint records against: the served names, which the run and judge
    records carry, the Hugging Face ids and the pinned commits."""
    tree, root, fake, env = _tree(tmp_path, ("platform_record.sh",))
    _services(tree).write_text("couchdb_image() { :; }\n", encoding="utf-8")
    _module_file(tree, "CausalDiagnosis.experiments.run_verification.run_ledger", PLATFORM_LEDGER)
    tree_id = hashlib.sha256((tree / "UPLOAD_MANIFEST.txt").read_bytes()).hexdigest()[:12]
    platform = root / "platform"
    platform.mkdir(parents=True)

    def endpoint(path, card):
        path.write_text(f"http://127.0.0.1:1/v1\njob 55 node node7 model m served_name m\nstarted now\n{card} gpu_uuid GPU-1 "
                        "driver_version 1.2.3 cuda_version 9.9 gpu_sampler s.csv\n", encoding="utf-8")
        return path

    agent = endpoint(tmp_path / "agent.txt", "gpu_name NVIDIA_TEST_NVL gpu_mib 1000")
    judge = platform / f"judge_endpoint_{tree_id}.txt"
    record = {**env, "ENDPOINT_FILE": str(agent), "GPU_OF_RECORD": "NVIDIA TEST NVL", "GPU_MIB_OF_RECORD": "1000"}
    job = ["bash", str(_in(tree, "platform_record.sh"))]
    done = subprocess.run(job, capture_output=True, text=True, cwd=tmp_path, env=record, timeout=60)
    assert done.returncode == 2 and "no judge endpoint record" in done.stderr and not (platform / f"{tree_id}.json").exists(), done.stderr
    for case, card, settings in (("judge", "gpu_name NVIDIA_TEST_PCIe gpu_mib 2000", {}), ("settings", "gpu_name NVIDIA_TEST_NVL gpu_mib 1000",
                                                                                          {"GPU_MIB_OF_RECORD": ""})):
        endpoint(judge, card)
        done = subprocess.run(job, capture_output=True, text=True, cwd=tmp_path, env={**record, **settings}, timeout=60)
        assert done.returncode == 2 and not (platform / f"{tree_id}.json").exists(), (case, done.stdout, done.stderr)
        assert ("GPU_NOT_OF_RECORD" in done.stdout) == (case == "judge"), case
    endpoint(agent, "gpu_name NVIDIA_TEST_PCIe gpu_mib 2000")
    done = subprocess.run(job, capture_output=True, text=True, cwd=tmp_path, env=record, timeout=60)
    assert done.returncode == 2 and "GPU_NOT_OF_RECORD" in done.stdout and str(agent) in done.stdout
    endpoint(agent, "gpu_name NVIDIA_TEST_NVL gpu_mib 1000")
    done = subprocess.run(job, capture_output=True, text=True, cwd=tmp_path, env=record, timeout=60)
    assert done.returncode == 0 and (platform / f"{tree_id}.json").read_text(encoding="utf-8") == "platform record", done.stderr
    agent_id, judge_id = "ibm-granite/granite-3.3-8b-instruct", "Qwen/Qwen3.8-27B-FP8"
    assert json.loads((platform / f"{tree_id}.json.names").read_text(encoding="utf-8")) == {
        "HYD_MODEL": "granite-3.3-8b-instruct", "HYD_HUB_ID": agent_id, "HYD_REVISION": PINS[agent_id],
        "HYD_JUDGE_MODEL": "Qwen3.8-27B-FP8", "HYD_JUDGE_HUB_ID": judge_id, "HYD_JUDGE_REVISION": PINS[judge_id]}


DOWNLOAD = """echo "$*" >> "$FAKE_LOG/uvx_calls"
target=$(echo "$*" | sed -n 's/.*--local-dir \\([^ ]*\\).*/\\1/p')
mkdir -p "$target/.cache/huggingface/download"
echo "{}" > "$target/config.json"
for f in config.json model.safetensors; do printf '%s\\nfeed\\n1\\n' "$FAKE_COMMIT" > "$target/.cache/huggingface/download/$f.metadata"; done
[ -z "${FAKE_OTHER:-}" ] || printf '%s\\nfeed\\n1\\n' "$FAKE_OTHER" > "$target/.cache/huggingface/download/tokenizer.json.metadata"
[ -z "${FAKE_FAIL:-}" ] || exit 1
"""
# the listing of the Hub at a pin as the download job reads it: the file that FAKE_LISTING names, else no large file
HUB_LISTING = 'if [ -n "${FAKE_LISTING:-}" ]; then cat "$FAKE_LISTING"; else echo "[]"; fi'


def test_a_model_download_takes_the_pinned_commit_alone(tmp_path):
    """The tracked serving file of record pins the Hub commit of the agent model and of the judge, which JUDGE_MODEL names by default, and
    download_model.sbatch fetches that commit alone: it refuses a model without a pin, takes a shared copy only when it holds
    the pin, writes the pin into HUB_COMMIT.txt and stops when the files it fetched come from another commit."""
    tree, root, fake, env = _tree(tmp_path, ("download_model.sbatch",))
    log = tmp_path / "log"
    log.mkdir()
    _stand_in(fake, "uvx", DOWNLOAD)
    _stand_in(fake, "curl", HUB_LISTING)
    done = _run(tmp_path, env, *_sourced(tree, 'echo "$MODEL $JUDGE_MODEL $(model_revision "$MODEL") $(model_revision "$JUDGE_MODEL")"'))
    agent, judge, agent_pin, judge_pin = done.stdout.split()
    assert (agent, judge) == ("ibm-granite/granite-3.3-8b-instruct", "Qwen/Qwen3.8-27B-FP8") and (agent_pin, judge_pin) == (PINS[agent], PINS[judge])
    assert all(re.fullmatch(r"[0-9a-f]{40}", pin) for pin in (agent_pin, judge_pin)) and agent_pin != judge_pin
    models = tmp_path / "models"
    job = ["bash", str(_in(tree, "download_model.sbatch"))]
    settings = {**env, "FAKE_LOG": str(log), "MODELS": str(models), "SHARED": str(tmp_path / "no_shared_folder")}

    def download(**extra):
        return subprocess.run(job, capture_output=True, text=True, cwd=tmp_path, env={**settings, **extra}, timeout=60)

    for model, pin in ((agent, agent_pin), (judge, judge_pin)):
        done = download(MODEL=model, FAKE_COMMIT=pin)
        assert done.returncode == 0 and f"CD_DOWNLOAD_DONE {models / model.split('/')[1]}" in done.stdout, done.stderr
        assert f"download {model} --local-dir {models / model.split('/')[1]} --revision {pin}" in _logged(settings, "uvx_calls")
        assert (models / model.split("/")[1] / "HUB_COMMIT.txt").read_text(encoding="utf-8") == f"{pin}\n"
    (log / "uvx_calls").unlink()
    for extra, message in (({"MODEL": "org/unpinned"}, "no pinned commit for org/unpinned"), ({"HF_REVISION": "main"}, "no pinned commit")):
        done = download(**extra)
        assert done.returncode == 2 and message in done.stderr and _logged(settings, "uvx_calls") is None, extra
    done = download(MODEL="org/other", FAKE_COMMIT="a" * 40, FAKE_OTHER="b" * 40, HF_REVISION="a" * 40)
    assert done.returncode == 3 and "not the pinned" in done.stderr and "CD_DOWNLOAD_DONE" not in done.stdout
    (log / "uvx_calls").unlink()
    shared = tmp_path / "shared" / "models" / "granite-3.3-8b-instruct"
    shared.mkdir(parents=True)
    (shared / "config.json").write_text("{}", encoding="utf-8")
    # a shared copy that records no commit is not used: the job fetches a copy of its own at the pin
    done = download(MODELS=str(tmp_path / "own_first"), SHARED=str(tmp_path / "shared"), FAKE_COMMIT=agent_pin)
    assert done.returncode == 0 and "records no commit" in done.stdout and "download " in (_logged(settings, "uvx_calls") or ""), done.stderr
    (log / "uvx_calls").unlink()
    for commit, status in (("0" * 40, 3), (agent_pin, 0)):
        (shared / "HUB_COMMIT.txt").write_text(f"{commit}\n", encoding="utf-8")
        done = download(MODELS=str(tmp_path / "own"), SHARED=str(tmp_path / "shared"))
        assert done.returncode == status and _logged(settings, "uvx_calls") is None, (commit, done.stdout, done.stderr)


def test_no_grid_job_waits_for_a_judge_and_a_job_that_serves_the_judge_judges_in_the_run(tmp_path):
    """judge_args gives the judge options of a run only when JUDGE_BASE_URL names the judge, which a co-served judge sets, or a
    remote route: with the judge of env.sh named and no such route it looks up no endpoint and gives no option, since run_judge.sh
    scores a grid after the grid. The reproduction judges in the run by the same rule, so no job waits for a judge served elsewhere."""
    tree, _root, _fake, env = _tree(tmp_path)
    done = _run(tmp_path, env, *_sourced(tree, 'endpoint_url() { echo LOOKUP >&2; }', 'echo "[$(judge_args)]"'))
    assert done.returncode == 0 and done.stdout.strip() == "[]" and "LOOKUP" not in done.stderr and "run_judge.sh" in done.stderr
    done = _run(tmp_path, {**env, "JUDGE_BASE_URL": "http://node:1/v1"}, *_sourced(tree, 'echo "[$(judge_args)]"'))
    assert done.stdout.strip() == "[--judge local --judge-model Qwen3.8-27B-FP8 --judge-base-url http://node:1/v1 --judge-protocol paper]"
    repro = (PACKAGE / "frameworks" / "AssetOpsBench" / "cluster_jobs" / "run_reproduction.sh").read_text(encoding="utf-8")
    assert 'if [ -n "$JUDGE_MODEL" ] && [ -n "${JUDGE_BASE_URL:-}" ]; then' in repro and "JUDGE_WAIT_TRIES" not in repro


def test_the_smoke_job_of_the_trainer_trains_into_a_folder_of_its_own(tmp_path):
    """With SMOKE=1 the training job reads no agent, arm, fold or examples: it runs train_lora.py --smoke in the training
    environment with the folder of the model, into ROOT/training_smoke and not into HYD_TRAINING, and ends with
    CD_TRAIN_DONE smoke. Without it the job still asks for its arm."""
    tree, root, fake, env = _tree(tmp_path, ("train_lora.sbatch",))
    log = tmp_path / "log"
    log.mkdir()
    venv = root / "train_venv" / "bin"
    venv.mkdir(parents=True)
    _stand_in(venv, "python", f'echo "TRAIN $PWD $*" >> "{log}/train"')
    _stand_in(fake, "nvidia-smi", "echo card")
    model = root / "models" / "agent"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    job = ["bash", str(_in(tree, "train_lora.sbatch"))]
    done = subprocess.run(job, capture_output=True, text=True, cwd=tmp_path, env={**env, "MODEL": str(model), "SMOKE": "1"}, timeout=60)
    assert done.returncode == 0 and done.stdout.split()[-2:] == ["CD_TRAIN_DONE", "smoke"], done.stderr
    smoke = f"TRAIN {tree} -m CausalDiagnosis.training.train_lora --smoke --training-root {root / 'training_smoke'} --base-model {model}\n"
    assert (log / "train").read_text(encoding="utf-8") == smoke
    done = subprocess.run(job, capture_output=True, text=True, cwd=tmp_path, env={**env, "MODEL": str(model)}, timeout=60)
    assert done.returncode == 2 and "ARM must be filtered or unfiltered" in done.stderr


SUITE_PYTHON = """#!{python}
import os, sys
from pathlib import Path
if sys.argv[1:3] == ["-m", "pytest"] and "--collect-only" in sys.argv:
    print("7 tests collected in 0.01s")
    sys.exit(0)
if sys.argv[1:3] == ["-m", "pytest"]:
    log = Path(sys.argv[0]).parent / "pytest_calls"
    with log.open("a") as f:
        f.write(" ".join(sys.argv[3:]) + "\\n" + " ".join(sorted(k for k in os.environ if not k.startswith("__"))) + "\\n")
    junit = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--junitxml="))
    Path(junit).write_text('<testsuites><testsuite><testcase classname="benchmarks.t" name="test_a"/></testsuite></testsuites>')
    sys.exit(0)
os.execv({python!r}, [{python!r}, *sys.argv[1:]])
"""


def test_the_suite_job_runs_the_roots_of_the_tree_in_a_clean_environment(tmp_path):
    tree, root, _fake, env = _tree(tmp_path, ("run_suite.sbatch",))
    _services(tree).write_text(
        "start_couchdb() { COUCH_PID=; export COUCHDB_URL=http://127.0.0.1:5984 COUCHDB_USERNAME=admin COUCHDB_PASSWORD=pw; }\n", encoding="utf-8")
    python = next((root / "envs").glob("*/bin/python"))
    python.write_text(SUITE_PYTHON.format(python=sys.executable), encoding="utf-8")
    fixed = {"HOME", "PATH", "PYTHONPATH", "PYTHONUTF8", "PYTHONDONTWRITEBYTECODE", "TMPDIR"}
    for extra, names in (({}, fixed), ({"SUITE_COUCHDB": "1", "SUITE_DIGESTS": "1"},
                                       fixed | {"COUCHDB_URL", "COUCHDB_USERNAME", "COUCHDB_PASSWORD", "ASSET_DIGESTS"})):
        log = python.parent / "pytest_calls"
        log.unlink(missing_ok=True)
        done = subprocess.run(["bash", str(_in(tree, "run_suite.sbatch"))], capture_output=True, text=True, cwd=tmp_path,
                              env={**env, "SLURM_JOB_ID": str(os.getpid()), "SUITE_COUNT": "7", **extra}, timeout=60)
        counts = next(line for line in done.stdout.splitlines() if line.startswith("SUITE_COUNTS "))
        assert done.returncode == 0 and counts.split()[1:] == ["passed", "1", "skipped", "0", "failed", "0", "errors", "0"], done.stderr
        assert "CD_SUITE_DONE" in done.stdout and "SUITE_COLLECTED 7, SUITE_COUNT 7" in done.stdout
        argv, environ = log.read_text(encoding="utf-8").splitlines()
        assert argv.split()[:3] == ["src", "CausalDiagnosis", "benchmarks"]
        # Python sets LC_CTYPE itself when it coerces the C locale of an empty environment, so it is no variable of the job
        assert f"--junitxml={root}/suites/" in argv and set(environ.split()) - {"LC_CTYPE"} == names, environ



def test_the_suite_job_stops_before_its_first_test_when_it_collects_another_count_than_the_mac(tmp_path):
    """SUITE_COUNT names the tests that the roots of the tree collect on the Mac for the commit: without it the job stops with
    status 2, and when the tree collects another number here it stops with SUITE_COUNT_MISMATCH and status 4 before its first test."""
    tree, root, _fake, env = _tree(tmp_path, ("run_suite.sbatch",))
    python = next((root / "envs").glob("*/bin/python"))
    python.write_text(SUITE_PYTHON.format(python=sys.executable), encoding="utf-8")
    for count, status, printed in (("", 2, "SUITE_COUNT=<n>"), ("8", 4, "SUITE_COUNT_MISMATCH the suite collects 7 tests here and 8 on the Mac")):
        done = subprocess.run(["bash", str(_in(tree, "run_suite.sbatch"))], capture_output=True, text=True, cwd=tmp_path,
                              env={**env, "SLURM_JOB_ID": str(os.getpid()), "SUITE_COUNT": count}, timeout=60)
        assert done.returncode == status and printed in done.stderr and not (python.parent / "pytest_calls").exists(), (count, done.stderr)


SUBMIT_STAND_IN = """n=$(( $(cat "$FAKE_LOG/jobs" 2>/dev/null || echo 500) + 1 )); echo "$n" > "$FAKE_LOG/jobs"
echo "$*" >> "$FAKE_LOG/submitted"
echo "SUBMITTED $1 $n $LOGS/cd-node-$n.out" """
SINFO = 'printf "%s\\n" "nodeA gpu:h100:2 mixed" "nodeB gpu:h100:2 down*" "nodeC gpu:l40s:8 idle" "nodeD gpu:h100:2(S:0-1) idle"'


def test_check_nodes_writes_the_label_of_record_then_checks_every_node_with_the_card_against_it(tmp_path):
    """The label of record is the label current of the tracked digest file, written in the clone by one CPU job on a node with
    the card of record, which the job checks first (CPU_CARD_OF_RECORD=1), under the commit of the clone. The check submits one job per node of that card that is up,
    each comparing its digests with that label, and reads back the nodes whose check found no changed key."""
    tree, root, fake, env = _tree(tmp_path, ("check_nodes.sh", "serve_model.sbatch"))
    log = tmp_path / "log"
    log.mkdir()
    _stand_in(_jobs(tree), "submit.sh", SUBMIT_STAND_IN)
    _stand_in(fake, "sinfo", SINFO)
    check = ["bash", str(_in(tree, "check_nodes.sh"))]
    settings = {**env, "FAKE_LOG": str(log), "PARTITION_GPU": "gpu"}
    done = subprocess.run([*check, "submit"], capture_output=True, text=True, cwd=tmp_path, env=settings)
    assert done.returncode == 2 and "check_nodes.sh label" in done.stderr and not (log / "submitted").exists()
    done = subprocess.run([*check, "label"], capture_output=True, text=True, cwd=tmp_path, env=settings)
    assert done.returncode == 0, done.stderr
    write = 'python -m CausalDiagnosis.experiments.run_verification.invariant_digests --facts "$DATA_FILE" --label current --write CausalDiagnosis/experiments/run_verification/invariant_digests.json --commit "$(git rev-parse HEAD)"'
    assert (log / "submitted").read_text(encoding="utf-8") == (
        f"cpu_job.sbatch -p gpu --gres=gpu:h100:1 -J cd-label --export=ALL,CPU_CARD_OF_RECORD=1 -- bash -c {write}\n")
    (log / "submitted").unlink()
    digests = tree / "CausalDiagnosis" / "experiments" / "run_verification" / "invariant_digests.json"
    digests.parent.mkdir(parents=True, exist_ok=True)
    digests.write_text('{"baselines": {}}', encoding="utf-8")
    done = subprocess.run([*check, "submit"], capture_output=True, text=True, cwd=tmp_path, env=settings)
    assert done.returncode == 0, done.stderr
    submitted = (log / "submitted").read_text(encoding="utf-8").splitlines()
    assert [line.split(" -w ")[1].split()[0] for line in submitted] == ["nodeA", "nodeD"] and "NODE_SKIPPED nodeB down*" in done.stdout
    assert all("cpu_job.sbatch -p gpu --gres=gpu:h100:1 -w " in line
               and "--label current --check CausalDiagnosis/experiments/run_verification/invariant_digests.json" in line for line in submitted)
    assert all("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits" in line for line in submitted)
    logs = root / "logs"
    (logs / "cd-node-502.out").write_text("GPU_OF_NODE NVIDIA TEST NVL, 1000\n12 same, 0 changed, 0 missing, 0 new against current\n"
                                          "CD_CPU_JOB_DONE 0\n", encoding="utf-8")
    (logs / "cd-node-503.out").write_text("GPU_OF_NODE NVIDIA TEST PCIe, 2000\n12 same, 0 changed, 0 missing, 0 new against current\n"
                                          "CD_CPU_JOB_DONE 0\n", encoding="utf-8")
    record = {**settings, "GPU_OF_RECORD": "NVIDIA TEST NVL", "GPU_MIB_OF_RECORD": "1000"}
    done = subprocess.run([*check, "read"], capture_output=True, text=True, cwd=tmp_path, env=record)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert lines[:5] == ["NODE nodeA 0 12 same, 0 changed, 0 missing, 0 new against current", "NODE_GPU nodeA NVIDIA TEST NVL 1000",
                         "NODE nodeD 0 12 same, 0 changed, 0 missing, 0 new against current", "NODE_GPU nodeD NVIDIA TEST PCIe 2000",
                         "NODE_OTHER_GPU nodeD NVIDIA TEST PCIe 2000"]
    assert 'NODES_OF_RECORD="nodeA"' in lines and "NODE_UNCHECKED nodeB" in lines and not any("nodeC" in line for line in lines)
    (logs / "cd-node-503.out").write_text("GPU_OF_NODE NVIDIA TEST NVL, 1000\n11 same, 1 changed, 0 missing, 0 new against current\n"
                                          "CD_CPU_JOB_DONE 1\n", encoding="utf-8")
    done = subprocess.run([*check, "read"], capture_output=True, text=True, cwd=tmp_path, env=record)
    assert 'NODES_OF_RECORD="nodeA"' in done.stdout.splitlines() and "NODE_OTHER_GPU" not in done.stdout
    done = subprocess.run([*check, "read"], capture_output=True, text=True, cwd=tmp_path, env=settings)
    assert done.returncode == 2 and "NODE_GPU nodeA NVIDIA TEST NVL 1000" in done.stdout and "NODES_OF_RECORD" not in done.stdout


CHECK_PYTHON = """case "$*" in
  *"grid_job_lists presets"*) echo "base timed:instant" ;;
  *episode_runners.run_episodes*) echo "$*" >> "$FAKE_LOG/runs"; [ -z "${FAKE_CHANGE:-}" ] || echo changed >> "$FORK_UNDER_TEST/$FAKE_CHANGE" ;;
  *) exec "$REAL_PYTHON" "$@" ;;
esac"""


def test_check_tables_names_a_changed_table(tmp_path):
    tree, root, _fake, env = _tree(tmp_path, ("check_tables.sh",))
    log = tmp_path / "log"
    log.mkdir()
    tables = ["CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/models.json", "CausalDiagnosis/mcp_servers/Causal_Diagnosis_MCP/offline/grades.json", "CausalDiagnosis/datasets/hyd_1/data/labels.json"]
    for table in tables:
        (tree / table).parent.mkdir(parents=True, exist_ok=True)
        (tree / table).write_text("{}", encoding="utf-8")
    (_jobs(tree) / "grids").mkdir()
    (_jobs(tree) / "grids" / "grid_agents.txt").write_text("--config base\n", encoding="utf-8")
    python = next((root / "envs").glob("*/bin/python"))
    _stand_in(python.parent, "python", CHECK_PYTHON)
    settings = {**env, "FAKE_LOG": str(log), "FORK_UNDER_TEST": str(tree), "REAL_PYTHON": sys.executable}
    command = ["bash", str(_in(tree, "check_tables.sh")), "grid_agents.txt=development"]
    script = 'source "$HPC_DIR/env.sh"; fork_python; cd "$FORK"; ' + " ".join(command)
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=tmp_path, env=settings)
    assert done.returncode == 0 and done.stdout.splitlines()[-1] == "C7_TABLES_KEPT", done.stderr
    runs = (log / "runs").read_text(encoding="utf-8").splitlines()
    assert len(runs) == 2 and all("--agent scripted" in run and "--starts development" in run and f"{root}/runs/smoke/" in run for run in runs)
    assert "--config timed --clock-mode instant" in runs[1]
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=tmp_path, env={**settings, "FAKE_CHANGE": tables[1]})
    assert done.returncode == 1 and f"C7_TABLES_CHANGED {tables[1]}" in done.stdout and "C7_TABLES_KEPT" not in done.stdout


def test_a_registered_job_of_the_loops_of_gepa_names_its_runs_in_flight_and_plays_at_most_that_many(tmp_path):
    """The driver registers a loop on its wall-clock preset only at the registered runs in flight, which a loop job reads from
    GRID_PARALLEL: a registered job without it stops with status 2 before any loop, and the job keeps at most GRID_PARALLEL
    loops alive, every loop at once when it is unset outside the kind of record."""
    script = (CLUSTER_JOBS / "3_run" / "run_gepa_online.sh").read_text(encoding="utf-8")
    lines = script.splitlines()
    first = lines.index('if [ "$RUNS_KIND" = record ] && [ -z "${GRID_PARALLEL:-}" ]; then')
    refusal = "\n".join(lines[first:lines.index("fi", first) + 1])
    for kind, parallel, status in (("record", "", 2), ("record", "4", 0), ("development", "", 0)):
        done = _bash(tmp_path, f"RUNS_KIND={kind} GRID_PARALLEL={parallel}", refusal)
        assert done.returncode == status and (("GRID_PARALLEL=<n>" in done.stderr) == (status == 2)), (kind, parallel, done.stderr)
    first = lines.index("PIDS=")
    loops = "\n".join(lines[first:lines.index("done", first) + 1])
    driver = ('python() { local id; id=$(echo "$*" | cksum | cut -d " " -f 1); mkdir -p running; : > "running/$id"; '
              'ls running | wc -l >> "$PWD/count_in_flight"; /bin/sleep 0.3; rm -f "running/$id"; }')
    for parallel, most in (("2", 2), ("", 6)):
        (tmp_path / "count_in_flight").unlink(missing_ok=True)
        done = _bash(tmp_path, driver, f'GEPA_AGENTS="a b c" FOLDS="1 2" CARDS=verifier OUT=. GRID_PARALLEL={parallel}', loops,
                     'for PID in $PIDS; do wait "$PID"; done')
        counts = [int(n) for n in (tmp_path / "count_in_flight").read_text(encoding="utf-8").split()]
        assert done.returncode == 0 and len(counts) == 6 and max(counts) <= most and (parallel or max(counts) > 2), (parallel, counts, done.stderr)


def test_a_loop_job_of_gepa_on_a_wall_clock_preset_plays_one_loop_at_a_time_and_hands_it_one_run_in_flight(tmp_path, monkeypatch):
    """On a preset whose clock charges the wall seconds of thinking, the preset of training by default, a loop job at
    GRID_PARALLEL=4 keeps one loop alive and hands it GRID_PARALLEL=1, which the start check of such a preset compares with
    design.RUNS_IN_FLIGHT, so the loop is of record, and its settings line names the 4 the job was given beside the one loop in
    flight. The clock is the one that grid_job_lists.py of the tree reads. The control: on the preset with the clock of generated
    tokens the loops of the job fly together, each is handed the 4 of the job, read as a process started by a loop reads its
    environment, and the settings line names the 4 alone."""
    lines = (CLUSTER_JOBS / "3_run" / "run_gepa_online.sh").read_text(encoding="utf-8").splitlines()
    first = lines.index('echo "--config ${GEPA_CONFIG:-deterioration}" > "$LA/preset.txt"')
    block = "\n".join(lines[first:lines.index("done", lines.index("PIDS=")) + 1])
    module = f'PYTHONPATH="{os.pathsep.join(p for p in sys.path if p)}" "{sys.executable}" -m {grid_lines.__name__}'
    driver = ('python() { case "$*" in '
              f'*" clocked "*) {module} clocked "${{@: -1}}" ;; '
              '*) local id; id=$(echo "$*" | cksum | cut -d " " -f 1); mkdir -p running; : > "running/$id"; '
              'ls running | wc -l >> "$PWD/count_in_flight"; bash -c \'echo "${GRID_PARALLEL:-unset}"\' >> "$PWD/handed"; '
              '/bin/sleep 0.3; rm -f "running/$id" ;; esac; }')
    for preset, wall in (("", True), ("deterioration_token_clock", False)):
        for name in ("count_in_flight", "handed"):
            (tmp_path / name).unlink(missing_ok=True)
        done = _bash(tmp_path, driver, f"GEPA_CONFIG={preset}", 'GEPA_AGENTS="a b c" FOLDS=1 CARDS=verifier OUT=. LA=.',
                     "export GRID_PARALLEL=4", block, 'for PID in $PIDS; do wait "$PID"; done')
        counts = [int(n) for n in (tmp_path / "count_in_flight").read_text(encoding="utf-8").split()]
        handed = (tmp_path / "handed").read_text(encoding="utf-8").split()
        assert done.returncode == 0 and len(counts) == len(handed) == 3, (preset, counts, handed, done.stderr)
        assert (max(counts) == 1) == wall and set(handed) == {"1" if wall else "4"}, (preset, counts, handed)
        settings = "loops in flight 1 on the wall clock, where the job named 4," if wall else "loops in flight 4,"
        assert settings in done.stdout and ("on the wall clock" in done.stdout) == wall, done.stdout
        asked = (tmp_path / "preset.txt").read_text(encoding="utf-8").split()
        monkeypatch.setenv("GRID_PARALLEL", handed[0])
        _name, config, _prices = run_episodes.configure(run_episodes.build_parser().parse_args([*asked, "--agent", "causal"]))
        assert (config.clock_mode == run_lines.WALL_CLOCK) == wall and (run_episodes.runs_in_flight() == design.RUNS_IN_FLIGHT) == wall, asked


def test_a_requeued_job_of_the_loops_of_gepa_resumes_them_and_a_failed_loop_sets_the_exit_status():
    """run_gepa_online.sh leaves the folder of a requeued job where it is and starts every loop with --resume, which takes up the
    unfinished loop of its settings there and plays a finished one never again, and a loop that ends with an error, a reflection
    endpoint that stopped answering among them, makes the exit status of the job other than 0. GEPA_SPLIT passes the split of a
    trial and leaves the registered split when unset."""
    script = (CLUSTER_JOBS / "3_run" / "run_gepa_online.sh").read_text(encoding="utf-8")
    assert '${GEPA_SPLIT:+--training-split $GEPA_SPLIT} --out "$OUT" --resume' in script and "_attempt" not in script
    assert 'PIDS="$PIDS $!"' in script and 'for PID in $PIDS; do\n  wait "$PID" || STATUS=5\ndone' in script


# the curl of the endpoint check: FAKE_KEYED answers the request with the key, FAKE_OPEN the request without it
CURL_CODES = """code=${FAKE_OPEN:-401}
case " $* " in *" -K "*) code=${FAKE_KEYED:-200} ;; esac
case " $* " in *" -w "*) printf "%s" "$code" ;; esac"""


def test_the_endpoint_check_fails_when_it_never_ran_when_the_key_is_refused_and_when_nothing_answers(tmp_path):
    """The check passes only when it ran and both answers hold: an endpoint file that never appeared, a model list refused
    with the key and a request without the key that no server answered (000) each end the job with status 1, after it wrote the
    stop file of the serving job."""
    tree, root, fake, env = _tree(tmp_path, ("check_endpoint.sbatch",))
    _stand_in(fake, "curl", CURL_CODES)
    _stand_in(fake, "squeue", "exit 0")  # the serving job is neither queued nor running, so a missing file is not waited for
    _stand_in(fake, "hostname", 'case "${1:-}" in -s) echo node7 ;; *) echo node7.cluster ;; esac')
    endpoints = root / "endpoints"
    endpoints.mkdir(parents=True)
    (endpoints / "900.key").write_text("k" * 64 + "\n", encoding="utf-8")

    def check(written=True, **values):
        (endpoints / "900.txt").unlink(missing_ok=True)
        if written:
            (endpoints / "900.txt").write_text("http://10.1.2.3:30123/v1\njob 900 node node3 model m served_name m\n", encoding="utf-8")
        done = subprocess.run(["bash", str(_in(tree, "check_endpoint.sbatch"))], capture_output=True, text=True, cwd=tmp_path,
                              env={**env, "SERVE_JOB": "900", **values}, timeout=60)
        stopped = (endpoints / "900.stop").exists()
        (endpoints / "900.stop").unlink(missing_ok=True)
        return done, stopped

    done, stopped = check()
    assert done.returncode == 0 and "HTTP_BETWEEN_NODES ok" in done.stdout and "ENDPOINT_CLOSED ok" in done.stdout and stopped, done.stderr
    for written, values, line in ((False, {}, "HTTP_BETWEEN_NODES not run"), (True, {"FAKE_KEYED": "401"}, "HTTP_BETWEEN_NODES failed with code 401"),
                                  (True, {"FAKE_OPEN": "000"}, "ENDPOINT_CLOSED unknown: HTTP 000")):
        done, stopped = check(written, **values)
        assert done.returncode == 1 and line in done.stdout and "CD_ENDPOINT_DONE" in done.stdout and stopped, (values, done.stdout)


def test_a_download_that_fails_writes_no_commit_file(tmp_path):
    """A transfer cut halfway leaves files whose metadata name the pin, so only the status of the download tells it: a failed
    download ends the job with status 3 before HUB_COMMIT.txt, which a serving job takes as the mark of a complete folder."""
    tree, _root, fake, env = _tree(tmp_path, ("download_model.sbatch",))
    log = tmp_path / "log"
    log.mkdir()
    _stand_in(fake, "uvx", DOWNLOAD)
    _stand_in(fake, "curl", HUB_LISTING)
    model = "ibm-granite/granite-3.3-8b-instruct"
    target = tmp_path / "models" / model.split("/")[1]
    settings = {**env, "FAKE_LOG": str(log), "MODELS": str(tmp_path / "models"), "SHARED": str(tmp_path / "no_shared_folder"), "MODEL": model,
                "FAKE_COMMIT": PINS[model]}
    done = subprocess.run(["bash", str(_in(tree, "download_model.sbatch"))], capture_output=True, text=True, cwd=tmp_path,
                          env={**settings, "FAKE_FAIL": "1"}, timeout=60)
    assert done.returncode == 3 and "failed with status 1" in done.stderr and "CD_DOWNLOAD_DONE" not in done.stdout, done.stderr
    assert (target / "config.json").exists() and not (target / "HUB_COMMIT.txt").exists()
    done = subprocess.run(["bash", str(_in(tree, "download_model.sbatch"))], capture_output=True, text=True, cwd=tmp_path, env=settings, timeout=60)
    assert done.returncode == 0 and (target / "HUB_COMMIT.txt").read_text(encoding="utf-8") == f"{PINS[model]}\n", done.stderr


# the apptainer of the image job: a pull writes the image, and the image imports sglang at FAKE_SGLANG unless FAKE_IMPORT_FAIL
APPTAINER_PULL = """case "$1" in
  pull) echo image > "$2" ;;
  exec) [ -z "${FAKE_IMPORT_FAIL:-}" ] || { echo "ModuleNotFoundError: No module named sglang" >&2; exit 1; }; echo "${FAKE_SGLANG:-0.5.19}" ;;
esac"""


def test_the_image_job_ends_only_when_the_image_imports_sglang_at_the_version_of_its_tag(tmp_path):
    """pull_image.sbatch reads the version that the tag of IMAGE names, v0.5.19 for the default image, and ends with
    CD_IMAGE_DONE only when the image imports sglang at that version: an image that cannot import it, or holds another
    version, ends the job with status 3."""
    tree, _root, fake, env = _tree(tmp_path, ("pull_image.sbatch",))
    _stand_in(fake, "apptainer", APPTAINER_PULL)
    _stand_in(fake, "module", "exit 0")
    _stand_in(fake, "df", 'case " $* " in *" --output=avail "*) echo Avail; echo 500G ;; *) echo "local free" ;; esac')
    settings = {**env, "SHARED": str(tmp_path / "shared"), "SIF": str(tmp_path / "own" / "sglang.sif")}

    def pull(**values):
        return subprocess.run(["bash", str(_in(tree, "pull_image.sbatch"))], capture_output=True, text=True, cwd=tmp_path,
                              env={**settings, **values}, timeout=60)

    done = pull()
    assert done.returncode == 0 and "sglang 0.5.19" in done.stdout and "CD_IMAGE_DONE" in done.stdout, done.stderr
    for values, words in (({"FAKE_IMPORT_FAIL": "1"}, "cannot import sglang"), ({"FAKE_SGLANG": "0.5.20"}, "holds SGLang 0.5.20, not the version 0.5.19")):
        done = pull(**values)
        assert done.returncode == 3 and words in done.stderr and "CD_IMAGE_DONE" not in done.stdout, (values, done.stderr)
    done = pull(IMAGE="lmsysorg/sglang:latest")
    assert done.returncode == 3 and "not the version latest" in done.stderr


def test_the_loop_job_of_gepa_runs_under_the_shell_options_of_every_job_script():
    """serve_model.sbatch runs its RUN_AFTER_SERVE script with bash, so the loop job sets its own options, and an unset or
    misspelled variable stops it as it stops every other job script."""
    script = (CLUSTER_JOBS / "3_run" / "run_gepa_online.sh").read_text(encoding="utf-8")
    assert f"\nset -uo pipefail\n{SETTINGS_LINE}\n" in script


# the commands of the reproduction job as shell functions: the driver prints its run folder, each step ends with the status that
# its variable names, and acceptance_check.py writes its report unless FAKE_NO_REPORT
REPRO_STEPS = """python() {
  [ "$1" = -m ] && shift  # the moved script starts each step as python -m <module>
  case "$1" in
    *.run_scenarios) echo "AOB_REPRO_RUN $PWD/run"; return "${FAKE_DRIVER:-0}" ;;
    *.score_runs) return "${FAKE_SCORE:-0}" ;;
    *.acceptance_check)
      local name=acceptance.md
      [ "${3:-}" != --port ] || name=acceptance_$4.md
      [ -n "${FAKE_NO_REPORT:-}" ] || { mkdir -p "$OUT/results"; echo verdict > "$OUT/results/$name"; }
      return "${FAKE_ACCEPTANCE:-3}" ;;
  esac
}
OUT=out LAYER=layer SERVED_NAME=m BASE_URL=http://node:1/v1 AOB_AGENTS="agent_as_tool_paper plan_execute_paper" AGENTS="agent_as_tool_paper plan_execute_paper"
SET_ARGS=(--set pilot) JUDGE_ARGS=()
mkdir -p "$OUT"
"""


def test_the_reproduction_job_ends_with_the_first_step_that_failed_and_reads_acceptance_from_its_report(tmp_path):
    """run_reproduction.sh scores what the driver played even when the driver failed, and ends with the first nonzero status of
    the driver and the scoring. The exit code of acceptance_check.py is its verdict, a pass, a miss or runs that cannot
    decide, and stays in the log, while a check that wrote no report fails the job with status 3."""
    lines = (PACKAGE / "frameworks" / "AssetOpsBench" / "cluster_jobs" / "run_reproduction.sh").read_text(encoding="utf-8").splitlines()
    block = "\n".join(lines[next(i for i, line in enumerate(lines) if line.startswith("# 5. The episodes")):])
    cases = (({}, 0), ({"FAKE_DRIVER": "1"}, 1), ({"FAKE_SCORE": "2"}, 2),
             ({"FAKE_ACCEPTANCE": "1"}, 0), ({"FAKE_NO_REPORT": "1"}, 3))
    for n, (values, status) in enumerate(cases):
        folder = tmp_path / str(n)
        folder.mkdir()
        exports = " ".join(f"{key}={value}" for key, value in values.items())
        done = _bash(folder, "set -uo pipefail", REPRO_STEPS, f"export {exports}" if exports else ":", block)
        assert done.returncode == status and "CD_AOB_REPRO_DONE" in done.stdout, (values, done.stdout, done.stderr)
        assert "SCORE_EXIT" in done.stdout and "ACCEPTANCE_EXIT" in done.stdout and "PE_ACCEPTANCE_EXIT" in done.stdout, values


# the uv of the environment job: a sync makes bin/python first, as uv does, and then ends with FAKE_SYNC
UV_SYNC = f"""echo "$*" >> "$FAKE_LOG/uv_calls"
case "$1" in
  --version) echo "uv 0.11.7" ;;
  sync) mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"; printf '#!/bin/bash\\nexec "{sys.executable}" "$@"\\n' > "$UV_PROJECT_ENVIRONMENT/bin/python"
        chmod +x "$UV_PROJECT_ENVIRONMENT/bin/python"; exit "${{FAKE_SYNC:-0}}" ;;
  pip) echo "package==1" ;;
esac"""


def test_a_sync_that_stopped_halfway_is_built_again_and_no_job_takes_its_environment(tmp_path):
    """uv sync makes bin/python before it installs, so the environment job marks a sync that finished with the SHA-256 of the
    lock and builds again over any environment without that mark, and fork_python refuses such an environment."""
    tree, root, _fake, env = _tree(tmp_path, ("setup_env.sbatch",))
    shutil.rmtree(root / "envs")
    log = tmp_path / "log"
    log.mkdir()
    (root / "tools" / "uv-0.11.7").mkdir(parents=True)
    _stand_in(root / "tools" / "uv-0.11.7", "uv", UV_SYNC)
    settings = {**env, "FAKE_LOG": str(log)}

    def setup(**values):
        return subprocess.run(["bash", str(_in(tree, "setup_env.sbatch"))], capture_output=True, text=True, cwd=tmp_path,
                              env={**settings, **values}, timeout=60)

    def syncs():
        return [line for line in (log / "uv_calls").read_text(encoding="utf-8").splitlines() if line.startswith("sync ")]

    done = setup(FAKE_SYNC="1")
    assert done.returncode == 3 and "CD_SETUP_DONE" not in done.stdout and len(syncs()) == 1, done.stderr
    marks = list((root / "envs").glob("*/.cd_synced"))
    assert list((root / "envs").glob("*/bin/python")) and marks == []
    done = _run(tmp_path, env, *_sourced(tree, "fork_python", "echo IMPORTED"))
    assert done.returncode == 2 and "IMPORTED" not in done.stdout and "no mark of a finished sync" in done.stderr
    done = setup()
    assert done.returncode == 0 and "CD_SETUP_DONE" in done.stdout and len(syncs()) == 2, done.stderr
    assert [mark.read_text(encoding="utf-8") for mark in (root / "envs").glob("*/.cd_synced")] == [hashlib.sha256(LOCK.encode()).hexdigest() + "\n"]
    done = setup()
    assert done.returncode == 0 and "not built again" in done.stdout and len(syncs()) == 2, done.stderr


def test_a_download_hashes_every_file_and_checks_the_large_files_against_the_listing_of_the_hub(tmp_path):
    """download_model.sbatch writes FILES.sha256, one hash per file of the model folder, and stops with status 3 when a large file of
    the Hub's listing at the pin is missing or holds another SHA-256; the endpoint file of a serving job then names the hash of that
    list as weights_files_sha256."""
    tree, _root, fake, env = _tree(tmp_path, ("download_model.sbatch",))
    log = tmp_path / "log"
    log.mkdir()
    _stand_in(fake, "uvx", DOWNLOAD)
    _stand_in(fake, "curl", HUB_LISTING)
    model = "ibm-granite/granite-3.3-8b-instruct"
    target = tmp_path / "models" / model.split("/")[1]
    settings = {**env, "FAKE_LOG": str(log), "MODELS": str(tmp_path / "models"), "SHARED": str(tmp_path / "no_shared"), "MODEL": model,
                "FAKE_COMMIT": PINS[model]}
    job = ["bash", str(_in(tree, "download_model.sbatch"))]
    listing = tmp_path / "listing.json"
    config_sha = hashlib.sha256(b"{}\n").hexdigest()
    for oid, status in (("0" * 64, 3), (config_sha, 0)):
        shutil.rmtree(target, ignore_errors=True)
        listing.write_text(json.dumps([{"type": "file", "path": "config.json", "lfs": {"oid": oid}}]), encoding="utf-8")
        done = subprocess.run(job, capture_output=True, text=True, cwd=tmp_path, env={**settings, "FAKE_LISTING": str(listing)}, timeout=60)
        assert done.returncode == status and (target / "HUB_COMMIT.txt").exists() == (status == 0), (oid, done.stdout, done.stderr)
    assert (target / "FILES.sha256").read_text(encoding="utf-8") == f"{config_sha}  config.json\n"
    done = _run(tmp_path, env, *_sourced(tree, f'weights_record "{target}"'))
    assert done.stdout.splitlines() == [f"weights_commit {PINS[model]}", f"weights_files_sha256 {hashlib.sha256((target / 'FILES.sha256').read_bytes()).hexdigest()}"]


def test_the_judge_smoke_fails_when_a_reply_of_the_judge_holds_thinking(tmp_path):
    """The chat template of the judge decides whether the request turns thinking off, so the smoke of every judge job prints
    JUDGE_THINKING and fails on any count but 0."""
    tree, grid, env = _judge_tree(tmp_path)
    for count, status in (("0", 0), ("1", 1)):
        done = _judge(tree, {**env, "JUDGE_SMOKE": str(grid), "FAKE_THINKING": count})
        assert f"JUDGE_THINKING {count}" in done.stdout and (done.returncode == 0) == (status == 0), (count, done.stdout, done.stderr)


def test_a_judge_job_counts_the_thinking_of_its_judge_before_it_judges_the_folders_and_stops_on_a_count_above_0(tmp_path):
    """Every judge job of grid folders counts the replies of its judge that hold thinking under the request of the judge before
    its first pass (episode_judge.py --thinking): on a count above 0 it stops with status 2, runs no pass of judge_finished_runs.py and writes no
    record. The control: on a count of 0 the same job judges the folders."""
    tree, grid, env = _judge_tree(tmp_path / "thinking", "--fixed")
    done = _judge(tree, {**env, "JUDGE_FOLDERS": str(grid), "FAKE_THINKING": "1"})
    assert done.returncode == 2 and "JUDGE_THINKING 1" in done.stdout and "the job judges no episode" in done.stderr, (done.returncode, done.stdout, done.stderr)
    assert "JUDGE_PLAN" not in done.stdout and "JUDGE_PASS" not in done.stdout and not list(grid.glob("*/*/hidden/judge.json"))
    tree, grid, env = _judge_tree(tmp_path / "plain", "--fixed")
    done = _judge(tree, {**env, "JUDGE_FOLDERS": str(grid), "FAKE_THINKING": "0"})
    assert done.returncode == 0 and "JUDGE_THINKING 0" in done.stdout and "JUDGE_PASS 1 judged 2" in done.stdout, done.stderr
    assert list(grid.glob("*/*/hidden/judge.json"))


def test_the_agent_model_is_served_with_every_flag_of_the_serving_file():
    """The agent model's launch carries each flag that the agent block of serving_of_record.json declares beyond the context
    length, the dtype and the chat template, which the script sets from CONTEXT_LENGTH, AGENT_DTYPE and the template it
    writes; today that is --enable-cache-report (with --mem-fraction-static and --cuda-graph-max-bs of the co-served mode from
    env.example.sh). The judge served alone takes its own flags and no such flag."""
    serve = (CLUSTER_JOBS / "2_serve_model" / "serve_model.sbatch").read_text(encoding="utf-8")
    block = json.loads(SERVING_FILE.read_text(encoding="utf-8"))["agent"]["launch_args"]
    set_by_script = {"--context-length", "--dtype", "--chat-template"}
    for mode, flags in block.items():
        named = [flag for flag in flags if flag.startswith("--") and flag not in set_by_script]
        missing = [flag for flag in named if flag not in serve and flag not in ENV_SCRIPT]
        assert missing == [], (mode, missing)
    # the agent's branch ends at the TEMPLATE_KIND line after it, not at the default that the script sets among its first lines
    start = serve.index("then AGENT_FLAGS=$JUDGE_ALONE_SGLANG_EXTRA\n")
    agent_branch = serve[start:serve.index("TEMPLATE_KIND=model", start)]
    assert agent_branch.index("--enable-cache-report") > agent_branch.index("else\n")
