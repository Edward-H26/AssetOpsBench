"""The stage profile of the episodes of run folders: one row per episode in a flat table, a summary per cell and per cluster job,
and a readable table on the standard output (CausalDiagnosis/performance/README.md).

    python -m CausalDiagnosis performance profile-table RUN_FOLDER [RUN_FOLDER ...] --out FOLDER [--engine-export PATH ...]

It writes FOLDER/episodes.csv, FOLDER/engine_calls.csv and FOLDER/cells.json and nothing into a run folder. A run folder is a
folder with run.json or a run that did not finish, which holds its ledger or scores.csv without run.json. A run folder inside a grid folder
grid_<job id> of a cluster job also brings the view of its job, which reads every run of that grid folder, and the GPU figures of
its episodes from the job's sampler. The export of the engine of one job, its folder or its files, splits the seconds of the model
calls of the job's model runs into the queue, the prefill and the decode of the engine, the rest of the calls outside those three
stages and the calls that the join cannot read. The outputs hold measured times of runs, which stay outside the fork as the run
folders do, so an output folder inside the fork is refused."""
import argparse
import json
import sys
from pathlib import Path

from CausalDiagnosis.experiments.episode_runners.run_episodes.agent_kinds import MODEL_AGENTS
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import run_dirs
from CausalDiagnosis.utils.csv_rows import write_rows
from CausalDiagnosis.utils.json_files import write_json
from CausalDiagnosis.utils.paths import ROOT
from . import cells, engine_export, gpu_sampler, jobs, run_files, stage_profile

EPISODES_FILE, CALLS_FILE, CELLS_FILE = "episodes.csv", "engine_calls.csv", "cells.json"
IDENTITY = ("cell", "run", "grid", "episode", "seed", "scenario_id", "asset", "agent", "config", "clock_mode", "model", "runs_in_flight",
            "card", "card_mib", "unfinished_run", "pass", "left_out")
GPU = ("sampler_rows", "gpu_busy_share", "gpu_energy_j", "gpu_period_s", "gpu_note")
COLUMNS = (*IDENTITY, *stage_profile.COLUMNS, *GPU, "held_card_s", *engine_export.COLUMNS)
NO_MODEL = "the agent reaches no served model"
NO_AGENT = "the run records no agent"  # a run whose folder holds no ledger, whose agent may or may not reach the served model
NO_EXPORT = "no engine export is given"
LABELS = {"total_s": "total T", "creation_s": "creation C", "cache_flush_s": "cache flush Fl, in C", "agent_part_s": "agent part P",
          "end_s": "end E", "model_s": "model M", "failed_model_s": "failed model Mf, in M", "tool_service_s": "tool service S",
          "tool_service_cpu_s": "processor part of S", "no_call_service_s": "no_call steps, in rest", "server_starts_s": "server starts R",
          "env_tools_client_s": "environment tools Tc", "diagnostics_client_s": "diagnostics server Tm",
          "transport_hold_s": "transport and hold Tc - S", "rest_s": "rest"}
ENGINE_LABELS = ("queue", "prefill", "decode", "outside the three stages", "unjoined calls", "differing calls")
PART_LABELS = {"server_start_s": "server start", "opening_s": "opening", "opening_check_s": "opening checks",
               "opening_call_s": "opening call", "to_first_episode_s": "to the first episode", "episodes_span_s": "episodes",
               "after_last_episode_s": "after the last episode"}


def no_model_note(unrecorded):
    """Why a run gets no figure of the served model: its agent reaches none, or its folder records no agent."""
    return NO_AGENT if "agent" in unrecorded else NO_MODEL


def run_sampler(run_dir, meta, unfinished, model_agent):
    """(rows, period, note) of the sampler for the episodes of a run: none for an agent without a served model or a run that
    records no agent, and none for a run that did not finish, since the prefix check of the sampler needs the record that run.json
    keeps."""
    if not model_agent:
        return [], None, no_model_note(meta.get("unrecorded") or [])
    if unfinished:
        return [], None, "the run has no run.json, whose record of the sampler the prefix check needs"
    return gpu_sampler.run_sampler(run_dir, meta)


def episode_rows(run_dir, meta, cell, unfinished):
    """(rows, call seconds, calls) of the episodes of one run folder: the identity of each episode beside its profile, and for the
    kept episodes of a model agent the GPU figures from the job's sampler and the card-seconds it held the card alone, its total
    at one run in flight; and of a model agent the calls of llm_calls.json of every episode folder that holds it, for the join to
    the engine's export."""
    serving = (meta.get("serving") or {}).get("agent") or {}
    grid, _job = run_files.grid_of(run_dir)
    model_agent = meta.get("agent") in MODEL_AGENTS
    samples, step, note = run_sampler(run_dir, meta, unfinished, model_agent)
    rows, call_seconds, calls = [], [], {}
    for name, row, folder in run_files.episode_entries(run_dir):
        files = run_files.episode_files(folder) if folder is not None else None
        profile, seconds = stage_profile.episode_profile(row, folder, files)
        call_seconds += seconds
        if model_agent and files is not None and files["llm_calls"] is not None:
            calls[name] = files["llm_calls"]
        out = {**dict.fromkeys(COLUMNS), "cell": cell, "run": Path(run_dir).name, "grid": grid.name if grid else "", "episode": name,
               "seed": (row or {}).get("seed"), "scenario_id": (row or {}).get("scenario_id"), "asset": meta.get("asset_id"),
               "agent": meta.get("agent_label"),
               "config": meta.get("config_name"), "clock_mode": meta.get("clock_mode"), "model": meta.get("model"),
               "runs_in_flight": meta.get("runs_in_flight"), "card": serving.get("gpu_name"), "card_mib": serving.get("gpu_mib"),
               "unfinished_run": int(unfinished), "pass": (row or {}).get("pass"), **profile, "gpu_note": note}
        if not out["left_out"] and model_agent:
            end = out["started_at"] + out["total_s"]
            out.update(gpu_sampler.episode_figures(samples, out["started_at"], end, step) if samples else {})
            out["held_card_s"] = out["total_s"] if meta.get("runs_in_flight") == 1 else None
        rows.append(out)
    return rows, call_seconds, calls


def grid_runs(grid):
    """Every run folder of a grid folder, with run.json or of a run that did not finish, in name order."""
    return sorted({*run_dirs(grid), *run_files.unfinished_runs(grid)})


def read_run(path, keys):
    """One run folder as the job view and the cells read it: its run.json, or for a run that did not finish what its folder records
    with the fields whose record the folder lacks, whether its agent reaches the served model, its cell index, numbered in
    keys by its first run, its process in in_flight.tsv, its rows, its call seconds and its calls."""
    unfinished = run_files.is_unfinished(path)
    meta = run_files.ledger_meta(path) if unfinished else run_files.run_meta(path)
    key = cells.cell_key(meta)
    index = keys.setdefault(json.dumps(key, sort_keys=True), len(keys) + 1)
    rows, call_seconds, calls = episode_rows(path, meta, index, unfinished)
    return {"dir": path, "meta": meta, "unfinished": unfinished, "missing": meta.get("unrecorded") or [], "model_agent": meta.get("agent") in MODEL_AGENTS,
            "cell": index, "key": key, "process": run_files.process_of(path), "rows": rows, "call_seconds": call_seconds, "calls": calls}


def read_runs(run_folders):
    """(given, runs): the run folders given as absolute paths, and per run folder read, the given ones and every other run of their
    grid folders, what read_run gives. Cells are numbered by their first run, the given runs first."""
    given = [run_files.absolute(path) for path in run_folders]
    grids = {grid for grid, _job in map(run_files.grid_of, given) if grid is not None}
    order = list(dict.fromkeys([*given, *(run_files.absolute(path) for grid in sorted(grids) for path in grid_runs(grid))]))
    keys = {}
    return given, {path: read_run(path, keys) for path in order}


def engine_join(runs, exports):
    """(entries, calls, summary) of the export of the one job of the runs read, as engine_export.job_join gives them over the
    calls of every model run read; none, with the reason in summary, without an export, when a run read records no agent, whose
    calls may have left lines, or when a model run plays at more runs in flight than one, where the time does not tie a line to
    its call."""
    model = [run for run in runs.values() if run["model_agent"]]
    flights = sorted({str(run["meta"].get("runs_in_flight")) for run in model})
    unknown = [run["dir"].name for run in runs.values() if "agent" in run["missing"]]
    if not exports:
        return {}, [], {"note": NO_EXPORT}
    if unknown:
        return {}, [], {"note": f"the join by time needs the agent of every run read, and no agent is recorded for {', '.join(unknown)}"}
    if not model:
        return {}, [], {"note": "no run read reaches the served model"}
    if flights != ["1"]:
        return {}, [], {"note": f"the join by time needs one run in flight, and the model runs read play {', '.join(flights)}"}
    episode_calls = {(run["dir"].name, name): calls for run in model for name, calls in run["calls"].items()}
    return engine_export.job_join(episode_calls, exports)


def add_engine_columns(runs, entries, note):
    """Puts the engine columns into the rows of the kept episodes of the model runs, each with the reason where the join gave
    nothing for it, and the note of the agent into the rows of the other runs."""
    for run in runs.values():
        for row in run["rows"]:
            key = (run["dir"].name, row["episode"])
            if not run["model_agent"]:
                row["engine_note"] = no_model_note(run["missing"])
            elif row["left_out"]:
                continue
            elif key in entries:
                row.update(engine_export.episode_columns(entries[key]))
            else:
                row["engine_note"] = note or f"the episode holds no {run_files.LLM_CALLS}"


def build(run_folders, exports=()):
    """(rows, cells, jobs, calls, engine) of the given run folders: the rows of their episodes, the summary of every cell that one of
    them plays, the view of every job whose grid folder holds one of them, the figures of every call that the join to the engine's
    export reached, and the summary of that join."""
    given, runs = read_runs(run_folders)
    entries, joined, engine = engine_join(runs, exports)
    add_engine_columns(runs, entries, engine.get("note"))
    views = []
    for grid, job in sorted({run_files.grid_of(path) for path in given if run_files.grid_of(path)[0] is not None}):
        views.append(jobs.job_view(grid, job, [run for path, run in runs.items() if path.parent == grid]))
    summaries = []
    for index in dict.fromkeys(runs[path]["cell"] for path in given):
        mine = [runs[path] for path in given if runs[path]["cell"] == index]
        kept = {(run["dir"].name, row["episode"]) for run in mine for row in run["rows"] if not row["left_out"]}
        summaries.append(cells.cell_summary(index, mine[0]["key"], [run["dir"].name for run in mine],
                                            [row for run in mine for row in run["rows"]],
                                            [s for run in mine for s in run["call_seconds"]],
                                            [row for row in joined if (row["run"], row["episode"]) in kept], views, [jobs.unfinished_note(run) for run in mine if run["unfinished"]]))
    calls = [row for path in given for row in joined if row["run"] == path.name]
    return [row for path in given for row in runs[path]["rows"]], summaries, views, calls, engine


def _number(value, places=3):
    return "-" if value is None else f"{value:.{places}f}"


def _share(value):
    return "-" if value is None else f"{100 * value:.1f}%"


def _count(value):
    return "-" if value is None else str(value)


def render_cell(cell):
    """The readable block of one cell."""
    key, serving = cell["key"], cell["key"]["serving"]
    card = f"card {serving['gpu_name']} {serving['gpu_mib']} MiB, engine {serving['engine']}" if serving["gpu_name"] else "no served model"
    lines = [f"cell {cell['cell']}: asset {key['asset']}, agent {key['agent']}, preset {key['config']}, clock {key['clock_mode']}, model {key['model']}, "
             f"runs in flight {key['runs_in_flight']}, {card}",
             f"  runs {len(cell['runs'])}, episodes {cell['episodes']}, kept {cell['kept']}, left out "
             + (", ".join(f"{count} {reason}" for reason, count in cell["left_out"].items()) or "none"),
             f"  {'quantity':<28}{'sum s':>12}{'share':>9}{'median':>10}{'p10':>10}{'p90':>10}"]
    for name, label in LABELS.items():
        found = cell["seconds"][name]
        lines.append(f"  {label:<28}{_number(found['sum']):>12}{_share(found['share']):>9}{_number(found['median']):>10}"
                     f"{_number(found['p10']):>10}{_number(found['p90']):>10}")
    checks, clock, calls, gpu, cost, view = cell["checks"], cell["clock"], cell["calls"], cell["gpu"], cell["cost"], cell["jobs_view"]
    engine = cell["engine"]
    lines += ["  checks, failing of checked: " + ", ".join(f"{label} {checks[name]['failing']} of {checks[name]['checked']}" for name, label in
                                                     (("lines_in_model_spans", "a line inside a model span"), ("rest_negative", "rest below zero"),
                                                      ("late_server_starts", "server starts outside the opening"))),
              f"  thinking: {_number(clock['thinking_s'])} s and {_count(clock['thinking_cycles'])} cycles charged, "
              f"{_share(clock['thinking_share_of_horizon'])} of the horizon; credited starts {_number(clock['credited_starts_s'])} s, "
              f"calls {_number(clock['credited_calls_s'])} s; uncharged after the last call {_number(clock['uncharged_after_last_s'])} s",
              f"  model calls {_count(calls['model_calls'])} ({_count(calls['failed_model_calls'])} failed), seconds per call median "
              f"{_number(calls['call_seconds']['median'])} p10 {_number(calls['call_seconds']['p10'])} p90 "
              f"{_number(calls['call_seconds']['p90'])}, tokens in {_count(calls['input_tokens'])} out {_count(calls['output_tokens'])}, "
              f"out per call-second {_number(calls['output_tokens_per_call_second'], 1)}",
              f"  engine, {engine['episodes']} episodes: calls joined {_count(engine['engine_calls_joined'])}, unjoined "
              f"{_count(engine['engine_calls_unjoined'])}, token mismatches {_count(engine['engine_input_mismatches'])} in and "
              f"{_count(engine['engine_output_mismatches'])} out, lines of retries {_count(engine['engine_extra_lines'])}",
              "  of the call seconds: " + ", ".join(f"{label} {_share(engine[share])}" for label, share in
                                                     zip(ENGINE_LABELS, engine_export.CALL_SHARES))
              + f"; decode tokens per s {_number(engine['decode_tokens_per_s'], 1)}",
              f"  GPU: {_count(gpu['sampler_rows'])} sampler rows, busy {_share(gpu['gpu_busy_share'])}, energy per episode "
              f"{_number(gpu['gpu_energy_j_per_episode'], 0)} J, per passed episode {_number(gpu['gpu_energy_j_per_passed_episode'], 0)} J",
              f"  pass rate {_share(cost['pass_rate'])}; card-seconds held alone per episode {_number(cost['held_card_s_per_episode'], 1)}, "
              f"per passed episode {_number(cost['held_card_s_per_passed_episode'], 1)}",
              f"  jobs that played the cell alone {len(view['jobs_alone'])}: card-seconds per episode {_number(view['card_s_per_episode'], 1)}, "
              f"per passed episode {_number(view['card_s_per_passed_episode'], 1)}; episodes per card-hour, ending inside "
              f"{_number(view['ending_per_card_hour'], 2)}, shares inside {_number(view['shares_per_card_hour'], 2)}"]
    return lines + [f"  note: {note}" for note in cell["notes"]]


def render_job(view):
    """The readable block of one job."""
    lines = [f"job {view['grid']}: elapsed {_number(view['elapsed_s'], 1)} s, model runs {view['model_runs']}, "
             f"model episodes {view['model_episodes']}, cell {view['cell']}",
             "  " + ", ".join(f"{label} {_number(view[name], 1)} s ({_share(view['share_' + name[:-2]])})" for name, label in PART_LABELS.items())]
    if view["cell"] is not None:
        lines += [f"  card-seconds per episode {_number(view['card_s_per_episode'], 1)}, per passed episode "
                  f"{_number(view['card_s_per_passed_episode'], 1)}",
                  f"  at {view['runs_in_flight']} runs in flight: windows {_number(view['window_s'], 1)} s, episodes ending inside "
                  f"{_count(view['episodes_ending_inside'])}, shares inside {_number(view['episode_shares_inside'], 2)}; per second "
                  f"{_number(view['ending_per_s'], 6)} and {_number(view['shares_per_s'], 6)}, per card-hour "
                  f"{_number(view['ending_per_card_hour'], 2)} and {_number(view['shares_per_card_hour'], 2)}"]
    return lines + [f"  note: {note}" for note in view["notes"]]


def render_engine(engine):
    """The readable line of the join to the engine's export."""
    if "note" in engine:
        return [f"engine export: {engine['note']}"]
    return [f"engine export: {engine['export_files']} files, {engine['lines']} lines read, {engine['unread_lines']} unread, "
            f"{engine['passed_lines']} received while no call was open, {engine['ambiguous_lines']} while two were; "
            f"{engine['joined_calls']} of {engine['calls']} model calls joined, {engine['extra_lines']} lines of retried attempts"]


def render(summaries, views, engine):
    blocks = [*map(render_cell, summaries), *map(render_job, views), render_engine(engine)]
    return "\n".join(line for block in blocks for line in [*block, ""])


def parser():
    ap = argparse.ArgumentParser(description="The stage profile of the episodes of run folders; the module docstring and "
                                             "CausalDiagnosis/performance/README.md hold what it reads and writes.")
    ap.add_argument("runs", nargs="+", help="run folders, each holding its run.json, or its ledger or scores.csv when the run did not finish")
    ap.add_argument("--out", required=True, help="the folder outside the fork that receives episodes.csv, engine_calls.csv and cells.json")
    ap.add_argument("--engine-export", nargs="+", default=[], metavar="PATH",
                    help="the export folder of the engine of the one job of the run folders, or its files")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    out = Path(a.out).resolve()
    if out.is_relative_to(ROOT.resolve()):
        print(f"{out} lies inside the fork, and the profile holds measured times of runs, which stay outside it", file=sys.stderr)
        return 2
    missing = [path for path in a.runs if not (Path(path) / run_files.RUN_FILE).is_file() and not run_files.is_unfinished(path)]
    if missing:
        print(f"no {run_files.RUN_FILE}, ledger or scores.csv in {', '.join(missing)}: the profile reads run folders", file=sys.stderr)
        return 2
    places = {run_files.grid_of(path)[0] for path in a.runs}
    if a.engine_export and len(places) > 1:
        print(f"the run folders lie in {len(places)} places, and an engine export is of the one job that served them", file=sys.stderr)
        return 2
    rows, summaries, views, calls, engine = build(a.runs, a.engine_export)
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / EPISODES_FILE, rows, COLUMNS)
    write_rows(out / CALLS_FILE, calls, engine_export.CALL_COLUMNS)
    write_json(out / CELLS_FILE, {"cells": summaries, "jobs": views, "engine": engine})
    print(render(summaries, views, engine))
    return 0


if __name__ == "__main__":
    sys.exit(main())
