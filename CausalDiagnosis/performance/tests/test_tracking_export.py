"""tracking_export.py and tracking_client.py, the export to Weights and Biases: the allow-list of the module is the one written
here, every name of it is one that its source declares, and the payload holds its names alone; a column or a figure that no list
names stays back, and one name in a list sends it; no string of a hidden, private or text file or of an unlisted field of a
made-up run folder with the fields of a real run, no count of the faulty components, no fact of the scenario or of the oracle, no
path, no user name and no job id reaches the payload or the client; the client is handed the payload and nothing else; each
tracked run goes to the project of its own kind, read from its ledger, or nowhere, and a client that opens it elsewhere stops; the
dry run prints the whole payload with the address, the team or user and every project, and sends nothing, and so do a shell
without a declared address, one without the team or user, one that names one project for both kinds and one without a key; a run
not recorded finished and an open key file are refused; a folder exported again keeps its id and sends no history row twice; a
training folder gives its trainer's configuration and loss history; the client keeps every setting that stops it from sending
the machine, the user, the working folder, the command line, the repository and the installed packages; and the shipped path
starts the client and its login through uv with the payload alone and an environment of the listed variables, the recorder of
the client's errors off and its folders in the temporary folder of the start. Every value here is made up, the stand-ins are the
only clients and no test reaches a network. Each test has a control."""
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from CausalDiagnosis.experiments.episode_runners.episode_launchers import LAUNCH_PRIVATE
from CausalDiagnosis.experiments.episode_runners.run_episodes.start_check import STAMP
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS as SCORES_CSV, HIDDEN, HIDDEN_FILES, SCORE_COLUMNS
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.run_verification.public_record import PRIVATE_FILE
from CausalDiagnosis.performance import profile_table, tracking_client, tracking_export as export
from CausalDiagnosis.performance.tests import made_up
from CausalDiagnosis.performance.tests.stand_in_client import StandInClient, handed, leaves
from CausalDiagnosis.utils.paths import ROOT, SRC

PLANTED = "MADE_UP_PLANTED"  # every string planted in a file or field that must stay back holds it
JOB = "9081726354"  # the made-up job id of the made-up grid folder, which no value sent may hold
USER = "made_up_user"  # the made-up user name of the made-up paths
# a made-up address under the top-level domain that never resolves (RFC 2606)
PLACE = {"address": "https://api.made-up.invalid", "entity": "made-up-team",
         "projects": {export.RECORD: "made-up-record-project", STAMP: "made-up-development-project"}}
# the variables of the place on the service as env.example.sh names them, written here apart from the module that reads them
ADDRESS_VARIABLE, ENTITY_VARIABLE = "WANDB_BASE_URL", "WANDB_ENTITY"
PROJECT_VARIABLES = {export.RECORD: "WANDB_PROJECT", STAMP: "WANDB_PROJECT_DEVELOPMENT"}
# variables of the client that a shell may hold and that must never reach it, beside the place's own
SHELL_CLIENT_VARIABLES = {"WANDB_SWEEP_ID": "made-up-sweep", "WANDB_RUN_ID": "made-up-run-id", "WANDB_NOTES": f"{PLANTED}_shell_notes",
                          "WANDB_TAGS": "made-up-tag", "WANDB_MODE": "offline", "WANDB_ERROR_REPORTING": "true"}
# the variables of the client that the environment of a start holds: the key, the recorder of its errors and its three folders
CLIENT_WANDB = ("WANDB_API_KEY", "WANDB_ERROR_REPORTING", "WANDB_CONFIG_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR")
ENTRY = ("id", "name", "project", "group", "job_type", "tags", "config", "summary", "table", "history")
# the settings that stop the client from sending the machine, the user, the working folder, the command line, the repository and
# the installed packages by itself, each with the value it must hold
REQUIRED = {"x_disable_meta": True, "x_disable_machine_info": True, "x_disable_stats": True, "host": "", "disable_git": True,
            "disable_code": True, "save_code": False, "x_save_requirements": False, "console": "off", "disable_job_creation": True}
# the allow-list as written here, apart from the module, so that a name added to a list of the module fails a test until it is
# written here as well: the configuration of a run by its source, the derived names, the tags, the columns of the table, the
# outcome columns of the means, the summed columns, the parts of the summary and the fields of a training
TABLE_SCORES = ("pass", "diagnosis_all_correct", "faulty_set_right", "recovered", "unnecessary", "false_alarms", "cost",
                "downtime_total", "detection_delay", "repair_delay", "turns_to_correct_diagnosis", "abstained", "abstention_cites_reading",
                "end_reason", "time_used", "steps", "provenance_ok", "write_count_ok", "timing_ok", "ordering_ok", "read_gate",
                "judge_pass", "static_faulty_set_correct", "static_repairs_match_oracle")
RUN_PARTS = ("episodes", "kept", "left_out/*", "checks/*/failing", "checks/*/checked", "seconds/*/sum", "seconds/*/share",
             "seconds/*/median", "gpu/sampler_rows", "gpu/gpu_busy_share", "gpu/gpu_energy_j_per_episode",
             "gpu/gpu_energy_j_per_passed_episode", "cost/pass_rate", "cost/held_card_s_per_episode",
             "cost/held_card_s_per_passed_episode")
PINNED = {
    "RUN_CONFIG": {
        "asset": ("run.json", ("asset_id",)), "agent": ("run.json", ("agent",)), "agent_label": ("run.json", ("agent_label",)),
        "preset": ("run.json", ("config_name",)), "clock_mode": ("run.json", ("clock_mode",)), "model": ("run.json", ("model",)),
        "served_name": ("run.json", ("serving", "agent", "model")), "dtype": ("run.json", ("serving", "agent", "dtype")),
        "engine": ("run.json", ("serving", "agent", "engine")), "engine_version": ("run.json", ("serving", "agent", "engine_version")),
        "launch_args_sha256": ("run.json", ("serving", "agent", "launch_args_sha256")),
        "card": ("run.json", ("serving", "agent", "gpu_name")), "card_mib": ("run.json", ("serving", "agent", "gpu_mib")),
        "runs_in_flight": ("run.json", ("runs_in_flight",)), "commit": ("run.json", ("code", "commit")),
        "upload_manifest_sha256": ("run.json", ("code", "upload_manifest_sha256")), "node_kind": ("ledger.json", ("numeric", "processor"))},
    "RUN_DERIVED": ("kind", "cell"),
    "TAGS": (("asset", "asset", None), ("card", "card", None), ("commit", "commit", 12), ("upload", "upload_manifest_sha256", 12),
             ("preset", "preset", None), ("agent", "agent_label", None)),
    "TABLE_IDENTITY": ("run", "episode", "seed", "agent", "config", "clock_mode", "model", "runs_in_flight", "card", "card_mib",
                       "left_out", "cell"),
    "TABLE_PROFILE": (
        "total_s", "creation_s", "cache_flush_s", "agent_part_s", "end_s", "model_s", "failed_model_s", "tool_service_s",
        "tool_service_cpu_s", "no_call_service_s", "server_starts_s", "env_tools_client_s", "diagnostics_client_s", "transport_hold_s",
        "rest_s", "share_creation", "share_cache_flush", "share_agent_part", "share_end", "share_model", "share_failed_model",
        "share_tool_service", "share_tool_service_cpu", "share_no_call_service", "share_server_starts", "share_env_tools_client",
        "share_diagnostics_client", "share_transport_hold", "share_rest", "model_calls_without_span", "counted_calls",
        "counted_calls_without_end", "no_call_steps", "stamped_steps", "server_start_count", "env_tool_calls", "diagnostics_calls",
        "lines_in_model_spans", "rest_negative", "late_server_starts", "thinking_s", "thinking_cycles", "horizon_cycles",
        "think_seconds_per_cycle", "thinking_share_of_horizon", "credited_starts_s", "credited_calls_s", "uncredited_starts_s",
        "uncredited_calls_s", "failed_model_attempts", "uncharged_after_last_s", "model_calls", "failed_model_calls",
        "call_seconds_sum", "call_seconds_median", "input_tokens", "output_tokens", "output_tokens_per_call_second", "sampler_rows",
        "gpu_busy_share", "gpu_energy_j", "gpu_period_s", "held_card_s"),
    "TABLE_SCORES": TABLE_SCORES,
    "MEAN_SCORES": TABLE_SCORES,
    "SCORE_TOTALS": ("wall_s", "input_tokens", "output_tokens", "cached_tokens"),
    "RUN_PARTS": RUN_PARTS,
    "CELL_PARTS": (*RUN_PARTS, "jobs_view/card_s_per_episode", "jobs_view/card_s_per_passed_episode", "jobs_view/window_s",
                   "jobs_view/ending_per_card_hour", "jobs_view/shares_per_card_hour"),
    "JOB_PARTS": ("window_s", "episodes_ending_inside", "episode_shares_inside", "ending_per_s", "shares_per_s", "ending_per_card_hour",
                  "shares_per_card_hour", "card_s_per_episode", "card_s_per_passed_episode"),
    "TRAINER_CONFIG": ("train_batch_size", "num_train_epochs", "max_steps", "logging_steps", "save_steps", "eval_steps"),
    "TRAINER_SUMMARY": ("global_step", "epoch"),
    "TRAINER_HISTORY": ("loss", "epoch"),
}
# the columns of scores.csv that stay back, each with a value planted in a test: the counts of the faulty components, the
# separability of the start, the cost of the oracle's answer and the two columns that give it back, and the columns from which the
# separability or the counts follow alone or with a column that is sent
HELD_BACK = ("n_faulty", "graded_top1_n", "graded_pass_n", "separable", "cost_oracle", "cost_normalized", "reward",
             "commitment_on_inseparable", "abstention_correct", "success", "failure_class", "graded_top1", "graded_pass")
COUNT = 987654  # a made-up number that no other value of a made-up run holds
LATER = "made_up_later_column"  # a made-up name of a column or a figure that a later writer adds
# string fields of a real run.json and its ledger that no name of the allow-list reads, each planted in the made-up run
REAL_RUN_FIELDS = ("mode", "prices", "tool_listings/iot", "backend/kind", "backend/url", "model_id", "endpoint", "judge",
                   "judge_endpoint", "judge_model", "judge_protocol", "ranking_model", "abstention_rule", "abstention_cut", "briefing",
                   "starts", "training_variant", "training_split", "training_fold", "training_sample", "facts_sha256",
                   "server_env/IOT_DBNAME", "python", "mcp", "openai", "platform/python", "platform/installed_sha256",
                   "platform/lock_sha256", "data/facts", "serving/agent/weights_files_sha256", "serving/agent/image_sha256",
                   "serving/agent/template_sha256", "serving/agent/quantization", "serving/agent/kv_cache_dtype",
                   "serving/agent/driver_version", "serving/agent/cuda_version", "serving/agent/endpoint_file",
                   "serving/agent/dtype_launch", "serving/agent/matches_endpoint", "granted/scheduler", "prices_sha256",
                   "price_table_source", "decoding/route", "transport/retry_status", "transport/sessions", "transport/deadline",
                   "limits/reads", "code/identity", "code/dirty", "code/differs_from_upload", "adapter", "agent_as_tool", "anchored",
                   "causal_options", "clock", "diagnostics_server", "episodes_planned", "grade_lookup", "guidance_level",
                   "load_averages", "lora_adapters", "master_seed", "on_origin", "plan_execute", "replicate", "replicate_rule",
                   "sampling_temperature", "seed_pool", "seeds", "servers", "start_cap", "start_cap_seed", "task_block",
                   "verifier_scope")
REAL_LEDGER_FIELDS = ("command/python", "command/program", "command/grid_file", "command/grid_file_sha256", "command/env/HYD_GRID_FILE",
                      "command/env/HYD_ENDPOINT_FILE", "command/env/GRID_EXTRA", "code/upload_manifest_sha256", "data/facts", "judge",
                      "events", "run_files", "updated", "ended")
TEXT_FILES = ("answer.txt", "question.txt", "plan.txt", "react_log.txt", "report.md")


def planted(place):
    return f"{PLANTED}_{place}"


def put(record, path, value):
    """Sets the value under a path a/b/c of a nested record, making the dicts on the way."""
    *heads, last = path.split("/")
    for key in heads:
        record = record.setdefault(key, {})
    record[last] = value


def set_scores(run, **cells):
    """Sets cells of every row of the scores.csv of a run, a column added when the file lacks it."""
    with (run / "scores.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [{**row, **cells} for row in rows]
    with (run / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(name for row in rows for name in row)))
        writer.writeheader()
        writer.writerows(rows)


def set_ledger(folder, **fields):
    """Sets fields of the ledger of a run folder, a field given as None taken out."""
    record = json.loads((folder / ledger.LEDGER).read_text(encoding="utf-8"))
    record.update(fields)
    (folder / ledger.LEDGER).write_text(json.dumps({key: value for key, value in record.items() if value is not None}), encoding="utf-8")


def with_planted(parts):
    """The parts of a made-up episode with a planted string in its prompts, replies, tool replies and the answer in its transcript,
    and a row of scores.csv with outcome columns, a planted error and start state, and two cells of a JSON object or list."""
    parts = copy.deepcopy(parts)
    for n, call in enumerate(parts["llm_calls"]):
        call.update(prompt=planted(f"prompt_{n}"), reply=planted(f"reply_{n}"), agent="made_up_agent", kind="made_up")
    for n, record in enumerate(parts["trajectory"]):
        record.update(response=planted(f"tool_reply_{n}"), tool_args={"made_up": planted(f"tool_args_{n}")})
    for line in parts["lines"]:
        if "answer" in line:
            line["answer"] = planted("transcript_answer")
    parts["row"].update(error=planted("error"), block=planted("block"), entry_offset="3", cost="3", steps="7", end_reason="finish",
                        input_tokens="600", output_tokens="100", prediction_verdict=json.dumps([{"found": planted("verdict_detail")}]),
                        grade=json.dumps({"source": planted("grade")}))
    return parts


def ledger_record(run, name, started, stamp):
    """The ledger of a made-up run folder: the start record that write_run gives, with the id of the run, its status finished, the
    processor of its node, the start check of a run of record (no reason) or of a development run, and a planted string in the
    endpoint address, the command line, the working folder and every field of a real ledger that the allow-list does not name."""
    record = json.loads((run / ledger.LEDGER).read_text(encoding="utf-8"))
    record.update({"format": 1, "run": name, "label": name, "started": started, "status": export.FINISHED,
                   "numeric": {"processor": "MADE_UP_PROCESSOR"}, export.START_CHECK: [planted("start_check_reason")] if stamp else []})
    record["model"]["endpoint_file"]["base_url"] = planted("ledger_url")
    record["command"].update(argv=[*record["command"]["argv"], planted("argv")], cwd=f"/home/{USER}/fork")
    for field in REAL_LEDGER_FIELDS:
        put(record, field, planted(f"ledger_{field}"))
    (run / ledger.LEDGER).write_text(json.dumps(record), encoding="utf-8")


def made_up_run(root, stamp=STAMP, started="2026-01-05T10:00:00", agent_label="causal"):
    """(run folder, endpoints folder): a made-up grid folder grid_<JOB> of a served job with one run folder of two episodes, whose
    job plays one cell and so gets a rate, of record when stamp is None and else a development run, as its ledger and run.json
    record it, and a planted string in every file and field of the run that must stay back, the hidden files, the private files,
    the text files, the paths of run.json and of the ledger, the fields of a real run.json and ledger that the allow-list does not
    name, the key file and the endpoint file."""
    grid = made_up.write_grid(
        Path(root) / f"grid_{JOB}", JOB, record=[f"{JOB}|COMPLETED|2026-01-05T10:00:00|2026-01-05T11:00:00|01:00:00|00:00:00|node7"],
        log=[made_up.runner_header(10.0), "OPENING_CALL answered seconds 2.500 cycles made_up 0.250", planted("slurm_log")],
        in_flight=[(int(made_up.wall(15.0)), "start", "units", "run", "line_0_replicate_none", 1, 0),
                   (int(made_up.wall(600.0)), "end", "units", "run", "line_0_replicate_none", 1, 0)],
        sampler=made_up.sampler_text([(t, 50, 100.0) for t in range(0, 600, 5)]))
    name = "made_up_run"
    fields = {"asset_id": "made_up_asset", "label": name, "agent_label": agent_label, "facts_path": f"/home/{USER}/{planted('facts')}",
              "loaded_files": [planted("loaded_file")], "harness": {"finish_stamp": planted("harness")},
              "code": {"commit": "c" * 40, "upload_manifest_sha256": None, "files": {"harness": {planted("code_file"): "0"}}},
              **({"stamp": stamp, "stamp_reasons": [planted("stamp_reason")]} if stamp else {})}
    meta = made_up.model_meta(**fields)
    meta["serving"]["agent"]["sampler"] = made_up.recorded_sampler(grid / "job_logs" / f"gpu-{JOB}.csv")
    for field in REAL_RUN_FIELDS:
        put(meta, field, planted(f"run_json_{field}"))
    episodes = [(seed, with_planted(made_up.shifted(made_up.known_episode(), shift))) for seed, shift in ((7, 20.0), (8, 40.0))]
    run = made_up.write_run(grid / name, episodes, meta=meta, process=(0, "none"))  # the run of line 0 of the job's in_flight.tsv
    ledger_record(run, name, started, stamp)
    (run / PRIVATE_FILE).write_text(json.dumps({"starts": [{"job": {"slurm_job_id": JOB}, "host": planted("host")}]}), encoding="utf-8")
    for folder in (run, *(run / f"seed{seed:06d}" for seed, _parts in episodes)):
        (folder / LAUNCH_PRIVATE).write_text(json.dumps({"src": [f"/home/{USER}/{planted('launch_src')}"]}), encoding="utf-8")
    for seed, _parts in episodes:
        episode = run / f"seed{seed:06d}"
        (episode / HIDDEN).mkdir()
        for file in HIDDEN_FILES:
            (episode / HIDDEN / file).write_text(json.dumps({"made_up": planted(f"hidden_{file}")}), encoding="utf-8")
        for file in TEXT_FILES:
            (episode / file).write_text(planted(file), encoding="utf-8")
    endpoints = Path(root) / "endpoints"
    endpoints.mkdir()
    (endpoints / f"{JOB}.key").write_text(planted("serving_key"), encoding="utf-8")
    (endpoints / "agent.txt").write_text(f"http://{planted('endpoint')}:1\nstarted {made_up.STARTED}\n", encoding="utf-8")
    return run, endpoints


@pytest.fixture
def made_up_shell(monkeypatch, tmp_path):
    """The variables of a shell that holds a planted key, the endpoint file of a made-up serving job, a made-up home folder without
    a key file and no place on the service."""
    run, endpoints = made_up_run(tmp_path)
    home = tmp_path / "made_up_home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv(export.KEY_FILE, str(home / ".netrc"))
    monkeypatch.setenv("HYD_ENDPOINT_FILE", str(endpoints / "agent.txt"))
    monkeypatch.setenv("API_KEY", planted("api_key"))
    monkeypatch.setenv("WANDB_API_KEY", planted("wandb_key"))
    for name in (ADDRESS_VARIABLE, ENTITY_VARIABLE, *PROJECT_VARIABLES.values(), *SHELL_CLIENT_VARIABLES, "RUN_KIND"):
        monkeypatch.delenv(name, raising=False)
    return run


def shell_place(monkeypatch, development=True):
    """The variables of PLACE in the shell, the project of the development runs left out when development is False."""
    monkeypatch.setenv(ADDRESS_VARIABLE, PLACE["address"])
    monkeypatch.setenv(ENTITY_VARIABLE, PLACE["entity"])
    for kind, name in PROJECT_VARIABLES.items():
        if development or kind == export.RECORD:
            monkeypatch.setenv(name, PLACE["projects"][kind])
        else:
            monkeypatch.delenv(name, raising=False)


def leaked(values, root):
    """The values that hold a planted string, the made-up user name, the job id or a path of the made-up root."""
    return [value for value in map(str, values) if PLANTED in value or USER in value or JOB in value or str(root) in value]


def summary_allowed(name, kind, training):
    head, _, rest = name.partition("/")
    if head != kind:
        return False
    if training:
        return rest in PINNED["TRAINER_SUMMARY"]
    section, _, path = rest.partition("/")
    parts = {"run": PINNED["RUN_PARTS"], "cell": PINNED["CELL_PARTS"], "job": PINNED["JOB_PARTS"]}
    if section in parts:
        return export.part_of(path, parts[section])
    return path in {"outcome": PINNED["MEAN_SCORES"], "totals": PINNED["SCORE_TOTALS"]}.get(section, ())


def outside_allow_list(payload):
    """Every name of the payload that the allow-list written here does not hold, with the part it stands in."""
    out = [f"payload/{key}" for key in payload if key not in ("address", "entity", "runs")]
    for entry in payload["runs"]:
        kind, training = entry["config"].get("kind"), entry["job_type"] == export.TRAINING
        out += [f"entry/{key}" for key in entry if key not in ENTRY]
        out += [] if kind in (export.RECORD, STAMP) and entry["group"].startswith(f"{kind}:") else [f"kind/{kind}"]
        allowed = {*PINNED["TRAINER_CONFIG"], "kind"} if training else {*PINNED["RUN_CONFIG"], *PINNED["RUN_DERIVED"]}
        out += [f"config/{name}" for name in entry["config"] if name not in allowed]
        out += [f"summary/{name}" for name in entry["summary"] if not summary_allowed(name, kind, training)]
        prefixes = {prefix for prefix, _name, _kept in PINNED["TAGS"]}
        out += [f"tag/{tag}" for tag in entry["tags"] if tag != kind and tag != (export.TRAINING if training else None)
                and (training or tag.partition(":")[0] not in prefixes)]
        if entry["table"] is not None:
            out += [] if entry["table"]["key"] == f"{kind}/{export.EPISODES}" else [f"table/{entry['table']['key']}"]
            columns = {*PINNED["TABLE_IDENTITY"], *PINNED["TABLE_PROFILE"], *PINNED["TABLE_SCORES"]}
            out += [f"column/{name}" for name in entry["table"]["columns"] if name not in columns]
        out += [f"history/{name}" for row in entry["history"] for name in row["values"]
                if name not in {f"{kind}/{field}" for field in PINNED["TRAINER_HISTORY"]}]
    return out


def built(run):
    return export.build_payload([str(run)], PLACE)


def fake_uv(folder):
    """A stand-in of uv that appends each call to calls.txt beside it and starts nothing."""
    path = Path(folder) / "uv"
    path.write_text('#!/bin/sh\necho "$@" >> "$(dirname "$0")/calls.txt"\n', encoding="utf-8")
    path.chmod(0o755)
    return path


# ----- the allow-list, and what the payload holds and never holds -----

@pytest.mark.parametrize("name", sorted(PINNED))
def test_the_allow_list_of_the_module_is_the_one_written_here(name):
    """Each list of the module equals the one written here, so that a name added to the module alone fails; the control is
    test_a_name_added_to_a_list_of_the_module_is_reported_and_its_planted_value_found, where one such name leaves."""
    assert getattr(export, name) == PINNED[name]


def unknown_sources(lists, cell, view):
    """The names of the allow-list that their source does not declare: a column of the table that the profile, scores.csv or the
    export does not write, a summed column that scores.csv lacks, and a part of a summary that no leaf of the summary of a cell
    or of the view of a job matches, so that a name that its writer renames fails here instead of leaving the payload unseen."""
    paths = list(export.flat(cell))

    def declared(part):
        head, _, rest = part.partition("/")
        return any(export.part_of(path, [part]) for path in paths) or (rest == "*" and isinstance(cell.get(head), dict))
    out = [name for name in (*lists["TABLE_IDENTITY"], *lists["TABLE_PROFILE"]) if name not in profile_table.COLUMNS]
    out += [name for name in (*lists["TABLE_SCORES"], *lists["MEAN_SCORES"]) if name not in SCORE_COLUMNS]
    out += [name for name in lists["SCORE_TOTALS"] if name not in SCORES_CSV]
    out += [part for part in (*lists["RUN_PARTS"], *lists["CELL_PARTS"]) if not declared(part)]
    return out + [part for part in lists["JOB_PARTS"] if part not in view]


def test_every_name_of_the_allow_list_is_one_that_its_source_declares(made_up_shell):
    _rows, (cell,), (view,) = export.profiled([made_up_shell])
    assert unknown_sources(PINNED, cell, view) == []
    # control: a made-up name in each list is reported
    planted_lists = {name: (*value, LATER) if isinstance(value, tuple) else value for name, value in PINNED.items()}
    planted_lists.update(RUN_PARTS=(*PINNED["RUN_PARTS"], f"cost/{LATER}"), CELL_PARTS=(*PINNED["CELL_PARTS"], f"gpu/{LATER}"))
    assert sorted(unknown_sources(planted_lists, cell, view)) == sorted([LATER] * 6 + [f"cost/{LATER}", f"gpu/{LATER}"])


def test_the_payload_holds_the_names_of_the_allow_list_alone(made_up_shell):
    payload, _notes = built(made_up_shell)
    assert outside_allow_list(payload) == []
    (entry,) = payload["runs"]
    assert {"development/job/ending_per_card_hour", "development/job/shares_per_card_hour", "development/outcome/pass",
            "development/run/seconds/model_s/share", "development/cell/gpu/gpu_busy_share", "development/totals/wall_s",
            "development/run/checks/rest_negative/failing", "development/cell/cost/pass_rate"} <= set(entry["summary"])
    assert entry["config"]["node_kind"] == "MADE_UP_PROCESSOR" and entry["config"]["asset"] == "made_up_asset"
    assert {"pass", "cost"} <= set(entry["table"]["columns"]) and not {"prediction_verdict", "grade", "error", "block", "grid",
                                                                      "gpu_note", "started_at", "scenario_id"} & set(entry["table"]["columns"])
    # control: a name of each part outside the allow-list is reported
    planted_payload = copy.deepcopy(payload)
    (bad,) = planted_payload["runs"]
    bad["config"]["host"] = "made-up"
    bad["summary"]["development/run/notes"] = 1
    bad["table"]["columns"].append("grid")
    bad["tags"].append("grid:made-up")
    bad["history"].append({"step": 1, "values": {"development/prompt": 1}})
    bad["extra"] = 1
    assert sorted(outside_allow_list(planted_payload)) == sorted(["config/host", "summary/development/run/notes", "column/grid",
                                                                  "tag/grid:made-up", "history/development/prompt", "entry/extra"])


def test_the_cached_prompt_tokens_leave_as_a_total_over_the_episodes_whose_cell_holds_a_count(made_up_shell):
    """cached_tokens, a public count of scores.csv, leaves in the totals of the summary beside the input tokens, summed over the
    episodes of the run. The control: with every cell empty, as on a server that reports no count, no total of it leaves."""
    set_scores(made_up_shell, cached_tokens="40")
    payload = built(made_up_shell)[0]
    (entry,) = payload["runs"]
    assert entry["summary"]["development/totals/cached_tokens"] == 40 * len(entry["table"]["rows"]) and outside_allow_list(payload) == []
    set_scores(made_up_shell, cached_tokens="")
    (entry,) = built(made_up_shell)[0]["runs"]
    assert "development/totals/cached_tokens" not in entry["summary"]


def test_a_name_added_to_a_list_of_the_module_is_reported_and_its_planted_value_found(made_up_shell, monkeypatch):
    """A later column of scores.csv or of the profile and a later figure of the summary of a cell, each holding a planted number,
    stay back while no list of the module names them; one name added to one list sends it, and the allow-list written here reports
    it until it is written here too."""
    set_scores(made_up_shell, **{LATER: str(COUNT)})
    profiled = export.profiled

    def with_later(paths):
        rows, summaries, views = profiled(paths)
        for summary in summaries:
            summary["cost"][LATER] = COUNT
        return [{**row, LATER: COUNT} for row in rows], summaries, views
    monkeypatch.setattr(export, "profiled", with_later)
    payload = built(made_up_shell)[0]
    assert COUNT not in leaves(payload) and LATER not in str(leaves(payload)) and outside_allow_list(payload) == []
    added = {"TABLE_SCORES": (LATER, "column"), "MEAN_SCORES": (LATER, "summary"), "TABLE_PROFILE": (LATER, "column"),
             "RUN_PARTS": (f"cost/{LATER}", "summary")}
    for name, (entry, part) in added.items():
        with monkeypatch.context() as patch:
            patch.setattr(export, name, (*getattr(export, name), entry))
            payload = built(made_up_shell)[0]
            reported = outside_allow_list(payload)
            assert COUNT in leaves(payload) and reported and all(found.startswith(f"{part}/") and LATER in found for found in reported), name


def test_no_hidden_private_or_text_string_no_path_no_user_and_no_job_id_reaches_the_payload_or_the_client(made_up_shell, capsys):
    root = made_up_shell.parent.parent
    payload, notes = built(made_up_shell)
    assert leaked(leaves(payload), root) == []
    client = StandInClient()
    assert tracking_client.send(payload, client) == 0 and len(client.runs) == 1
    assert leaked(handed(client), root) == []
    assert leaked(export.render(payload, notes).splitlines(), root) == []
    # control: a planted string in an allowed field of run.json reaches the payload and the client, and the check reports it
    meta = json.loads((made_up_shell / "run.json").read_text(encoding="utf-8"))
    meta["agent_label"] = planted("allowed_field")
    (made_up_shell / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    payload, _notes = built(made_up_shell)
    client = StandInClient()
    tracking_client.send(payload, client)
    assert leaked(leaves(payload), root) and leaked(handed(client), root)


def count_found(run):
    """Every value of the payload, of what the client is handed and of the dry run text that holds COUNT."""
    payload, notes = built(run)
    client = StandInClient()
    tracking_client.send(payload, client)
    values = [*leaves(payload), *handed(client), *export.render(payload, notes).split()]
    return payload, [value for value in values if str(COUNT) in str(value)]


def test_no_count_of_the_faulty_components_and_no_fact_of_the_oracle_leaves(made_up_shell, capsys):
    set_scores(made_up_shell, **{name: str(COUNT) for name in HELD_BACK})
    payload, found = count_found(made_up_shell)
    (entry,) = payload["runs"]
    assert found == [] and {*entry["table"]["columns"], *(key.rsplit("/", 1)[-1] for key in entry["summary"])}.isdisjoint(HELD_BACK)
    # control: the same number in an outcome column that is sent reaches the payload, the client and the dry run
    set_scores(made_up_shell, cost=str(COUNT))
    _payload, found = count_found(made_up_shell)
    assert COUNT in found and str(COUNT) in "".join(map(str, found))


def test_a_json_cell_with_a_leading_space_stays_back(made_up_shell):
    root = made_up_shell.parent.parent
    set_scores(made_up_shell, cost=' {"made_up": "' + planted("json_cell") + '"}')
    (entry,) = built(made_up_shell)[0]["runs"]
    assert "cost" not in entry["table"]["columns"] and f"{STAMP}/outcome/cost" not in entry["summary"]
    assert leaked(leaves(entry), root) == []
    # control: a number in the same column is sent
    set_scores(made_up_shell, cost="3")
    (entry,) = built(made_up_shell)[0]["runs"]
    assert "cost" in entry["table"]["columns"] and entry["summary"][f"{STAMP}/outcome/cost"] == 3


def test_the_client_is_handed_the_payload_and_nothing_else(made_up_shell, capsys):
    payload, _notes = built(made_up_shell)
    client = StandInClient()
    assert tracking_client.send(payload, client) == 0
    (run,), (entry,) = client.runs, payload["runs"]
    names = ("id", "name", "group", "job_type", "tags")
    assert {name: run["init"][name] for name in names} == {name: entry[name] for name in names} and run["config"] == [entry["config"]]
    assert (run["init"]["entity"], run["init"]["project"]) == (PLACE["entity"], entry["project"]) and run["init"]["resume"] == "allow"
    assert run["summary"][0] == entry["summary"] and run["log"] == [] and run["finished"]
    table = run["summary"][1][entry["table"]["key"]]
    assert table.columns == entry["table"]["columns"] and table.data == entry["table"]["rows"] and table.allow_mixed_types
    fixed = leaves([run["init"][name] for name in ("resume", "mode", "settings")])
    known = {(type(value), repr(value)) for value in [*leaves(payload), *fixed]}
    assert [value for value in handed(client, local_folder=False) if (type(value), repr(value)) not in known] == []
    # control: a value that the payload does not hold is found
    run["summary"][0]["development/made_up"] = "made-up value"
    assert [value for value in handed(client, local_folder=False) if (type(value), repr(value)) not in known] == ["development/made_up",
                                                                                                               "made-up value"]


def test_a_development_run_sends_every_key_under_development_and_a_run_of_record_under_record(tmp_path):
    for stamp, kind in ((STAMP, STAMP), (None, export.RECORD)):
        run, _endpoints = made_up_run(tmp_path / str(kind), stamp=stamp)
        (entry,) = built(run)[0]["runs"]
        keys = [*entry["summary"], entry["table"]["key"]]
        assert entry["config"]["kind"] == kind and entry["tags"][0] == kind and entry["group"].startswith(f"{kind}:")
        assert keys and all(key.startswith(f"{kind}/") for key in keys)
    # control: the two kinds share no key, so no chart of a result holds a value of a development run
    record = built(tmp_path / export.RECORD / f"grid_{JOB}" / "made_up_run")[0]["runs"][0]["summary"]
    development = built(tmp_path / STAMP / f"grid_{JOB}" / "made_up_run")[0]["runs"][0]["summary"]
    assert set(record).isdisjoint(development) and len(record) == len(development)


def test_the_run_section_holds_the_run_s_own_episodes_and_the_cell_section_its_whole_cell(made_up_shell):
    """Two runs of one cell in one grid folder: the run section of each counts its own episodes, the cell section those of both."""
    meta = json.loads((made_up_shell / "run.json").read_text(encoding="utf-8"))
    other = made_up.write_run(made_up_shell.parent / "made_up_run_2", [(9, made_up.shifted(made_up.known_episode(), 60.0))], meta=meta,
                              process=(1, "none"))
    ledger_record(other, other.name, "2026-01-05T10:00:02", STAMP)
    first, second = export.build_payload([str(made_up_shell), str(other)], PLACE)[0]["runs"]
    assert (first["summary"][f"{STAMP}/run/episodes"], second["summary"][f"{STAMP}/run/episodes"]) == (2, 1)
    assert first["summary"][f"{STAMP}/cell/episodes"] == second["summary"][f"{STAMP}/cell/episodes"] == 3
    # control: the run alone gives one section equal to the other
    (alone,) = built(made_up_shell)[0]["runs"]
    assert alone["summary"][f"{STAMP}/run/episodes"] == alone["summary"][f"{STAMP}/cell/episodes"] == 2


# ----- the dry run, the place and the key -----

def test_the_dry_run_prints_the_whole_payload_and_sends_nothing(made_up_shell, monkeypatch, capsys, tmp_path):
    uv = fake_uv(tmp_path)
    monkeypatch.setenv("UV_BIN", str(uv))
    shell_place(monkeypatch)
    assert export.main([str(made_up_shell), "--dry-run"]) == 0
    printed = capsys.readouterr().out
    payload, notes = built(made_up_shell)
    assert export.render(payload, notes) in printed and "nothing is sent" in printed and not (tmp_path / "calls.txt").exists()
    (entry,) = payload["runs"]
    assert all(f"{name} = {json.dumps(value)}" in printed for name, value in entry["config"].items())
    assert f"row {len(entry['table']['rows'])}: run=" in printed and f"project {json.dumps(PLACE['projects'][STAMP])}" in printed
    assert "a send would be refused" not in printed
    # control: the same shell without --dry-run starts the client
    export.main([str(made_up_shell)])
    assert (tmp_path / "calls.txt").read_text(encoding="utf-8").startswith("run --python ")


def test_a_shell_that_declares_no_address_sends_nothing_and_the_dry_run_prints_the_place(made_up_shell, monkeypatch, capsys, tmp_path):
    uv = fake_uv(tmp_path)
    monkeypatch.setenv("UV_BIN", str(uv))
    shell_place(monkeypatch)
    monkeypatch.delenv(ADDRESS_VARIABLE)
    assert export.main([str(made_up_shell)]) == tracking_client.NOTHING_SENT
    out, err = capsys.readouterr()
    assert f"{ADDRESS_VARIABLE} is not set" in err and "address of the service: null" in out and not (tmp_path / "calls.txt").exists()
    assert export.main([str(made_up_shell), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"a send would be refused: {ADDRESS_VARIABLE} is not set" in out and f"team or user on the service: {json.dumps(PLACE['entity'])}" in out
    assert f"project {json.dumps(PLACE['projects'][STAMP])}" in out
    # control: the declared address is printed by the dry run, and the send starts the client
    monkeypatch.setenv(ADDRESS_VARIABLE, PLACE["address"])
    assert export.main([str(made_up_shell), "--dry-run"]) == 0 and f"address of the service: {json.dumps(PLACE['address'])}" in capsys.readouterr().out
    export.main([str(made_up_shell)])
    assert (tmp_path / "calls.txt").exists()


def test_a_shell_without_the_team_or_user_prints_the_payload_and_sends_nothing(made_up_shell, monkeypatch, capsys, tmp_path):
    uv = fake_uv(tmp_path)
    monkeypatch.setenv("UV_BIN", str(uv))
    shell_place(monkeypatch)
    monkeypatch.delenv(ENTITY_VARIABLE)
    assert export.main([str(made_up_shell)]) == tracking_client.NOTHING_SENT
    out, err = capsys.readouterr()
    assert f"{ENTITY_VARIABLE} is not set" in err and "tracked run 1 of 1" in out and not (tmp_path / "calls.txt").exists()
    # control: with the team or user the client is started
    monkeypatch.setenv(ENTITY_VARIABLE, PLACE["entity"])
    export.main([str(made_up_shell)])
    assert (tmp_path / "calls.txt").exists()


def test_one_project_named_for_both_kinds_is_refused(made_up_shell, monkeypatch, capsys, tmp_path):
    """Two project variables that differ in case alone name one project, which would put the development runs into the project
    that is opened: the send is refused with the payload printed, and the dry run names the refusal."""
    uv = fake_uv(tmp_path)
    monkeypatch.setenv("UV_BIN", str(uv))
    shell_place(monkeypatch)
    monkeypatch.setenv(PROJECT_VARIABLES[STAMP], PLACE["projects"][export.RECORD].upper())
    assert export.main([str(made_up_shell)]) == tracking_client.NOTHING_SENT
    out, err = capsys.readouterr()
    assert "name one project" in err and "tracked run 1 of 1" in out and not (tmp_path / "calls.txt").exists()
    assert export.main([str(made_up_shell), "--dry-run"]) == 0
    assert f"a send would be refused: {PROJECT_VARIABLES[export.RECORD]} and {PROJECT_VARIABLES[STAMP]} name one project" in capsys.readouterr().out
    # control: two projects let the client start
    shell_place(monkeypatch)
    export.main([str(made_up_shell)])
    assert (tmp_path / "calls.txt").exists()


def test_each_tracked_run_goes_to_the_project_of_the_kind_that_its_own_ledger_records(tmp_path):
    record, _endpoints = made_up_run(tmp_path / "of_record", stamp=None)
    development, _endpoints = made_up_run(tmp_path / "of_development")
    entries = export.build_payload([str(record), str(development)], PLACE)[0]["runs"]
    assert [(entry["config"]["kind"], entry["project"]) for entry in entries] == [(export.RECORD, PLACE["projects"][export.RECORD]),
                                                                                  (STAMP, PLACE["projects"][STAMP])]
    # control: the ledger alone turns the routing, run.json and every other file as they were
    set_ledger(record, **{export.START_CHECK: ["made-up reason"]})
    set_ledger(development, **{export.START_CHECK: []})
    entries = export.build_payload([str(record), str(development)], PLACE)[0]["runs"]
    assert [entry["project"] for entry in entries] == [PLACE["projects"][STAMP], PLACE["projects"][export.RECORD]]
    # a ledger that records no start check gives a development run, never one of record
    held = json.loads((development / ledger.LEDGER).read_text(encoding="utf-8"))
    (development / ledger.LEDGER).write_text(json.dumps({key: value for key, value in held.items() if key != export.START_CHECK}),
                                             encoding="utf-8")
    assert export.build_payload([str(development)], PLACE)[0]["runs"][0]["project"] == PLACE["projects"][STAMP]


def test_the_kind_of_a_run_ignores_the_shell(made_up_shell, monkeypatch):
    """A shell that names the kind of a run, as a job's shell names RUN_KIND, turns no development run into one of record."""
    monkeypatch.setenv("RUN_KIND", export.RECORD)
    (entry,) = built(made_up_shell)[0]["runs"]
    assert entry["config"]["kind"] == STAMP and entry["project"] == PLACE["projects"][STAMP]
    # control: the ledger of a registered run makes it a run of record under the same shell
    set_ledger(made_up_shell, **{export.START_CHECK: []})
    assert built(made_up_shell)[0]["runs"][0]["config"]["kind"] == export.RECORD


def test_the_shell_names_the_project_of_each_kind_by_its_own_variable():
    environ = {ADDRESS_VARIABLE: PLACE["address"] + "/", ENTITY_VARIABLE: PLACE["entity"],
               **{name: PLACE["projects"][kind] for kind, name in PROJECT_VARIABLES.items()}}
    assert export.place_of(environ) == PLACE  # the address as the client takes it, without its closing slash
    # control: a shell without the project of the development runs names none for them, and never the project of record
    del environ[PROJECT_VARIABLES[STAMP]]
    assert export.place_of(environ)["projects"] == {export.RECORD: PLACE["projects"][export.RECORD], STAMP: None}


def test_a_quantity_of_the_profile_is_sent_as_a_number_alone(made_up_shell, monkeypatch):
    """A column of the profile's groups that held a text, as a note would, sends nothing of it; the numbers of the column stay. The
    text stands in a count that no summary of a cell adds up."""
    profiled = export.profiled

    def with_a_text(paths):
        rows, summaries, views = profiled(paths)
        return [{**row, "counted_calls": planted("text_in_a_quantity")} if row["seed"] == "7" else row for row in rows], summaries, views
    monkeypatch.setattr(export, "profiled", with_a_text)
    (entry,) = built(made_up_shell)[0]["runs"]
    column = entry["table"]["columns"].index("counted_calls")
    assert entry["table"]["rows"][0][column] is None and leaked(leaves(entry), made_up_shell.parent.parent) == []
    # control: the episode whose column holds a number sends it
    assert isinstance(entry["table"]["rows"][1][column], int)


def test_a_development_run_whose_project_is_not_named_is_printed_and_kept_back(made_up_shell, monkeypatch, capsys, tmp_path):
    uv = fake_uv(tmp_path)
    monkeypatch.setenv("UV_BIN", str(uv))
    shell_place(monkeypatch, development=False)
    assert export.main([str(made_up_shell)]) == tracking_client.NOTHING_SENT
    out, err = capsys.readouterr()
    assert "kept back" in out and "kept back" in err and not (tmp_path / "calls.txt").exists()
    # control: with the project of the development runs named, the run is sent there
    shell_place(monkeypatch)
    export.main([str(made_up_shell)])
    assert (tmp_path / "calls.txt").exists()


def test_the_client_keeps_back_an_entry_that_names_no_project(made_up_shell, capsys):
    payload, _notes = built(made_up_shell)
    client = StandInClient()
    unnamed = {**payload, "runs": [{**entry, "project": None} for entry in payload["runs"]]}
    assert tracking_client.send(unnamed, client) == 0 and client.runs == [] and "KEPT_BACK" in capsys.readouterr().err
    # control: the entry with its project is sent
    assert tracking_client.send(payload, client) == 0 and [run["init"]["project"] for run in client.runs] == [PLACE["projects"][STAMP]]


def test_a_client_without_a_key_sends_nothing_and_says_so(made_up_shell, capsys):
    payload, _notes = built(made_up_shell)
    client = StandInClient(key=False)
    assert tracking_client.send(payload, client) == tracking_client.NOTHING_SENT and client.runs == []
    assert client.logins == [{"prompt": False, "verify": False, "host": PLACE["address"]}] and "nothing is sent" in capsys.readouterr().err
    # control: a client with a key sends
    client = StandInClient(key=True)
    assert tracking_client.send(payload, client) == 0 and len(client.runs) == 1


def test_the_client_goes_to_the_declared_address_alone(made_up_shell, capsys):
    payload, _notes = built(made_up_shell)
    client = StandInClient()
    assert tracking_client.send(payload, client) == 0
    assert client.logins == [{"prompt": False, "verify": False, "host": PLACE["address"]}]
    assert [run["init"]["settings"]["base_url"] for run in client.runs] == [PLACE["address"]]
    # control: a payload that declares no address starts neither the login nor a tracked run
    client = StandInClient()
    assert tracking_client.send({**payload, "address": None}, client) == tracking_client.NOTHING_SENT
    assert client.logins == [] and client.runs == [] and "declares no address" in capsys.readouterr().err


@pytest.mark.parametrize("placed", [{"id": "made-up-other-id"}, {"project": "made-up-other-project"}, {"entity": "made-up-other-team"}])
def test_a_tracked_run_that_the_client_opens_elsewhere_stops_the_export(made_up_shell, capsys, placed):
    """A client that a variable of its own turns opens the tracked run under another id, project or team or user: the export stops
    before the configuration, the summary, the table or the history is sent, and the tracked run is closed as failed."""
    payload, _notes = built(made_up_shell)
    client = StandInClient(placed=placed)
    assert tracking_client.send(payload, client) == tracking_client.WRONG_PLACE and "WRONG_PLACE" in capsys.readouterr().err
    (run,) = client.runs
    assert "config" not in run["init"] and (run["config"], run["summary"], run["log"], run["finished"], run["exit_code"]) == ([], [], [], True, 1)
    # control: the names in another case are the place that the service resolves, and the tracked run is sent whole
    (entry,) = payload["runs"]
    client = StandInClient(placed={"project": entry["project"].upper(), "entity": PLACE["entity"].upper()})
    assert tracking_client.send(payload, client) == 0 and client.runs[0]["config"] == [entry["config"]] and client.runs[0]["summary"]


def test_an_open_key_file_stops_the_login_and_the_send(made_up_shell, monkeypatch, tmp_path, capsys):
    """The login of the client keeps the mode of a key file that exists, so a file that group or others may read stops the login
    and the send before uv starts."""
    uv_calls, _client_calls = shipped_shell(monkeypatch, tmp_path / "shell")
    key = Path(os.environ[export.KEY_FILE])
    key.write_text("machine api.made-up.invalid\n  login user\n  password made-up\n", encoding="utf-8")
    key.chmod(0o644)
    assert export.main(["--login"]) == 2 and export.main([str(made_up_shell)]) == 2
    assert "chmod 600" in capsys.readouterr().err and not uv_calls.exists()
    # control: the file readable by its owner alone lets both start
    key.chmod(0o600)
    assert export.main(["--login"]) == 0 and export.main([str(made_up_shell)]) == 0
    assert [call["argv"][0] for call in json.loads(uv_calls.read_text(encoding="utf-8"))] == ["tool", "run"]


# ----- the id, the second export, the training folder and the folders that are refused -----

def test_a_folder_exported_again_or_from_its_synced_copy_updates_one_tracked_run(made_up_shell, tmp_path, capsys):
    first = built(made_up_shell)[0]["runs"][0]
    copy_root = tmp_path / "synced"
    shutil.copytree(made_up_shell.parent, copy_root / made_up_shell.parent.name, symlinks=True)
    synced = built(copy_root / made_up_shell.parent.name / made_up_shell.name)[0]["runs"][0]
    assert first["id"] == built(made_up_shell)[0]["runs"][0]["id"] == synced["id"]
    assert first["id"] == ledger.canonical_sha256({"run": "made_up_run", "label": "made_up_run", "started": "2026-01-05T10:00:00"})
    client = StandInClient(held={first["id"]: 1})
    tracking_client.send({"address": PLACE["address"], "entity": PLACE["entity"], "runs": [first, synced]}, client)
    assert [run["init"]["id"] for run in client.runs] == [first["id"]] * 2 and {run["init"]["resume"] for run in client.runs} == {"allow"}
    # control: another start in the ledger is another run and gives another id
    set_ledger(made_up_shell, started="2026-01-05T10:00:01")
    assert built(made_up_shell)[0]["runs"][0]["id"] != first["id"]


def made_up_training(root, steps=(10, 20), manifest=None):
    """A made-up training folder checkpoints/<adapter> with SOURCE.json and a checkpoint per step of steps, the later one holding
    the longer log history, a planted path and trial name among the fields that stay back, and the adapter's manifest when given."""
    folder = Path(root) / "training" / export.CHECKPOINTS / "rft_made_up_f1"
    folder.mkdir(parents=True)
    (folder / export.SOURCE).write_text(json.dumps({"examples_sha256": "e" * 64, "script_sha256": "f" * 64}), encoding="utf-8")
    history = [{"loss": 2.0 - step / 10, "epoch": step / 20, "step": step, "learning_rate": 0.001, "grad_norm": 1.0} for step in (5, 10, 15, 20)]
    for step in steps:
        state = {"global_step": step, "epoch": step / 20, "max_steps": 20, "logging_steps": 5, "save_steps": 10, "eval_steps": 500,
                 "train_batch_size": 4, "num_train_epochs": 1, "log_history": [row for row in history if row["step"] <= step],
                 "best_model_checkpoint": f"/home/{USER}/{planted('best_checkpoint')}", "trial_name": planted("trial")}
        (folder / f"{export.CHECKPOINT}{step}").mkdir()
        (folder / f"{export.CHECKPOINT}{step}" / export.TRAINER_STATE).write_text(json.dumps(state), encoding="utf-8")
    if manifest is not None:
        adapter = Path(root) / "training" / export.ADAPTERS / folder.name
        adapter.mkdir(parents=True)
        (adapter / export.ADAPTER_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def test_a_training_folder_gives_its_trainer_configuration_and_loss_history_from_the_latest_checkpoint(tmp_path):
    folder = made_up_training(tmp_path)
    (entry,) = export.build_payload([str(folder)], PLACE)[0]["runs"]
    assert outside_allow_list({"address": None, "entity": None, "runs": [entry]}) == [] and leaked(leaves(entry), tmp_path) == []
    assert entry["project"] == PLACE["projects"][STAMP]
    assert entry["config"] == {"kind": STAMP, "train_batch_size": 4, "num_train_epochs": 1, "max_steps": 20, "logging_steps": 5,
                               "save_steps": 10, "eval_steps": 500}
    assert entry["summary"] == {f"{STAMP}/global_step": 20, f"{STAMP}/epoch": 1.0} and entry["table"] is None
    assert [(row["step"], row["values"][f"{STAMP}/loss"]) for row in entry["history"]] == [(5, 1.5), (10, 1.0), (15, 0.5), (20, 0.0)]
    # control: a training whose latest checkpoint is the earlier one gives the shorter history
    earlier = made_up_training(tmp_path / "earlier", steps=(10,))
    assert [row["step"] for row in export.build_payload([str(earlier)], PLACE)[0]["runs"][0]["history"]] == [5, 10]


OF_RECORD = {"stamp": None, "smoke": False, "training_split": "registered"}  # the manifest of an adapter of record, as train_lora.py writes it


@pytest.mark.parametrize("manifest, kind", [
    (OF_RECORD, export.RECORD), (None, STAMP), ({**OF_RECORD, "stamp": STAMP}, STAMP), ({**OF_RECORD, "smoke": True}, STAMP),
    ({**OF_RECORD, "training_split": "development"}, STAMP), ({**OF_RECORD, "smoke": None}, STAMP), ({**OF_RECORD, "smoke": 0}, STAMP),
    *(({key: value for key, value in OF_RECORD.items() if key != left_out}, STAMP) for left_out in OF_RECORD)])
def test_a_training_is_of_record_only_when_the_manifest_of_its_adapter_records_it_so(tmp_path, manifest, kind):
    """Of record only with a manifest that names no stamp, no smoke run (False itself, not another value that reads as false) and
    the registered split, each key present; the first case is the one of record, and every other, a key turned or left out, is the
    control."""
    folder = made_up_training(tmp_path, manifest=manifest)
    (entry,) = export.build_payload([str(folder)], PLACE)[0]["runs"]
    assert entry["config"]["kind"] == kind and set(entry["summary"]) == {f"{kind}/global_step", f"{kind}/epoch"}
    assert entry["project"] == PLACE["projects"][kind]


def test_two_trainings_of_one_adapter_from_other_inputs_get_two_ids(tmp_path):
    first, second = made_up_training(tmp_path / "first"), made_up_training(tmp_path / "second")

    def id_of(folder):
        return export.build_payload([str(folder)], PLACE)[0]["runs"][0]["id"]
    assert id_of(first) == id_of(second)  # one folder name and one SOURCE.json name one training
    # the change: other examples under the same folder name give another tracked run
    (second / export.SOURCE).write_text(json.dumps({"examples_sha256": "d" * 64, "script_sha256": "f" * 64}), encoding="utf-8")
    assert id_of(first) != id_of(second)


def test_a_training_exported_again_sends_no_history_row_twice(tmp_path, capsys):
    (entry,) = export.build_payload([str(made_up_training(tmp_path))], PLACE)[0]["runs"]
    taken_up = StandInClient(held={entry["id"]: 16})
    tracking_client.send({"address": PLACE["address"], "entity": PLACE["entity"], "runs": [entry]}, taken_up)
    assert [row["step"] for row in taken_up.runs[0]["log"]] == [20]
    # control: a tracked run that holds no row yet is sent every row
    new = StandInClient()
    tracking_client.send({"address": PLACE["address"], "entity": PLACE["entity"], "runs": [entry]}, new)
    assert [row["step"] for row in new.runs[0]["log"]] == [5, 10, 15, 20]


def test_folders_that_cannot_be_exported_are_refused_by_name_before_anything_is_read(made_up_shell, tmp_path, capsys):
    plain = tmp_path / "plain_folder"
    plain.mkdir()
    for folder in (plain, tmp_path / f"grid_{JOB}0", tmp_path / "no_such_run_folder"):  # a folder, and two names of no folder
        assert export.main([str(folder), "--dry-run"]) == 2
        out, err = capsys.readouterr()
        assert out == "" and str(folder) in err
    record = json.loads((made_up_shell / ledger.LEDGER).read_text(encoding="utf-8"))
    set_ledger(made_up_shell, started=None)
    assert export.main([str(made_up_shell), "--dry-run"]) == 2 and "lacks started" in capsys.readouterr().err
    # control: the run folder with its whole record is exported
    (made_up_shell / ledger.LEDGER).write_text(json.dumps(record), encoding="utf-8")
    assert export.main([str(made_up_shell), "--dry-run"]) == 0


def test_a_run_that_its_ledger_does_not_record_finished_is_refused(made_up_shell, capsys):
    """A run whose ledger records it running, or records no status, may still play, so its values may change after the export;
    a grid folder that holds such a run, one without run.json among them, is refused by the name of that run."""
    for status in ("running", None):
        set_ledger(made_up_shell, status=status)
        assert export.main([str(made_up_shell), "--dry-run"]) == 2
        out, err = capsys.readouterr()
        assert out == "" and f"not {export.FINISHED!r}" in err and made_up_shell.name in err
    set_ledger(made_up_shell, status=export.FINISHED)
    cut = made_up.write_run(made_up_shell.parent / "made_up_cut_run", [(9, made_up.known_episode())], unfinished=True)
    set_ledger(cut, run=cut.name, label=cut.name, started="2026-01-05T10:00:03", status="running")
    assert export.main([str(made_up_shell.parent), "--dry-run"]) == 2
    out, err = capsys.readouterr()
    assert out == "" and cut.name in err and f"not {export.FINISHED!r}" in err and "status 'running'" in err
    # control: the finished run folder of the same grid folder is exported
    assert export.main([str(made_up_shell), "--dry-run"]) == 0


# ----- the client, its settings and the shipped path through uv -----

def test_the_client_keeps_every_setting_that_stops_it_from_sending_the_machine_by_itself(made_up_shell, capsys):
    assert {name: tracking_client.SETTINGS.get(name) for name in REQUIRED} == REQUIRED
    client = StandInClient()
    tracking_client.send(built(made_up_shell)[0], client)
    assert client.runs[0]["init"]["settings"] == {**tracking_client.SETTINGS, "base_url": PLACE["address"]}


@pytest.mark.parametrize("name", sorted(REQUIRED))
def test_a_setting_dropped_from_the_client_is_found(name):
    """The control of the test above: SETTINGS without one required setting, or with its value turned, fails the comparison."""
    dropped = {key: value for key, value in tracking_client.SETTINGS.items() if key != name}
    turned = {**tracking_client.SETTINGS, name: "made-up"}
    assert {key: dropped.get(key) for key in REQUIRED} != REQUIRED and {key: turned.get(key) for key in REQUIRED} != REQUIRED


def test_the_header_of_the_client_pins_one_version_of_one_package():
    (pin,) = tracking_client.pinned()
    package, equals, version = pin.partition("==")
    assert package == tracking_client.COMMAND and equals and version.replace(".", "").isdigit()
    # control: a header without its block is no header
    assert tracking_client.HEADER.search("# dependencies = []\n") is None


def client_environment_problems(environ, folder, home):
    """What is wrong with the environment of one start of the client: a variable of the client beside those of CLIENT_WANDB,
    PYTHONPATH, which points at the tree, the key missing, another home folder, the recorder of the client's errors on, or a
    folder of the client outside folder, the temporary folder of the start."""
    problems = [f"variable {name}" for name in environ if name.startswith("WANDB_") and name not in CLIENT_WANDB]
    problems += ["PYTHONPATH"] if "PYTHONPATH" in environ else []
    problems += ["no key"] if environ.get("WANDB_API_KEY") != planted("wandb_key") else []
    problems += ["home"] if environ.get("HOME") != str(home) else []
    problems += ["error recorder on"] if environ.get("WANDB_ERROR_REPORTING") != "false" else []
    return problems + [f"folder {name}" for name in CLIENT_WANDB[2:]
                       if Path(environ.get(name, "/")).parent.resolve() != Path(folder).resolve()]


def test_the_client_starts_with_the_listed_variables_alone_its_error_recorder_off_and_its_folders_in_the_start_folder(made_up_shell, monkeypatch, tmp_path):
    for name, value in SHELL_CLIENT_VARIABLES.items():
        monkeypatch.setenv(name, value)
    shell_place(monkeypatch)
    monkeypatch.setenv("PYTHONPATH", f"{ROOT}:{SRC}")
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "made_up_home" / "data_of_the_client"))
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "made_up_uv_cache"))
    folder = tmp_path / "start"
    environ = export.client_environment(folder)
    assert client_environment_problems(environ, folder, tmp_path / "made_up_home") == []
    assert environ["UV_CACHE_DIR"] == str(tmp_path / "made_up_uv_cache") and environ[export.KEY_FILE] == os.environ[export.KEY_FILE]
    # control: each wrong form of the environment is found
    forms = {"variable WANDB_SWEEP_ID": {**environ, "WANDB_SWEEP_ID": "made-up-sweep"},
             "error recorder on": {name: value for name, value in environ.items() if name != "WANDB_ERROR_REPORTING"},
             "folder WANDB_DATA_DIR": {**environ, "WANDB_DATA_DIR": os.environ["WANDB_DATA_DIR"]},
             "PYTHONPATH": {**environ, "PYTHONPATH": f"{ROOT}:{SRC}"}, "no key": {**environ, "WANDB_API_KEY": ""}}
    assert {name: client_environment_problems(form, folder, tmp_path / "made_up_home") for name, form in forms.items()} == {
        name: [name] for name in forms}


STAND_IN_PACKAGE = '''"""A made-up stand-in of the client package for the test of the shipped path: every call goes into the file at RECORD, and
nothing reaches a network. Its login finds a key when WANDB_API_KEY is set, as the client's own login does."""
import json
import os
from pathlib import Path

RECORD = Path({record!r})


def _write(entry):
    calls = json.loads(RECORD.read_text()) if RECORD.exists() else []
    RECORD.write_text(json.dumps([*calls, entry], default=lambda table: {{"columns": table.columns, "data": table.data}}))


class Table:
    def __init__(self, columns=None, data=None, allow_mixed_types=False):
        self.columns, self.data = columns, data


class _Summary:
    def update(self, values):
        _write({{"summary": values}})


class _Config:
    def update(self, values, allow_val_change=None):
        _write({{"config": values}})


class _Run:
    def __init__(self, arguments):
        self.id, self.project, self.entity = arguments.get("id"), arguments.get("project"), arguments.get("entity")
        self.step, self.summary, self.config = 0, _Summary(), _Config()

    def log(self, values, step=None):
        _write({{"log": values, "step": step}})

    def finish(self, exit_code=None):
        _write({{"finish": True}})


def login(prompt=True, verify=True, host=None):
    _write({{"login": {{"prompt": prompt, "verify": verify, "host": host}}}})
    return bool(os.environ.get("WANDB_API_KEY"))


def init(**arguments):
    _write({{"init": {{name: value for name, value in arguments.items() if name != "dir"}}}})
    return _Run(arguments)
'''
FAKE_UV = '''#!{python}
import json, os, subprocess, sys
from pathlib import Path
calls = Path({calls!r})
seen = json.loads(calls.read_text()) if calls.exists() else []
payload = json.loads(Path(sys.argv[-1]).read_text()) if sys.argv[1:2] == ["run"] else None
calls.write_text(json.dumps([*seen, {{"argv": sys.argv[1:], "environ": dict(os.environ), "cwd": os.getcwd(), "payload": payload}}]))
if sys.argv[1:2] == ["run"]:
    sys.exit(subprocess.run([{python!r}, sys.argv[-2], sys.argv[-1]], env=dict(os.environ, PYTHONPATH={stand_in!r})).returncode)
'''


def shipped_shell(monkeypatch, folder):
    """A stand-in uv on UV_BIN that records each call with its whole environment and starts the client script with the stand-in
    package on its path, PLACE, the PYTHONPATH of a job's shell, which points at the tree, every variable of the client that a
    shell may hold (SHELL_CLIENT_VARIABLES), a data folder of the client in the home folder, and a key in the shell."""
    stand_in = Path(folder) / "stand_in"
    stand_in.mkdir(parents=True)
    (stand_in / f"{tracking_client.COMMAND}.py").write_text(STAND_IN_PACKAGE.format(record=str(Path(folder) / "client_calls.json")),
                                                             encoding="utf-8")
    uv = Path(folder) / "uv"
    uv.write_text(FAKE_UV.format(python=sys.executable, calls=str(Path(folder) / "uv_calls.json"), stand_in=str(stand_in)), encoding="utf-8")
    uv.chmod(0o755)
    monkeypatch.setenv("UV_BIN", str(uv))
    monkeypatch.setenv("PYTHONPATH", f"{ROOT}:{SRC}")
    for name, value in SHELL_CLIENT_VARIABLES.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("WANDB_DATA_DIR", str(Path(os.environ["HOME"]) / "data_of_the_client"))
    shell_place(monkeypatch)
    return Path(folder) / "uv_calls.json", Path(folder) / "client_calls.json"


def test_the_shipped_path_starts_the_client_through_uv_with_the_payload_alone(made_up_shell, monkeypatch, tmp_path, capfd):
    uv_calls, client_calls = shipped_shell(monkeypatch, tmp_path / "shell")
    before = {path: path.read_bytes() for path in made_up_shell.parent.rglob("*") if path.is_file()}
    assert export.main([str(made_up_shell.parent)]) == 0  # the grid folder stands for its run folder
    assert {path: path.read_bytes() for path in made_up_shell.parent.rglob("*") if path.is_file()} == before  # no file of it changed
    (call,) = json.loads(uv_calls.read_text(encoding="utf-8"))
    assert call["argv"][:5] == ["run", "--python", sys.executable, "--script", str(export.CLIENT)] and len(call["argv"]) == 6
    assert Path(call["argv"][5]).parent.resolve() == Path(call["cwd"]).resolve()  # the payload lies in the temporary folder of the start
    assert client_environment_problems(call["environ"], call["cwd"], tmp_path / "made_up_home") == []
    assert not Path(call["cwd"]).exists()  # the temporary folder, with the payload and the folders of the client, is gone
    payload = json.loads(json.dumps(built(made_up_shell)[0]))
    assert call["payload"] == payload and payload["address"] == PLACE["address"]
    calls = json.loads(client_calls.read_text(encoding="utf-8"))
    (init,) = [entry["init"] for entry in calls if "init" in entry]
    (entry,) = payload["runs"]
    assert calls[0] == {"login": {"prompt": False, "verify": False, "host": PLACE["address"]}} and init["id"] == entry["id"]
    assert {"config": entry["config"]} in calls and init["project"] == PLACE["projects"][STAMP] and init["entity"] == PLACE["entity"]
    assert init["settings"] == {**tracking_client.SETTINGS, "base_url": PLACE["address"]} and {"finish": True} in calls
    assert f"SENT {entry['name']} as {entry['id']}" in capfd.readouterr().out  # the client's own line, from its process
    # control: a shell without a key, where the client's login finds none, sends nothing, and the command prints the payload instead
    monkeypatch.delenv("WANDB_API_KEY")
    client_calls.unlink()
    assert export.main([str(made_up_shell)]) == tracking_client.NOTHING_SENT
    assert [entry for entry in json.loads(client_calls.read_text(encoding="utf-8")) if "init" in entry] == []
    assert "tracked run 1 of 1" in capfd.readouterr().out


def test_the_login_runs_the_pinned_client_through_uv_for_the_declared_address_alone(made_up_shell, monkeypatch, tmp_path, capsys):
    uv_calls, _client_calls = shipped_shell(monkeypatch, tmp_path / "shell")
    monkeypatch.setenv(ADDRESS_VARIABLE, PLACE["address"] + "/")
    assert export.main(["--login"]) == 0
    (call,) = json.loads(uv_calls.read_text(encoding="utf-8"))
    assert call["argv"] == ["tool", "run", "--python", sys.executable, "--from", *tracking_client.pinned(), tracking_client.COMMAND, "login",
                            "--host", PLACE["address"]]
    assert client_environment_problems(call["environ"], call["cwd"], tmp_path / "made_up_home") == []
    # control: a shell that declares no address starts no login, and a dry run starts no uv at all
    uv_calls.unlink()
    monkeypatch.delenv(ADDRESS_VARIABLE)
    assert export.main(["--login"]) == 2 and ADDRESS_VARIABLE in capsys.readouterr().err and not uv_calls.exists()
    assert export.main([str(made_up_shell), "--dry-run"]) == 0 and not uv_calls.exists()


# ----- the script of the cluster jobs in the login shell -----

CLUSTER_SCRIPT = ROOT / "CausalDiagnosis" / "cluster_jobs" / "4_after_run" / "tracking_export.sh"
PYTHON_STAND_IN = '''#!/bin/sh
{{ printf '%s\\n' "$@"; echo "ADDRESS=$WANDB_BASE_URL"; echo "ENTITY=$WANDB_ENTITY"; echo "PROJECT=$WANDB_PROJECT"
   echo "DEVELOPMENT=$WANDB_PROJECT_DEVELOPMENT"; echo "UV_BIN=$UV_BIN"; echo "UV_CACHE_DIR=$UV_CACHE_DIR"; }} > {calls}
'''


def made_up_jobs_folder(root, settings=True):
    """A made-up tree with the cluster script of the export, a submit.sh, and an env.sh of made-up values with a fork_python that
    puts a stand-in python first on PATH, which records its arguments and the variables it sees; settings False writes no env.sh."""
    jobs = Path(root) / "tree" / "CausalDiagnosis" / "cluster_jobs"
    (jobs / "4_after_run").mkdir(parents=True)
    (Path(root) / "tree" / "uv.lock").write_text("made up\n", encoding="utf-8")
    (jobs / "submit.sh").write_text("# made up\n", encoding="utf-8")
    shutil.copy(CLUSTER_SCRIPT, jobs / "4_after_run" / CLUSTER_SCRIPT.name)
    bin_folder = Path(root) / "bin"
    bin_folder.mkdir()
    (bin_folder / "python").write_text(PYTHON_STAND_IN.format(calls=Path(root) / "python_calls.txt"), encoding="utf-8")
    (bin_folder / "python").chmod(0o755)
    if settings:
        (jobs / "env.sh").write_text(
            f"ROOT={root}/root\nUV_BIN={root}/made_up_uv\nWANDB_BASE_URL=https://api.made-up.invalid\nWANDB_ENTITY=made-up-team\n"
            "WANDB_PROJECT=made-up-record-project\nWANDB_PROJECT_DEVELOPMENT=made-up-development-project\nCD_FUNCTIONS=1\n"  # pragma: allowlist secret
            f"fork_python() {{ PATH={bin_folder}:$PATH; }}\n", encoding="utf-8")
    return jobs / "4_after_run" / CLUSTER_SCRIPT.name


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_cluster_script_runs_the_command_with_the_names_of_env_sh_and_the_uv_of_the_tree(tmp_path):
    script = made_up_jobs_folder(tmp_path)
    done = subprocess.run(["bash", str(script), "--dry-run", "made_up_folder"], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin"}, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    calls = (tmp_path / "python_calls.txt").read_text(encoding="utf-8").splitlines()
    assert calls[:7] == ["-P", "-m", "CausalDiagnosis", "performance", "tracking-export", "--dry-run", "made_up_folder"]
    assert calls[7:] == ["ADDRESS=https://api.made-up.invalid", "ENTITY=made-up-team", "PROJECT=made-up-record-project",
                         "DEVELOPMENT=made-up-development-project", f"UV_BIN={tmp_path}/made_up_uv",
                         f"UV_CACHE_DIR={tmp_path}/root/tools/uv-cache"]
    # control: without env.sh the script stops before the command, which never runs
    (tmp_path / "python_calls.txt").unlink()
    bare = made_up_jobs_folder(tmp_path / "bare", settings=False)
    done = subprocess.run(["bash", str(bare), "--dry-run", "made_up_folder"], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert done.returncode == 2 and "no env.sh" in done.stderr and not (tmp_path / "bare" / "python_calls.txt").exists()
