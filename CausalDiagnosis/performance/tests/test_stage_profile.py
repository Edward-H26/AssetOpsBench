"""stage_profile.py and run_files.py: every stage of the profile and every check on made-up episodes with planted times, each with a
control that fails when the formula is wrong, and the episodes that the profile leaves out by name."""
import json
import shutil

import pytest

from CausalDiagnosis.performance import run_files, stage_profile
from CausalDiagnosis.performance.tests import made_up
from CausalDiagnosis.performance.tests.made_up import MCP, RIG, wall


def profile_of(tmp_path, parts, name="run"):
    run = made_up.write_run(tmp_path / name, [(7, parts)])
    _name, row, folder = run_files.episode_entries(run)[0]
    return stage_profile.episode_profile(row, folder, run_files.episode_files(folder))[0]


def test_the_known_episode_gives_back_every_planted_stage_and_its_share(tmp_path):
    parts = made_up.known_episode()
    profile = profile_of(tmp_path, parts)
    assert profile["left_out"] == ""
    for name, value in parts["answer"].items():
        assert profile[name] == pytest.approx(value, abs=1e-9), name
    for name, share in zip(stage_profile.SECONDS, stage_profile.SHARES):
        assert profile[share] == pytest.approx(parts["answer"][name] / 16.5, abs=1e-12), share
    # the parts that the rest leaves after the subtraction add up to the total, the flush and the failed seconds inside theirs
    parts_of_total = ("creation_s", "end_s", "model_s", "env_tools_client_s", "diagnostics_client_s", "server_starts_s", "rest_s")
    assert sum(profile[name] for name in parts_of_total) == pytest.approx(profile["total_s"])


def test_the_total_is_the_wall_s_cell_and_the_end_takes_its_rounding_without_a_failed_check(tmp_path):
    parts = made_up.known_episode()
    parts["row"]["wall_s"] = "16.0"
    parts["state"]["episode_ended_at"] = wall(16.04)  # the agent's part ends after the rounded total
    profile = profile_of(tmp_path, parts)
    assert profile["total_s"] == 16.0 and profile["agent_part_s"] == pytest.approx(14.04)
    assert profile["end_s"] == pytest.approx(-0.04) and profile["rest_negative"] == 0
    control = profile_of(tmp_path, made_up.known_episode(), "control")
    assert control["end_s"] == pytest.approx(0.5)


def test_the_creation_runs_from_the_start_of_the_state_to_the_handover(tmp_path):
    parts = made_up.known_episode()
    parts["state"]["episode_started_at"] = wall(-1.0)
    profile = profile_of(tmp_path, parts)
    assert profile["creation_s"] == pytest.approx(3.0)
    assert profile["end_s"] == pytest.approx(-0.5)  # the control: the total stays the wall_s cell, so the end gives the second back
    assert profile["agent_part_s"] == pytest.approx(14.0)


def test_the_cache_flush_comes_from_the_state_and_is_empty_without_a_flush(tmp_path):
    assert profile_of(tmp_path, made_up.known_episode())["cache_flush_s"] == pytest.approx(1.5)
    parts = made_up.known_episode()
    del parts["state"]["cache_flush"]
    profile = profile_of(tmp_path, parts, "control")
    assert profile["cache_flush_s"] is None and profile["share_cache_flush"] is None and profile["creation_s"] == pytest.approx(2.0)


def test_the_model_time_is_the_union_of_the_generation_spans_inside_the_agent_part(tmp_path):
    parts = made_up.known_episode()
    parts["llm_calls"] += [{"started_at": wall(5.0), "ended_at": wall(6.0), "seconds": 1.0},  # inside the first call
                           {"started_at": wall(1.0), "ended_at": wall(2.5), "seconds": 1.5},  # starts before the handover
                           {"kind": "tokenize", "started_at": wall(13.0), "ended_at": wall(13.5), "seconds": 0.5, "error": "made up"}]
    profile = profile_of(tmp_path, parts)
    # the union clipped to [H, end]: 6.5 of the known calls and the half second of the second span after the handover
    assert profile["model_s"] == pytest.approx(7.0)
    plain = sum(call["ended_at"] - call["started_at"] for call in parts["llm_calls"])
    assert plain == pytest.approx(9.5) and profile["model_s"] != pytest.approx(plain)  # the control: a plain sum reads 9.5
    assert profile["model_calls"] == 5  # the failed token count is no generation call


def test_the_failed_model_seconds_come_from_scores_csv_and_not_from_the_lagging_state(tmp_path):
    parts = made_up.known_episode()
    assert parts["state"]["failed_model_seconds"] == 0.25 and parts["row"]["failed_model_s"] == "0.5"
    profile = profile_of(tmp_path, parts)
    assert profile["failed_model_s"] == pytest.approx(0.5) and profile["failed_model_attempts"] == 1
    parts["row"]["failed_model_s"] = "0.75"
    assert profile_of(tmp_path, parts, "control")["failed_model_s"] == pytest.approx(0.75)


def test_the_tool_service_holds_the_agent_s_counted_calls_alone(tmp_path):
    parts = made_up.known_episode()
    # a harness finish that its caller stamped: its end line repeats its entry, so it has no measured service
    parts["lines"] += [{"kind": "enter", "server": RIG, "tool": "finish", "wall": wall(15.75), "mono": 15.75, "thread_cpu": 1.5},
                       {"kind": "call", "server": RIG, "tool": "finish", "stepped": True, "enter_n": 16, "end_wall": wall(15.75),
                        "end_mono": 15.75, "end_thread_cpu": 1.5}]
    profile = profile_of(tmp_path, parts)
    assert profile["tool_service_s"] == pytest.approx(1.75)
    assert (profile["counted_calls"], profile["no_call_steps"], profile["stamped_steps"]) == (2, 1, 1)
    lines = made_up.numbered(parts["lines"])
    every_pair = sum(end["end_wall"] - entry["wall"] for entry in lines if entry["kind"] == "enter"
                     for end in lines if end.get("enter_n") == entry["n"] and end.get("end_wall") is not None)
    assert every_pair == pytest.approx(2.0)  # the control: the sum over every pair would hold the no_call step


def test_the_processor_part_of_the_service_reads_the_thread_clock(tmp_path):
    parts = made_up.known_episode()
    profile = profile_of(tmp_path, parts)
    assert profile["tool_service_cpu_s"] == pytest.approx(1.125) and profile["tool_service_s"] == pytest.approx(1.75)
    parts["lines"][7]["end_thread_cpu"] = 0.75
    assert profile_of(tmp_path, parts, "control")["tool_service_cpu_s"] == pytest.approx(0.625)


def test_a_no_call_step_has_its_own_column_and_lands_in_the_rest(tmp_path):
    parts = made_up.known_episode()
    profile = profile_of(tmp_path, parts)
    assert profile["no_call_service_s"] == pytest.approx(0.25) and profile["rest_s"] == pytest.approx(2.75)
    parts["lines"] = [line for line in parts["lines"] if line.get("tool") != "no_call" and line.get("enter_n") != 9]
    control = profile_of(tmp_path, parts, "control")
    assert control["no_call_service_s"] == 0 and control["rest_s"] == pytest.approx(2.75)  # the rest never subtracted it


def test_the_server_starts_run_from_the_launcher_stamp_to_the_ready_reading(tmp_path):
    parts = made_up.known_episode()
    assert profile_of(tmp_path, parts)["server_starts_s"] == pytest.approx(1.25)
    parts["lines"].insert(5, {"kind": "ready", "server": "made_up_server", "s_wall": None, "s_mono": None, "s_thread_cpu": None,
                              "r_wall": wall(3.5), "r_mono": 3.5, "r_thread_cpu": 0.0})
    for line in parts["lines"]:
        if line.get("enter_n") is not None:
            line["enter_n"] += 1
    control = profile_of(tmp_path, parts, "control")
    assert control["server_starts_s"] == pytest.approx(1.25) and control["server_start_count"] == 3  # no launcher stamp, no start


def test_the_client_seconds_split_at_the_diagnostics_server(tmp_path):
    parts = made_up.known_episode()
    profile = profile_of(tmp_path, parts)
    assert (profile["env_tools_client_s"], profile["diagnostics_client_s"]) == (pytest.approx(2.5), pytest.approx(1.0))
    parts["trajectory"][1]["server"] = RIG
    parts["trajectory"].append({"step": 4, "server": MCP, "tool": None, "duration_ms": 9000.0})
    control = profile_of(tmp_path, parts, "control")
    assert (control["env_tools_client_s"], control["diagnostics_client_s"]) == (pytest.approx(3.5), 0)


def test_the_transport_and_hold_is_the_client_seconds_less_the_service(tmp_path):
    parts = made_up.known_episode()
    assert profile_of(tmp_path, parts)["transport_hold_s"] == pytest.approx(0.75)
    parts["lines"][7]["end_wall"] = wall(9.25)  # a longer service inside the same client call
    control = profile_of(tmp_path, parts, "control")
    assert control["tool_service_s"] == pytest.approx(2.0) and control["transport_hold_s"] == pytest.approx(0.5)


def test_the_rest_subtracts_each_part_once(tmp_path):
    parts = made_up.known_episode()
    parts["llm_calls"].append({"started_at": wall(4.5), "ended_at": wall(6.5), "seconds": 2.0})
    profile = profile_of(tmp_path, parts)
    assert profile["rest_s"] == pytest.approx(14.0 - 6.5 - 2.5 - 1.0 - 1.25)
    parts["trajectory"][0]["duration_ms"] = 3000.0
    assert profile_of(tmp_path, parts, "control")["rest_s"] == pytest.approx(1.75)


def test_a_counted_call_or_a_start_inside_a_model_span_is_counted(tmp_path):
    parts = made_up.known_episode()
    parts["lines"][6]["wall"], parts["lines"][7]["end_wall"] = wall(5.0), wall(5.5)  # the service inside the first model call
    parts["lines"][4]["r_wall"] = wall(4.25)  # the start of the diagnostics server reaches into it
    assert profile_of(tmp_path, parts)["lines_in_model_spans"] == 2
    parts = made_up.known_episode()
    parts["lines"][6]["wall"] = wall(7.0)  # an entry at the instant the first model call ends is no overlap
    assert profile_of(tmp_path, parts, "control")["lines_in_model_spans"] == 0


def test_a_rest_below_zero_is_flagged(tmp_path):
    parts = made_up.known_episode()
    parts["trajectory"][0]["duration_ms"] = 6000.0  # a client call that holds more than the time between the model calls
    profile = profile_of(tmp_path, parts)
    assert profile["rest_s"] == pytest.approx(-1.25) and profile["rest_negative"] == 1
    assert profile_of(tmp_path, made_up.known_episode(), "control")["rest_negative"] == 0


def test_a_server_start_outside_the_opening_is_counted(tmp_path):
    """A runner that starts a server process for every call starts a server again after the first counted call, and the start lies
    inside the client seconds of that call, so the rest subtracts it twice."""
    parts = made_up.known_episode()
    parts["lines"].append({"kind": "ready", "server": RIG, "s_wall": wall(15.25), "s_mono": 15.25, "s_thread_cpu": 0.0,
                           "r_wall": wall(15.3), "r_mono": 15.3, "r_thread_cpu": 0.0})
    parts["lines"].append({"kind": "ready", "server": "made_up_server", "s_wall": wall(15.3), "s_mono": 15.3, "s_thread_cpu": 0.0,
                           "r_wall": wall(15.35), "r_mono": 15.35, "r_thread_cpu": 0.0})
    assert profile_of(tmp_path, parts)["late_server_starts"] == 2
    assert profile_of(tmp_path, made_up.known_episode(), "control")["late_server_starts"] == 0


def test_the_charged_thinking_is_read_against_the_horizon_of_the_configuration(tmp_path):
    parts = made_up.known_episode()
    profile = profile_of(tmp_path, parts)
    assert (profile["thinking_s"], profile["thinking_cycles"], profile["horizon_cycles"]) == (pytest.approx(7.75), 1, 60)
    assert profile["thinking_share_of_horizon"] == pytest.approx(1 / 60) and profile["think_seconds_per_cycle"] == 10
    assert (profile["credited_starts_s"], profile["credited_calls_s"]) == (1.0, 1.25)  # the report, not the state's calls 1.0
    assert (profile["uncredited_calls_s"], profile["uncharged_after_last_s"]) == (0.5, 0.375)
    parts["state"]["config"]["horizon"] = 30
    assert profile_of(tmp_path, parts, "control")["thinking_share_of_horizon"] == pytest.approx(1 / 30)


def test_the_model_calls_give_their_seconds_tokens_and_output_rate(tmp_path):
    parts = made_up.known_episode()
    profile = profile_of(tmp_path, parts)
    assert (profile["model_calls"], profile["call_seconds_sum"], profile["call_seconds_median"]) == (3, pytest.approx(6.5), 2.5)
    assert (profile["input_tokens"], profile["output_tokens"]) == (600, 100)
    assert profile["output_tokens_per_call_second"] == pytest.approx(100 / 6.5)
    parts["llm_calls"].append({"started_at": wall(15.0), "ended_at": wall(15.2), "seconds": 0.2, "error": "made up"})
    control = profile_of(tmp_path, parts, "control")
    assert control["failed_model_calls"] == 1 and control["output_tokens_per_call_second"] == pytest.approx(100 / 6.5)


@pytest.mark.parametrize("change, reason", [
    (lambda parts: parts["row"].update(clock_skew="1"), stage_profile.SKEW),
    (lambda parts: parts["state"].pop("handover_wall"), "no handover_wall in the state file"),
    (lambda parts: parts["state"].pop("episode_ended_at"), "no episode_ended_at in the state file"),
    (lambda parts: parts["row"].update(wall_s=""), stage_profile.NO_TOTAL),
])
def test_an_episode_whose_stamps_cannot_be_read_is_left_out_by_name(tmp_path, change, reason):
    parts = made_up.known_episode()
    change(parts)
    profile = profile_of(tmp_path, parts)
    assert profile["left_out"] == reason and profile.get("rest_s") is None
    assert profile_of(tmp_path, made_up.known_episode(), "control")["left_out"] == ""


def test_a_torn_transcript_and_a_missing_state_are_left_out_by_name(tmp_path):
    run = made_up.write_run(tmp_path / "run", [(7, made_up.known_episode()), (8, made_up.known_episode())])
    with (run / "seed000007" / "transcript.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"n": 16, "kind": "ent')
    (run / "seed000008" / "episode.json").unlink()
    reasons = [stage_profile.episode_profile(row, folder, run_files.episode_files(folder) if folder else None)[0]["left_out"]
               for _name, row, folder in run_files.episode_entries(run)]
    assert reasons == [stage_profile.TORN, stage_profile.NO_STATE]  # the folder that the row names is read without its state file


def test_a_row_without_its_folder_and_a_folder_without_its_row_are_both_named(tmp_path):
    run = made_up.write_run(tmp_path / "run", [(7, made_up.known_episode()), (8, made_up.known_episode())])
    made_up.write_episode(run / "seed000009", made_up.known_episode())
    shutil.rmtree(run / "seed000007")
    found = {name: stage_profile.left_out(row, folder, run_files.episode_files(folder) if folder else None)
             for name, row, folder in run_files.episode_entries(run)}
    assert found == {"seed000007": stage_profile.NO_FOLDER, "seed000008": None, "seed000009": stage_profile.NO_ROW}


def test_an_episode_folder_with_its_row_but_no_state_file_is_left_out(tmp_path):
    parts = made_up.known_episode()
    files = {"state": None, "transcript": [], "transcript_torn": None, "llm_calls": [], "trajectory": []}
    assert stage_profile.left_out(parts["row"], tmp_path, files) == stage_profile.NO_STATE


@pytest.mark.parametrize("name, empty", [
    ("transcript.jsonl", ("tool_service_s", "server_starts_s", "counted_calls", "late_server_starts", "transport_hold_s", "rest_s",
                          "lines_in_model_spans", "rest_negative")),
    ("llm_calls.json", ("model_s", "model_calls", "call_seconds_sum", "rest_s", "lines_in_model_spans", "rest_negative")),
    ("trajectory.json", ("env_tools_client_s", "diagnostics_client_s", "transport_hold_s", "rest_s", "rest_negative")),
])
def test_a_missing_file_is_a_note_and_every_quantity_over_it_stays_empty(tmp_path, name, empty):
    """A folder without one of the three files is no episode of zero seconds: the note names the file, and every quantity that
    needs it, the differences and the checks over it among them, stays empty; the control is the whole folder."""
    run = made_up.write_run(tmp_path / "run", [(7, made_up.known_episode())])
    (run / "seed000007" / name).unlink()
    _name, row, folder = run_files.episode_entries(run)[0]
    profile = stage_profile.episode_profile(row, folder, run_files.episode_files(folder))[0]
    assert profile["left_out"] == "" and profile["note"] == f"no {name}"
    assert all(profile[column] is None for column in empty), [column for column in empty if profile[column] is not None]
    assert profile["creation_s"] == pytest.approx(2.0) and profile["end_s"] == pytest.approx(0.5)
    control = profile_of(tmp_path, made_up.known_episode(), "control")
    assert control["note"] == "" and all(control[column] is not None for column in empty)


def test_an_agent_entry_without_its_end_line_is_counted_and_adds_no_service(tmp_path):
    parts = made_up.known_episode()
    parts["lines"].append({"kind": "enter", "server": RIG, "tool": "read_sensors", "wall": wall(15.7), "mono": 15.7, "thread_cpu": 1.5})
    profile = profile_of(tmp_path, parts)
    assert profile["counted_calls_without_end"] == 1 and profile["counted_calls"] == 3 and profile["tool_service_s"] == pytest.approx(1.75)
    assert profile_of(tmp_path, made_up.known_episode(), "control")["counted_calls_without_end"] == 0


def test_the_reader_writes_nothing_into_the_run_folder(tmp_path):
    run = made_up.write_run(tmp_path / "run", [(7, made_up.known_episode())])
    before = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    for _name, row, folder in run_files.episode_entries(run):
        stage_profile.episode_profile(row, folder, run_files.episode_files(folder))
    run_files.endpoint_started(run)
    assert {path: path.read_bytes() for path in run.rglob("*") if path.is_file()} == before
    assert json.loads((run / "run.json").read_text(encoding="utf-8"))["agent"] == "causal"
