"""The harness rule of every agent: an episode its runner left open is ended by one finish with the runner's own
answer, an attempt the infrastructure broke is played once more under the same seed, two such attempts give a row of the class
harness_error, three in a row stop the run, the wall limit cuts an attempt, and the tools are listed before the first episode.
Real state files and the engine in process; the runner of the CausalDiagnosis agent is replaced by stubs that make real tool
calls, so no model and no endpoint is needed."""
import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from CausalDiagnosis.experiments.episode_runners import cache_flush
from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures
from CausalDiagnosis.evaluation.result_tables import leaderboard
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.cluster_runs import pending_grid_lines as pending_lines
from CausalDiagnosis.experiments.episode_runners import run_episodes
from CausalDiagnosis.experiments.episode_runners import wall_cut
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import HIDDEN, STATE_FILE, read_rows
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import RIG_SERVER
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config, load_world
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)

ANSWER = "The pump looks worn and no repair was made."
DOWN = "APIConnectionError: Connection error."


def _state(launchers):
    return Path(launchers[RIG_SERVER]).parent / STATE_FILE


def _causal(answers=(ANSWER,), errors=(None,), ends=("max_steps",), calls=((),), act=None, cut=None):
    """A stand-in of run_causal that plays attempt k with the k-th answer, error, end and model calls, calling act on the state
    file first, and returns the result the controller would. The default end is a runner's own limit, a port's step limit,
    since the causal controller has none of its own and the harness rule is the same for every runner. An attempt whose error
    is a cut at the wall limit records cut, the facts of what was open at the deadline, for its episode folder, as run_limited
    records them before it cancels a runner."""
    played = []

    def run_causal(backend, launchers, question, options, surface=None):
        k = len(played)
        played.append(question)
        if act is not None:
            act(_state(launchers))
        backend.calls.extend(dict(c) for c in calls[min(k, len(calls) - 1)])
        error, end = errors[min(k, len(errors) - 1)], ends[min(k, len(ends) - 1)]
        if cut is not None and (error or "").startswith(run_episodes.WALL_LIMIT):
            run_episodes.CUT_FACTS[str(_state(launchers).parent)] = {**cut, "at_wall": time.time()}
        result = SimpleNamespace(answer=answers[min(k, len(answers) - 1)], error=error, end=end, end_reason=end, trajectory=[],
                                 plans=[], surprises=[], revisions=0, fallbacks=0, mode_of_first_plan="", differential=[], model_calls=0, server=False, board={},
                                 reader_faults=[], replaced_characters=[])
        return result, error, 0.0
    run_causal.played = played
    return run_causal


# an endpoint on this machine where no service listens, so that no request of a test leaves the machine, and stand-ins of the
# two requests that the driver itself sends to the endpoint: the flush of the cache before an episode of a preset with the wall
# clock, and the probe of the model service at a cut, which a healthy endpoint answers with a token
LOCAL_ENDPOINT = "http://127.0.0.1:9/v1"
FLUSHED = {"flush_seconds": 0.0, "status": 200, "running": 0.0, "queued": 0.0, "first_request_cached_tokens": None}
# the facts of a cut at which the agent's model request was open, the one case in which the probe decides the end of the cut
REQUEST_OPEN = {"model_request_open": True, "failed_span_open": False, "tool_call_open": False}


def _main(tmp_path, monkeypatch, *args, seeds=1, healthy=True):
    monkeypatch.setenv("API_KEY", "stand-in-key")
    health = {"ok": healthy, "status": 200 if healthy else None, "error": None if healthy else "ConnectError: All connection attempts failed",
              "seconds": 0.0}
    monkeypatch.setattr(failures, "endpoint_health", lambda *a, **k: health)
    monkeypatch.setattr(cache_flush, "flush_before_episode", lambda base_url, api_key=None: dict(FLUSHED))
    probe = {**health, "cut_probes": 1 if healthy else 0}
    monkeypatch.setattr(wall_cut, "cut_probe", lambda *a, **k: dict(probe))
    return run_episodes.main(["--config", "static", "--agent", "causal", "--seeds", str(seeds), "--no-grade", "--base-url", LOCAL_ENDPOINT,
                              "--out", str(tmp_path / "runs"), *args])


def _finishes(run_dir, name="seed000000"):
    log = read_json(run_dir / name / HIDDEN / "log.json")
    return [e for e in log if e["event"] == "finish"], log


def test_an_episode_its_runner_left_open_is_finished_by_the_harness_with_the_runners_answer(tmp_path, monkeypatch):
    """The runner returned on its own limit without a finish, so the driver ends the episode with one finish whose
    summary is the runner's answer, the log has its end at reason finish, and the row and harness.json say who finished."""
    monkeypatch.setattr(run_episodes.agent_runs, "run_causal", _causal())
    run_dir = _main(tmp_path, monkeypatch)
    row = read_rows(run_dir)[0]
    finishes, log = _finishes(run_dir)
    assert [f["summary"] for f in finishes] == [ANSWER] and log[-1]["event"] == "end" and log[-1]["reason"] == "finish"
    assert (row["finish_by"], row["runner_end"], row["error_class"], row["replays"], row["wall_limit"]) == ("harness", "own_limit", "", "0", "0")
    harness = read_json(run_dir / "seed000000" / run_episodes.HARNESS_FILE)
    assert harness["attempts"] == 1 and harness["standing"]["finish_by"] == "harness" and harness["set_aside"] is None
    assert row["pass"] != ""  # the episode is ended, so the verifier scores it


def test_an_episode_the_agent_finished_gets_no_second_finish(tmp_path, monkeypatch):
    def finish(state_path):
        episode.call(state_path, "finish", {"summary": "done"}, {"type": "finish", "summary": "done"})
    monkeypatch.setattr(run_episodes.agent_runs, "run_causal", _causal(answers=("done",), ends=("finish",), act=finish))
    run_dir = _main(tmp_path, monkeypatch)
    finishes, _log = _finishes(run_dir)
    assert len(finishes) == 1 and read_rows(run_dir)[0]["finish_by"] == "agent" and read_rows(run_dir)[0]["runner_end"] == "finish"


def test_an_attempt_the_infrastructure_broke_is_played_again_under_the_same_seed_and_the_row_counts_one_replay(tmp_path, monkeypatch):
    stub = _causal(errors=(DOWN, None))
    monkeypatch.setattr(run_episodes.agent_runs, "run_causal", stub)
    run_dir = _main(tmp_path, monkeypatch)
    row = read_rows(run_dir)[0]
    assert len(stub.played) == 2 and stub.played[0] == stub.played[1]  # one briefing, one seed
    assert row["replays"] == "1" and row["error_class"] == "" and row["finish_by"] == "harness"
    assert (run_dir / "seed000000" / run_episodes.ATTEMPT_FOLDER / STATE_FILE).exists()
    aside = read_json(run_dir / "seed000000" / run_episodes.HARNESS_FILE)["set_aside"]
    assert aside["class"] == failures.INFRASTRUCTURE and aside["kinds"] == [failures.CONNECTION] and aside["folder"] == run_episodes.ATTEMPT_FOLDER


def test_two_such_attempts_give_a_harness_error_row_with_empty_scores_and_its_start_combination(tmp_path, monkeypatch):
    """The episode is listed by its id with empty scores, and it keeps its start combination, so its cluster."""
    monkeypatch.setattr(run_episodes.agent_runs, "run_causal", _causal(errors=(DOWN,)))
    run_dir = _main(tmp_path, monkeypatch)
    row = read_rows(run_dir)[0]
    assert row["error_class"] == failures.HARNESS_ERROR and row["replays"] == "1" and row["pass"] == "" and row["error"].startswith("APIConnectionError")
    assert json.loads(row["block"]) and row["n_faulty"] != ""


def test_an_episode_whose_start_never_ran_keeps_the_start_combination_of_its_seed(tmp_path, monkeypatch):
    """The same where no start ran: both starts of the episode raised, so the row of the class harness_error takes the start
    combination that the reset under its seed draws, the one a played episode of the seed starts from."""
    monkeypatch.setattr(run_episodes.agent_runs, "run_causal", _causal())
    played = read_rows(_main(tmp_path, monkeypatch))[0]
    real = run_episodes.EpisodeRunner.start

    def start(self, ep_dir, seed, episode_config):
        if ep_dir.name != run_episodes.SERVER_CHECK_FOLDER:
            raise OSError("the disk is full")
        return real(self, ep_dir, seed, episode_config)
    monkeypatch.setattr(run_episodes.EpisodeRunner, "start", start)
    row = read_rows(_main(tmp_path, monkeypatch))[0]
    assert row["error_class"] == failures.HARNESS_ERROR and row["replays"] == "1" and row["pass"] == ""
    assert (row["block"], row["n_faulty"]) == (played["block"], played["n_faulty"]) and json.loads(row["block"])


def test_a_failed_model_call_inside_a_run_that_returned_sets_the_attempt_aside(tmp_path, monkeypatch):
    """The controller went on after a call that failed on the endpoint, so the attempt is the infrastructure's."""
    failed = [{"prompt": "p", "error": "InternalServerError: Error code: 500", "ended_at": time.time()}]
    monkeypatch.setattr(run_episodes.agent_runs, "run_causal", _causal(calls=(failed, ())))
    run_dir = _main(tmp_path, monkeypatch)
    row = read_rows(run_dir)[0]
    assert row["replays"] == "1" and row["error_class"] == "" and row["failed_calls"] == "0"


def test_a_start_that_raises_is_an_attempt_of_the_infrastructure_and_the_run_goes_on(tmp_path, monkeypatch):
    """The first start after the server check raises, the second plays."""
    real, calls = run_episodes.EpisodeRunner.start, []

    def start(self, ep_dir, seed, episode_config):
        calls.append(ep_dir.name)
        if ep_dir.name != run_episodes.SERVER_CHECK_FOLDER and calls.count(ep_dir.name) == 1:
            raise OSError("the disk is full")
        return real(self, ep_dir, seed, episode_config)
    monkeypatch.setattr(run_episodes.EpisodeRunner, "start", start)
    monkeypatch.setattr(run_episodes.agent_runs, "run_causal", _causal())
    run_dir = _main(tmp_path, monkeypatch)
    row = read_rows(run_dir)[0]
    assert row["replays"] == "1" and row["error_class"] == "" and (run_dir / "run.json").exists()
    assert read_json(run_dir / "seed000000" / run_episodes.HARNESS_FILE)["set_aside"]["kinds"] == [failures.HARNESS]


def test_three_episodes_in_a_row_stop_the_run_with_status_75_without_run_json_and_its_line_stays_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(run_episodes.agent_runs, "run_causal", _causal(errors=(DOWN,)))
    with pytest.raises(SystemExit) as stopped:
        _main(tmp_path, monkeypatch, seeds=4)
    assert stopped.value.code == 75
    run_dir = next((tmp_path / "runs").iterdir())
    assert not (run_dir / "run.json").exists() and len(read_rows(run_dir)) == 3
    assert any(e["what"] == failures.STOP_EVENT for e in ledger.load(run_dir)["events"])
    assert list(pending_lines.finished_runs([tmp_path / "runs"])) == []
    assert leaderboard.progress_state(run_dir) == "stopped"


def test_cut_replies_counts_the_calls_that_ended_at_the_token_cap():
    """For the records of both recorders, which keep the finish reason of the reply."""
    calls = [{"finish_reason": "length"}, {"finish_reason": "stop"}, {"finish_reason": "length", "error": "x"}, {}]
    assert run_episodes.cut_replies(calls) == 2
    assert run_episodes.failed_calls(calls, [{"error": "ToolTransportError: no answer"}, {"error": "the argument is refused"}]) == 2


def test_refused_diagnoses_are_counted_by_reason():
    """On the log of the engine, the same on both surfaces."""
    log = [{"event": "refused", "action": {"type": "submit_diagnosis"}, "reason": "already diagnosed"},
           {"event": "refused", "action": {"type": "submit_diagnosis"}, "reason": "already diagnosed"},
           {"event": "refused", "action": {"type": "read_sensors"}, "reason": "unknown group"}]
    assert json.loads(run_episodes.diagnosis_refusals(log)) == {"already diagnosed": 2} and run_episodes.diagnosis_refusals(log[2:]) == ""


def test_run_json_holds_the_same_setting_keys_for_the_three_agents(tmp_path, monkeypatch):
    """For the controller here: the blocks decoding, transport, limits and harness."""
    monkeypatch.setattr(run_episodes.agent_runs, "run_causal", _causal())
    meta = read_json(_main(tmp_path, monkeypatch) / "run.json")
    assert {"decoding", "transport", "limits", "harness"} <= set(meta)
    assert meta["harness"]["episode_wall_limit_s"] == run_episodes.EPISODE_WALL_LIMIT_S == 1800 and meta["harness"]["stop_after_failures"] == 3
    assert meta["transport"]["sessions"] and meta["decoding"]["temperature"] is not None


def test_the_tools_are_listed_on_a_scratch_folder_that_is_gone_before_the_first_state_file_exists(tmp_path, monkeypatch):
    seen = []
    real = run_episodes.play_one_episode.check_servers

    def check(launchers, config=None):
        run_dir = Path(next(iter(launchers.values()))).parent.parent
        seen.append(sorted(p.name for p in run_dir.iterdir()))
        return real(launchers, config)
    monkeypatch.setattr(run_episodes.play_one_episode, "check_servers", check)
    monkeypatch.setattr(run_episodes.agent_runs, "run_causal", _causal())
    run_dir = _main(tmp_path, monkeypatch)
    # once, while the run folder holds its ledger and the scratch folder and no episode folder yet
    assert len(seen) == 1 and run_episodes.SERVER_CHECK_FOLDER in seen[0] and not [n for n in seen[0] if n.startswith("seed")]
    assert not (run_dir / run_episodes.SERVER_CHECK_FOLDER).exists() and (run_dir / "seed000000" / STATE_FILE).exists()


def test_the_static_prompt_names_no_dataset_and_no_data_file():
    prompt = run_episodes.static_prompt("Q", {"features": {"PS1": 1.0}, "failure_modes": []}, ["cooler"])
    assert "Facts of the current operating cycle of the asset (feature: value):" in prompt
    assert "IndustryAssetEQA" not in prompt and "facts file" not in prompt and ".jsonl" not in prompt


def test_the_wall_limit_cuts_a_runner_and_a_cancellation_the_run_did_not_send_is_an_interrupt():
    """The cut itself: a runner past the limit is cancelled, a controller that returns its partial result keeps it."""
    async def hangs():
        await asyncio.sleep(30)

    result, error, seconds = run_episodes.run_limited(hangs, limit_s=0.2)
    assert result is None and error == "WallLimit: the episode passed 0.2 s" and seconds < 10

    async def returns_partial():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            return SimpleNamespace(answer="partial", error=None, end="cancelled")

    result, error, _ = run_episodes.run_limited(returns_partial, limit_s=0.2)
    assert result.answer == "partial" and error.startswith("WallLimit")

    async def cancelled_by_someone_else():
        return SimpleNamespace(answer="", error=None, end="cancelled")

    with pytest.raises(KeyboardInterrupt):
        run_episodes.run_limited(cancelled_by_someone_else, limit_s=5)


@pytest.mark.parametrize("source", ["the cut at the wall limit", "the end of the last model call"])
@pytest.mark.parametrize("accrued, ended", [(1800.0, True), (95.0, False)])
def test_a_harness_finish_plays_the_accrued_thinking_through_the_no_call_entry_and_leaves_no_call_the_agent_never_made(
        tmp_path, accrued, ended, source):
    """Before the driver writes a finish on a row with a thinking clock, at a cut at the wall limit or after the runner ended on
    its own, the thinking accrued up to the finish's stamp is played through the episode's no_call entry, stamped at that
    moment. On the deterioration row 1,800 s of it end the episode there, which is the record: no finish follows, and the log
    holds no refusal and no tool call the agent never made. The control, 95 s of it, leaves the episode open, and the driver's
    finish ends it as without the play."""
    name, config = load_config("deterioration")
    facts_path, cycles, blocks = load_world(None, config.asset_id)
    runner = run_episodes.EpisodeRunner(config, name, "scripted", (RIG_SERVER,), cycles, blocks, facts_path, "finish", {"judge": None})
    ep_dir = tmp_path / "seed000011"
    ep_dir.mkdir()
    stamp_wall, stamp_mono = time.time(), time.monotonic()
    with episode.recorded_clock(stamp_wall - accrued, stamp_mono - accrued):
        episode.start(ep_dir / STATE_FILE, 11, config, facts_path)
    episode.hand_over(ep_dir / STATE_FILE, wall=stamp_wall - accrued, mono=stamp_mono - accrued)
    finish_by, finish = runner.finish_episode(ep_dir, ANSWER, stamp_wall, source)
    state = episode.load_state(ep_dir / STATE_FILE)
    env = episode.replay(state)
    events = [e["event"] for e in env.log]
    assert env.done and "refused" not in events and events.count("no_call") == 1, events
    if ended:
        assert (finish_by, finish) == ("", None) and "finish" not in events and run_episodes.tool_counts(state)["tool_calls"] == 0
    else:
        assert finish_by == "harness" and finish["refusal"] is None and events[-2:] == ["finish", "end"]
        assert run_episodes.tool_counts(state) == {"tool_calls": 1, "refusals": 0}


def test_an_episode_past_the_wall_limit_is_cut_ended_by_the_harness_and_scored(tmp_path, monkeypatch):
    """A model request was open at the cut and the model service answers the probe with a token, so the cut is the agent's:
    the episode is ended where it stands and scored."""
    monkeypatch.setattr(run_episodes.agent_runs, "run_causal", _causal(errors=("WallLimit: the episode passed 1800 s",), ends=("cancelled",),
                                                            cut=REQUEST_OPEN))
    run_dir = _main(tmp_path, monkeypatch)
    row = read_rows(run_dir)[0]
    assert (row["wall_limit"], row["finish_by"], row["runner_end"], row["replays"]) == ("1", "harness", "cancelled", "0")
    assert row["error_class"] == failures.AGENT and row["error"].startswith("WallLimit") and row["pass"] != ""


def test_no_option_of_the_command_line_changes_the_wall_limit():
    options = [o for action in run_episodes.build_parser()._actions for o in action.option_strings]
    assert not [o for o in options if "wall" in o or "limit" in o]


def test_a_stopped_run_and_a_stalled_run_do_not_read_still_running(tmp_path):
    for name in ("stopped", "stalled", "live"):
        (tmp_path / name).mkdir()
        (tmp_path / name / ledger.LEDGER).write_text(json.dumps({"status": "running", "events": []}), encoding="utf-8")
    ledger.append_event(tmp_path / "stopped", "run_episodes.py", failures.STOP_EVENT)
    old = time.time() - 3 * run_episodes.EPISODE_WALL_LIMIT_S
    os.utime(tmp_path / "stalled" / ledger.LEDGER, (old, old))
    assert [leaderboard.progress_state(tmp_path / n) for n in ("stopped", "stalled", "live")] == ["stopped", "stalled", "still running"]


def test_open_orders_settle_at_the_harness_finish_on_a_row_with_a_clock(tmp_path, monkeypatch):
    """The runner wrote the start as its diagnosis, ordered a repair of every faulty component, those a repair depends on
    first, and returned without a finish, so the harness finish settles the open orders and the asset is recovered, as after a
    finish of the agent."""
    def diagnose_and_order(state_path):
        env = episode.replay(episode.load_state(state_path))
        start = dict(zip(env.config.components, env.initial))
        episode.call(state_path, "submit_diagnosis", {"diagnosis": start, "evidence_ids": []},
                     {"type": "submit_diagnosis", "diagnosis": start, "evidence_ids": []})
        faulty = [c for c in env.config.components if start[c] != env.config.healthy[c]]
        for component in sorted(faulty, key=lambda c: len(env.config.depends_on(c))):
            order = {"component": component, "prediction_id": "", "evidence_ids": [], "mode": "standard"}
            episode.call(state_path, "create_work_order", order, {"type": "create_work_order", **order})
    monkeypatch.setattr(run_episodes.agent_runs, "run_causal", _causal(act=diagnose_and_order))
    run_dir = _main(tmp_path, monkeypatch)
    _, log = _finishes(run_dir)
    verdict = read_json(run_dir / "seed000000" / HIDDEN / "verdict.json")
    assert read_rows(run_dir)[0]["finish_by"] == "harness" and log[-1]["reason"] == "finish"
    assert [e for e in log if e["event"] == "refused"] == [] and [e for e in log if e["event"] == "work_order_created"]
    assert verdict["recovered"]


def test_an_episode_the_environment_ended_gets_no_harness_finish(tmp_path, monkeypatch):
    """The runner read the clock until the environment ended the episode, so the driver adds no finish and the row names
    no finisher."""
    def read_until_the_end(state_path):
        while not episode.replay(episode.load_state(state_path)).done:
            episode.call(state_path, "get_current_time", {}, {"type": "get_current_time"})
    monkeypatch.setattr(run_episodes.agent_runs, "run_causal", _causal(ends=("episode_over",), act=read_until_the_end))
    run_dir = _main(tmp_path, monkeypatch)
    finishes, log = _finishes(run_dir)
    row = read_rows(run_dir)[0]
    assert finishes == [] and log[-1]["event"] == "end" and log[-1]["reason"] in ("horizon", "max_steps")
    assert (row["finish_by"], row["runner_end"]) == ("", "episode_over")
    assert read_json(run_dir / "seed000000" / run_episodes.HARNESS_FILE)["standing"]["finish"] is None


def test_the_harness_finish_on_the_assetopsbench_surface_runs_through_the_metering_layer_and_its_documents_are_exported(tmp_path, monkeypatch):
    """On the AssetOpsBench servers the harness finish is a call of the task server through the metering layer, so the log
    ends at the finish with the runner's answer and the documents of the episode are exported beside it."""
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers.episode_documents import EXPORT_NAME
    monkeypatch.setattr(run_episodes.agent_runs, "run_causal", _causal())
    run_dir = _main(tmp_path, monkeypatch, "--servers", "aob")
    finishes, log = _finishes(run_dir)
    assert [f["summary"] for f in finishes] == [ANSWER] and log[-1]["reason"] == "finish"
    assert read_rows(run_dir)[0]["finish_by"] == "harness" and (run_dir / "seed000000" / EXPORT_NAME).exists()


def _clock_read(made, model_call_after_s):
    """An act of a stand-in runner: one read of the clock 100 s ago, and a model call that ended model_call_after_s after it."""
    def read_the_clock(state_path):
        called = time.time() - 100.0
        episode.call(state_path, "get_current_time", {}, {"type": "get_current_time"}, now=called, monotonic=time.monotonic() - 100.0)
        made.append({"prompt": "p", "ended_at": called + model_call_after_s})
    return read_the_clock


def _handed_over_before(monkeypatch, seconds):
    """The driver hands the task over seconds before now, so that the readings a stand-in runner stamps in the past lie
    after the handover, the origin of the thinking clock, as the calls of a played episode do."""
    hand_over = episode.hand_over
    monkeypatch.setattr(episode, "hand_over", lambda path, wall=None, mono=None: hand_over(path, wall=time.time() - seconds,
                                                                                         mono=time.monotonic() - seconds))


def test_the_harness_finish_charges_the_seconds_up_to_the_last_model_call_and_not_the_teardown(tmp_path, monkeypatch):
    """In the clock mode that charges the wall seconds between calls: the last model call ended 25 s after the runner's last
    tool call, so the no_call step played before the harness finish at its stamp is charged those 25 s, the finish nothing more,
    and the 75 s that passed after it are not charged."""
    made = []
    monkeypatch.setattr(run_episodes.agent_runs, "run_causal", _causal(calls=(made,), act=_clock_read(made, 25.0)))
    _handed_over_before(monkeypatch, 200.0)
    run_dir = _main(tmp_path, monkeypatch, "--clock-mode", "generation_time")
    state = episode.load_state(run_dir / "seed000000" / STATE_FILE)
    assert [a["type"] for a in state["actions"][-2:]] == ["no_call", "finish"]
    assert state["think_seconds"][-2] == pytest.approx(25.0) and state["think_seconds"][-1] == 0.0


def test_a_harness_finish_after_a_later_tool_call_charges_nothing(tmp_path, monkeypatch):
    """The last model call ended 5 s before the runner's last tool call, so the stamp lies before the call before it and
    the harness finish is charged nothing."""
    made = []
    monkeypatch.setattr(run_episodes.agent_runs, "run_causal", _causal(calls=(made,), act=_clock_read(made, -5.0)))
    run_dir = _main(tmp_path, monkeypatch, "--clock-mode", "generation_time")
    state = episode.load_state(run_dir / "seed000000" / STATE_FILE)
    assert state["actions"][-1]["type"] == "finish" and state["think_seconds"][-1] == 0.0


@pytest.mark.parametrize("diagnostics", [True, False], ids=["with_mcp", "control_without_mcp"])
def test_the_settings_blocks_have_the_same_keys_for_the_three_agents_and_are_the_values_of_their_accessors(monkeypatch, diagnostics):
    """The blocks decoding and transport of each agent are what the accessor of its route
    returns on the servers of the run, and the three agents write the same keys in both. The ports' max_tokens is the
    room of each request on a run with the CausalDiagnosis MCP and the paper's 500 without it."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.controller import request_settings as controller_settings
    from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.sub_agent_runtime import request_settings as port_settings
    from CausalDiagnosis.frameworks.AssetOpsBench.agents.Agent_As_Tool_Paper.text_completion_route import REPLY_ROOM
    monkeypatch.setenv("API_KEY", "stand-in-key")
    backend = run_episodes.RecordingBackend(run_episodes.model_backends.router_backend(run_episodes.DEFAULT_MODEL))
    runner = SimpleNamespace(temperature=None, servers=run_episodes.registered_servers(None, diagnostics), backend=backend, causal_options={})
    blocks = {agent: run_episodes.agent_settings(SimpleNamespace(agent=agent), runner) for agent in ("causal", "agent_as_tool", "plan_execute")}
    ports = port_settings(servers=runner.servers)
    expected = {"causal": controller_settings(backend), "agent_as_tool": ports, "plan_execute": ports}
    for agent, block in blocks.items():
        assert (block["decoding"], block["transport"]) == (expected[agent]["decoding"], expected[agent]["transport"]), agent
        assert block["limits"]
    assert len({tuple(sorted(b["decoding"])) for b in blocks.values()}) == 1
    assert len({tuple(sorted(b["transport"])) for b in blocks.values()}) == 1
    assert ports["decoding"]["max_tokens"] == (REPLY_ROOM if diagnostics else 500)


def test_a_cut_with_a_dead_endpoint_is_replayed(tmp_path, monkeypatch):
    """The wall limit cut the attempt while a model request was open, and the model service generated no token for the probe of
    the cut, so the cut is the infrastructure's and the episode is played once more under the same seed."""
    stub = _causal(errors=("WallLimit: the episode passed 1800 s", None), ends=("cancelled", "max_steps"), cut=REQUEST_OPEN)
    monkeypatch.setattr(run_episodes.agent_runs, "run_causal", stub)
    run_dir = _main(tmp_path, monkeypatch, healthy=False)
    row = read_rows(run_dir)[0]
    aside = read_json(run_dir / "seed000000" / run_episodes.HARNESS_FILE)["set_aside"]
    assert len(stub.played) == 2 and (row["replays"], row["wall_limit"], row["error_class"]) == ("1", "0", "")
    assert aside["class"] == failures.INFRASTRUCTURE and aside["wall_limit"] and aside["cut_end"] == failures.SERVICE
    assert aside["cut_probe"]["ok"] is False and aside["health"] is None  # a cut is classed by the end its facts and the probe give, no health check


def test_a_reference_policy_plays_one_episode_on_one_set_of_sessions_and_its_log_equals_the_shadow(tmp_path, monkeypatch):
    """The oracle plays its episode through one set of sessions, after the one of the tool listing, and the log of the
    action server equals that of the shadow environment."""
    opened, real = [], run_episodes.reference_policy_episodes.open_sessions

    def counted(servers):
        opened.append(sorted(servers))
        return real(servers)
    monkeypatch.setattr(run_episodes.reference_policy_episodes, "open_sessions", counted)
    run_dir = run_episodes.main(["--config", "static", "--agent", "oracle", "--seeds", "1", "--no-grade", "--out", str(tmp_path / "runs")])
    assert read_rows(run_dir)[0]["replay_matches"] == "True" and len(opened) == 2 and opened[1] == [RIG_SERVER]


def test_the_episode_report_and_the_run_summary_state_what_the_harness_did(tmp_path, monkeypatch):
    """The section Harness of the episode report and the line of counts of the run summary: one attempt set aside, the second
    finished by the harness."""
    monkeypatch.setattr(run_episodes.agent_runs, "run_causal", _causal(errors=(DOWN, None)))
    run_dir = _main(tmp_path, monkeypatch)
    report = (run_dir / "seed000000" / "report.md").read_text(encoding="utf-8")
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "## Harness" in report and "Attempts 2," in report and "finished by harness" in report
    assert f"set aside in {run_episodes.ATTEMPT_FOLDER}/ was of the class {failures.INFRASTRUCTURE}" in report
    assert ("Harness: episode folders holding harness.json 1, rows finished by the harness 1, rows with a replay 1, "
            "rows of the class harness_error 0, cut replies 0, failed calls 0.") in summary



def test_the_cached_tokens_of_an_episode_sum_the_counts_of_its_answered_calls_and_a_missing_count_is_zero_only_under_the_cache_report():
    """The cell cached_tokens sums the counts that the records of the answered calls hold, and a call without a reply adds nothing.
    An answered call without a count makes the cell empty, unless the run's server reports its cached tokens with SGLang's rule,
    under which a reply that reused nothing carries no count, so it reads as zero there. The control: the same calls with the
    count present give the sum either way."""
    from CausalDiagnosis.experiments.episode_runners.run_episodes import harness_rule

    counted = [{"reply": "a", "cached_tokens": 40}, {"reply": "b", "cached_tokens": 0}, {"error": "RuntimeError: down", "cached_tokens": None}]
    missing = [*counted, {"reply": "c", "cached_tokens": None}]
    assert [harness_rule.cached_tokens(counted, flag) for flag in (None, False, True)] == [40, 40, 40]
    assert [harness_rule.cached_tokens(missing, flag) for flag in (None, False, True)] == ["", "", 40]
    assert harness_rule.cached_tokens([], None) == 0
