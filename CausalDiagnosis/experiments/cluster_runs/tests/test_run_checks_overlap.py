"""The overlap check of cluster_run_checks.py: an episode of a run on the wall clock plays alone on a clean cache, so a run
of the same grid folder alive between the flush before the episode and its end is named with both intervals, and so is an episode
whose flush is missing, did not answer 200 or left cached tokens in the first request; the check then exits 1. A run before the
flush or after the end passes, and an episode without a recorded interval is named. The flush is asked of a run whose agent
reaches a model endpoint alone, since the driver records none for an agent without a model, and the run folders hold run.json in
the public form that the driver writes, with serving.agent and no endpoint URL. A run is placed by its attempts: a run that a later
job took up again is alive where each job played it, and a run that its job left without run.json up to the latest second it
recorded. The episodes of a run without run.json are judged by the clock that their states record, and an episode of an earlier
attempt of a taken-up run also against the runs of the grid folder where that attempt played. The runs are built with the
driver's writers."""
import json
import time
from datetime import datetime

from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.experiments.run_verification import public_record
from CausalDiagnosis.experiments.cluster_runs import cluster_run_checks as run_checks, run_health_checks
from CausalDiagnosis.experiments.episode_runners import run_episodes
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

CLEAN = {"flush_seconds": 0.2, "status": 200, "running": 0.0, "queued": 0.0, "first_request_cached_tokens": 0, "cache_report": True}
ENDPOINT = "http://node7:30123/v1"  # the model endpoint of a model agent, which the public run.json writes as null
MODEL = "granite"
CLEAN_SUMMARY = "OVERLAP_EPISODES 0 overlaps, 0 episodes without a recorded interval, 0 episodes without a clean flush"
# the start fields of the ledger of a policy run, and its command line
FIELDS = {"label": "real_time_oracle", "agent": "oracle", "agent_label": "oracle", "model": None, "judge": None, "data": {},
          "configuration": {"config_name": "real_time"}, "seeds": None}
ARGV = ["--config", "real_time", "--agent", "oracle"]


def _second(seconds):
    """A time as run.json and the ledger write it, ISO in local time to the second."""
    return datetime.fromtimestamp(seconds).isoformat(timespec="seconds")


def _state(ep_dir, **fields):
    """An episode folder with the state file that the driver writes (episode.save_state), the given fields among its own."""
    ep_dir.mkdir(parents=True)
    episode.save_state(ep_dir / "episode.json", {"format": episode.STATE_FORMAT, **fields})


def _started(run_dir, started):
    """A run folder with the ledger that the driver writes before its first episode (ledger.start)."""
    run_dir.mkdir(parents=True)
    ledger.start(run_dir, {"started": _second(started), "code": {}}, ARGV, fields=FIELDS)


def _killed(grid, name, started, last):
    """A run that its job left without run.json, killed or cut: the ledger of its start and one episode whose last call came at
    last, the latest second the run recorded."""
    run = grid / name
    _started(run, started)
    _state(run / "seed000000", episode_started_at=started + 1, last_call_wall_time=last)
    return run


def _run(grid, name, started, ended, clock_mode="declared", episodes=(), endpoint=ENDPOINT):
    """A run folder with run.json in the public form that run_episodes.write_run writes (times as the driver writes them, no
    endpoint URL, serving.agent the record of a model agent's endpoint and None for an agent without a model) and one episode
    folder per (start, end) in epoch seconds, with the flush record that the driver writes for an agent with a model endpoint
    alone."""
    run = grid / name
    run.mkdir(parents=True)
    iso = (lambda t: datetime.fromtimestamp(t).isoformat(timespec="seconds") if t is not None else None)
    agent = ledger.serving_record({"name": MODEL, "endpoint": endpoint, "endpoint_file": None, "endpoint_models": None}) if endpoint else None
    meta = {"clock_mode": clock_mode, "started": iso(started), "ended": iso(ended), "endpoint": endpoint,
            "serving": {"agent": agent, "judge": None}}
    (run / "run.json").write_text(json.dumps(public_record.public_value(meta)), encoding="utf-8")
    for k, (start, end) in enumerate(episodes):
        flush = {"cache_flush": {**CLEAN, "started_at": start - 2}} if endpoint else {}
        _state(run / f"seed{k:06d}", episode_started_at=start, episode_ended_at=end, **flush)
    return run


def _flush(grid, **record):
    """Replaces the flush record of the wall-clock episode; None removes it."""
    path = grid / "wall" / "seed000000" / "episode.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state["cache_flush"] = None if record.get("cache_flush", 1) is None else {**CLEAN, "started_at": state["episode_started_at"] - 2, **record}
    path.write_text(json.dumps(state), encoding="utf-8")


def test_a_run_alive_during_a_wall_clock_episode_or_an_unclean_flush_is_named_and_a_clean_lone_episode_passes(tmp_path, capsys):
    grid = tmp_path / "grid_7"
    t0 = 1_800_000_000
    _run(grid, "wall", t0, t0 + 100, run_health_checks.WALL_CLOCK, episodes=[(t0 + 10, t0 + 50)])
    _run(grid, "before", t0 - 100, t0 + 5)  # ended before the flush at t0 + 8
    _run(grid, "after", t0 + 60, t0 + 90)  # started after the episode's end
    assert run_checks.main(["overlap", str(grid)]) == 0
    clean = "OVERLAP_EPISODES 0 overlaps, 0 episodes without a recorded interval, 0 episodes without a clean flush"
    assert capsys.readouterr().out.splitlines()[-1] == clean
    for record, reason in (({"cache_flush": None}, "no record of the flush of the model server's cache before it"),
                           ({"status": 400}, "the flush of the model server's cache answered 400, not 200"),
                           ({"first_request_cached_tokens": 12}, "its first model request reused 12 cached tokens, not 0"),
                           ({"first_request_cached_tokens": None}, "its first model request reused None cached tokens, not 0"),
                           # a 0 from a server that reports no count, or whose rule is not known, shows no clean cache
                           ({"cache_report": False}, "the model server does not report its cached tokens (cache_report False), so no reply shows the cache clean"),
                           ({"cache_report": None}, "the model server does not report its cached tokens (cache_report None), so no reply shows the cache clean")):
        _flush(grid, **record)
        assert run_checks.main(["overlap", str(grid)]) == 1
        assert f"OVERLAP_FLUSH wall/seed000000: {reason}" in capsys.readouterr().out, record
    _flush(grid)
    _killed(grid, "filler", t0 + 9, t0 + 60)  # left by its job, alive from inside the flush past the episode's end
    assert run_checks.main(["overlap", str(grid)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"OVERLAP wall/seed000000 [{t0 + 8}, {t0 + 50}] overlaps filler [{t0 + 9}, {t0 + 60}]"
    assert out[-1] == "OVERLAP_EPISODES 1 overlaps, 0 episodes without a recorded interval, 0 episodes without a clean flush"
    state = grid / "wall" / "seed000000" / "episode.json"
    state.write_text(json.dumps({"think_seconds": [], "cache_flush": {**CLEAN}}), encoding="utf-8")
    assert run_checks.main(["overlap", str(grid)]) == 1
    assert "OVERLAP_UNRECORDED wall/seed000000: no recorded start or end of the episode" in capsys.readouterr().out


def test_a_run_without_a_model_endpoint_is_held_to_the_overlap_alone_and_a_model_run_to_its_flush_as_well(tmp_path, capsys):
    """An agent without a model endpoint, a baseline policy, has no model server whose cache could be flushed, so the driver writes
    no flush record for its episodes (run_episodes.Runner.flush_cache) and the flush check passes them, while a run alive in such an
    episode is still named. The control: an episode of a model agent without a flush record is named."""
    t0 = 1_800_000_000
    clean = "OVERLAP_EPISODES 0 overlaps, 0 episodes without a recorded interval, 0 episodes without a clean flush"
    grid = tmp_path / "grid_8"
    _run(grid, "oracle", t0, t0 + 100, run_health_checks.WALL_CLOCK, episodes=[(t0 + 10, t0 + 50)], endpoint=None)
    assert "cache_flush" not in json.loads((grid / "oracle" / "seed000000" / "episode.json").read_text(encoding="utf-8"))
    assert run_checks.main(["overlap", str(grid)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == clean
    _killed(grid, "filler", t0 + 20, t0 + 60)
    assert run_checks.main(["overlap", str(grid)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"OVERLAP oracle/seed000000 [{t0 + 10}, {t0 + 50}] overlaps filler [{t0 + 20}, {t0 + 60}]"
    assert not any(line.startswith("OVERLAP_FLUSH") for line in out)
    model = tmp_path / "grid_9"
    _run(model, "wall", t0, t0 + 100, run_health_checks.WALL_CLOCK, episodes=[(t0 + 10, t0 + 50)])
    _flush(model, cache_flush=None)
    assert run_checks.main(["overlap", str(model)]) == 1
    assert "OVERLAP_FLUSH wall/seed000000: no record of the flush of the model server's cache before it" in capsys.readouterr().out


def test_the_flush_check_reads_the_run_json_that_the_driver_writes(tmp_path, capsys):
    """run_episodes.write_run writes run.json in its public form, which holds no endpoint URL, so the flush check reads serving.agent,
    the record of a model agent's endpoint that this form keeps: a model run whose episode on the wall clock has no flush record is
    named. The control: the same run with a clean flush passes."""
    run = tmp_path / "grid_10" / "model"
    run.mkdir(parents=True)
    now = time.time()
    serving = {"agent": ledger.serving_record({"name": MODEL, "endpoint": ENDPOINT, "endpoint_file": None, "endpoint_models": None}),
               "judge": None}
    run_episodes.write_run(run, {"clock_mode": run_health_checks.WALL_CLOCK, "started": datetime.fromtimestamp(now - 60).isoformat(timespec="seconds"),
                                 "agent": "plan_execute", "model": MODEL, "endpoint": ENDPOINT, "serving": serving}, [])
    written = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert written["endpoint"] is None and written["serving"]["agent"], written
    (run / "seed000000").mkdir()
    state = run / "seed000000" / "episode.json"
    state.write_text(json.dumps({"episode_started_at": now - 30, "episode_ended_at": now - 20}), encoding="utf-8")
    assert run_checks.main(["overlap", str(run.parent)]) == 1
    assert "OVERLAP_FLUSH model/seed000000: no record of the flush of the model server's cache before it" in capsys.readouterr().out
    state.write_text(json.dumps({"episode_started_at": now - 30, "episode_ended_at": now - 20, "cache_flush": {**CLEAN, "started_at": now - 32}}),
                     encoding="utf-8")
    assert run_checks.main(["overlap", str(run.parent)]) == 0


def test_a_run_taken_up_again_is_alive_where_each_job_played_it_and_not_between(tmp_path, capsys):
    """A job that the time limit cut leaves its run in progress, and a later job takes it up (run_episodes.resume_run); between the
    two the run plays nowhere, so a wall-clock episode that another run played in the gap is not named, while one played during the
    first attempt is. The run is built by the driver's writers: its ledger (ledger.start, ledger.add_episode), the episode that the cut
    left set aside into INTERRUPTED, the event of the resume (ledger.record_resume), whose second starts the second attempt, and
    run.json (run_episodes.write_run)."""
    now = time.time()
    grid = tmp_path / "grid_11"
    oracle = grid / "oracle"
    _started(oracle, now - 100)
    _state(oracle / "seed000000", episode_started_at=now - 98, episode_ended_at=now - 96)
    ledger.ledger_file.add_episode(oracle, "seed000000", {"seed": 0})
    _state(oracle / ledger.INTERRUPTED / "seed000001", episode_started_at=now - 95)
    ledger.record_resume(oracle, [*ARGV, "--resume", str(oracle)], ["seed000000"], [("seed000001", "seed000001")])  # into the same folder
    _state(oracle / "seed000001", episode_started_at=now + 1, episode_ended_at=now + 2)
    run_episodes.write_run(oracle, {"clock_mode": run_health_checks.WALL_CLOCK, "started": _second(now - 100), "agent": "oracle", "model": None,
                                    "endpoint": None, "serving": {"agent": None, "judge": None}}, [])
    _run(grid, "greedy", now - 60, now - 50, run_health_checks.WALL_CLOCK, episodes=[(now - 58, now - 52)], endpoint=None)
    assert run_checks.main(["overlap", str(grid)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == CLEAN_SUMMARY
    _run(grid, "early", now - 99, now - 97)  # played during the first attempt
    assert run_checks.main(["overlap", str(grid)]) == 1
    named = [line for line in capsys.readouterr().out.splitlines() if line.startswith("OVERLAP ")]
    assert len(named) == 1 and named[0].startswith("OVERLAP oracle/seed000000 ") and " overlaps early [" in named[0], named


def test_a_run_that_its_job_left_without_run_json_is_alive_up_to_the_last_second_it_recorded(tmp_path, capsys):
    """The driver writes run.json at a run's end alone (run_episodes.write_run), so a run that its job left, killed or cut, holds its
    ledger and its episodes and no run.json. It is alive from its start to the latest second that its episodes recorded, and a
    wall-clock episode that another run played meanwhile is named; its own episode records no clock, so it is not judged here. The
    control: an episode played after that last second is not named."""
    now = time.time()
    grid = tmp_path / "grid_12"
    killed = _killed(grid, "oracle", now - 100, now - 40)
    assert (killed / ledger.LEDGER).exists() and not (killed / "run.json").exists()
    _run(grid, "greedy", now - 70, now - 50, run_health_checks.WALL_CLOCK, episodes=[(now - 68, now - 52)], endpoint=None)
    assert run_checks.main(["overlap", str(grid)]) == 1
    named = [line for line in capsys.readouterr().out.splitlines() if line.startswith("OVERLAP ")]
    assert named == [f"OVERLAP greedy/seed000000 [{now - 68:.0f}, {now - 52:.0f}] overlaps oracle [{run_health_checks._epoch(_second(now - 100)):.0f}, "
                     f"{now - 40:.0f}]"], named
    later = tmp_path / "grid_13"
    _killed(later, "oracle", now - 100, now - 40)
    _run(later, "greedy", now - 30, now - 10, run_health_checks.WALL_CLOCK, episodes=[(now - 28, now - 12)], endpoint=None)
    assert run_checks.main(["overlap", str(later)]) == 0


def _cut_oracle(old, now):
    """A wall-clock run that the time limit cut after two finished episodes, so that its job left it without run.json: the ledger of
    its start (ledger.start) and the state of each episode (episode.save_state), which records the clock of the episode, with the
    ledger record that the driver writes after it (ledger.add_episode)."""
    oracle = old / "oracle"
    _started(oracle, now - 100)
    for k, (start, end) in enumerate([(now - 98, now - 96), (now - 94, now - 92)]):
        _state(oracle / f"seed{k:06d}", episode_started_at=start, episode_ended_at=end, config={"clock_mode": run_health_checks.WALL_CLOCK})
        ledger.ledger_file.add_episode(oracle, f"seed{k:06d}", {"seed": k})
    return oracle


def _take_up(oracle, new):
    """The run that _cut_oracle left, taken up by a later job into its own grid folder as run_episodes.resume_run takes it up: the
    folder moves, the ledger and the private record note the take-up with its command line (ledger.record_resume), and the run
    plays its last episode and writes run.json (run_episodes.write_run) with the start of its first job."""
    taken = new / oracle.name
    new.mkdir(exist_ok=True)
    oracle.rename(taken)
    ledger.record_resume(taken, [*ARGV, "--resume", str(oracle), "--out", str(new)], ["seed000000", "seed000001"], [], str(oracle))
    now = time.time()
    _state(taken / "seed000002", episode_started_at=now, episode_ended_at=now, config={"clock_mode": run_health_checks.WALL_CLOCK})
    ledger.ledger_file.add_episode(taken, "seed000002", {"seed": 2})
    started = json.loads((taken / ledger.LEDGER).read_text(encoding="utf-8"))["started"]
    run_episodes.write_run(taken, {"clock_mode": run_health_checks.WALL_CLOCK, "started": started, "agent": "oracle", "model": None,
                                   "endpoint": None, "serving": {"agent": None, "judge": None}}, [])
    return taken


def test_a_run_cut_beside_another_run_and_taken_up_into_another_folder_is_named_at_the_cut_and_after_the_take_up(tmp_path, capsys):
    """A job that the time limit cut leaves its wall-clock run without run.json, and a later job takes the run up into its own grid
    folder (run_episodes.resume_run), so the episodes that the run keeps were played beside the runs that stay in the folder of the
    cut job. A run on a clock without wall seconds that was alive during a kept episode is named at the cut, where the episodes of a
    run without run.json are judged by the clock that their states record, and after the take-up, where the folder that the
    take-up's --resume named in the private record of the run places the runs of the cut job. A kept episode whose earlier folder is
    gone is named as well, since nothing then shows it alone."""
    now = time.time()
    old, new = tmp_path / "grid_14", tmp_path / "grid_15"
    oracle = _cut_oracle(old, now)
    _run(old, "greedy", now - 95, now - 91, endpoint=None)  # on the declared clock, alive during the second episode alone
    kept = f"OVERLAP oracle/seed000001 [{now - 94:.0f}, {now - 92:.0f}] overlaps"
    greedy = f"[{run_health_checks._epoch(_second(now - 95)):.0f}, {run_health_checks._epoch(_second(now - 91)):.0f}]"
    assert run_checks.main(["overlap", str(old)]) == 1
    assert [line for line in capsys.readouterr().out.splitlines() if line.startswith("OVERLAP ")] == [f"{kept} greedy {greedy}"]
    _take_up(oracle, new)
    assert run_checks.main(["overlap", str(new)]) == 1
    assert [line for line in capsys.readouterr().out.splitlines() if line.startswith("OVERLAP ")] == [f"{kept} grid_14/greedy {greedy}"]
    old.rename(tmp_path / "moved")
    assert run_checks.main(["overlap", str(new)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in out if line.startswith("OVERLAP_UNRECORDED ")] == [
        "OVERLAP_UNRECORDED oracle/seed000000", "OVERLAP_UNRECORDED oracle/seed000001"], out


def test_a_run_cut_alone_and_taken_up_into_another_folder_passes_at_the_cut_and_after_the_take_up(tmp_path, capsys):
    """The control of the test above, a valid cut and take-up: the cut run played after the other runs of the folder of the cut job,
    and the later job played a run on a clock without wall seconds before the take-up and a wall-clock run after the taken-up run
    ended, so neither folder names an overlap."""
    now = time.time()
    old, new = tmp_path / "grid_16", tmp_path / "grid_17"
    _run(old, "wall", now - 130, now - 120, run_health_checks.WALL_CLOCK, episodes=[(now - 128, now - 122)], endpoint=None)
    _run(old, "base", now - 118, now - 108, endpoint=None)
    oracle = _cut_oracle(old, now)
    assert run_checks.main(["overlap", str(old)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == CLEAN_SUMMARY
    _run(new, "free", now - 60, now - 50, endpoint=None)
    _take_up(oracle, new)
    _run(new, "after", now + 10, now + 20, run_health_checks.WALL_CLOCK, episodes=[(now + 12, now + 18)], endpoint=None)
    assert run_checks.main(["overlap", str(new)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == CLEAN_SUMMARY
