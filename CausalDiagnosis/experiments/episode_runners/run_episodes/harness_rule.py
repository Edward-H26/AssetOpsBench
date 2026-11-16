import asyncio
import json
import time
from CausalDiagnosis.experiments.table_definitions.experiment_tables import EPISODE_WALL_LIMIT_S, HEALTH_TIMEOUT_S, INFRASTRUCTURE_REPLAYS, STOP_AFTER_FAILURES  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures  # noqa: E402
from CausalDiagnosis.utils import model_request_retry as retry  # noqa: E402
from . import agent_kinds, play_one_episode


# ----- the harness rule of every agent: the wall limit, the end, the class of a failure, the replay and the stop -----
WALL_LIMIT = "WallLimit"  # the type name of the error of a cut, "WallLimit: the episode passed N s"
# the end a runner reports, in the words of the row: the own limits of the ports (their step limit, their loop detection) are
# one word; the causal controller has no limit of its own and ends by finish, episode_over, cancelled or error
RUNNER_ENDS = {"finish": "finish", "max_steps": "own_limit", "loop": "own_limit",
               "episode_over": "episode_over", "cancelled": "cancelled", "error": "error"}
ATTEMPT_FOLDER = "attempt1"  # the attempt that the infrastructure broke, set aside inside its episode folder
SERVER_CHECK_FOLDER = "server_check"  # the scratch folder of the tool listing before the first episode of a run
HARNESS_FILE = "harness.json"  # what the harness did to an episode


def run_limited(make_coroutine, limit_s=None):
    """(result, error, seconds) of one runner coroutine under the wall limit of an episode: the deadline the harness named for the
    episode (CausalDiagnosis.utils.model_request_retry.episode_deadline, a WallLimit of attempt) when limit_s names none, else limit_s from now
    (design.EPISODE_WALL_LIMIT_S without either). At the limit the runner is cancelled: a port raises the cancellation and keeps
    its calls on the runner object, the controller returns its partial result, and the error is "WallLimit: the episode passed
    N s". A runner that ends at or past the deadline by its own read of it (a request or a retry wait given the time left, a model
    call that held the event loop) is cut in the same way with the facts taken at the same deadline, so the order in which the
    runner and the harness read the deadline changes nothing; while the runner plays, the wait of a call does not end at the
    deadline by itself (CausalDiagnosis.utils.model_request_retry.driver_cut). A result that ends cancelled while no cut was sent comes from an interrupt of
    the user, which a controller that returns on a cancellation would swallow, so it is raised again as KeyboardInterrupt."""
    if limit_s is not None or retry.current_deadline() is None:
        limit_s = EPISODE_WALL_LIMIT_S if limit_s is None else limit_s
        with retry.episode_deadline(time.monotonic() + limit_s):
            return _run_to_deadline(make_coroutine, limit_s)
    return _run_to_deadline(make_coroutine, EPISODE_WALL_LIMIT_S)


def _run_to_deadline(make_coroutine, limit_s):
    """run_limited under the deadline named for the episode, a cut named by the limit limit_s that it enforces."""
    cut = False

    async def limited():
        nonlocal cut
        ended = []  # the instant the runner's coroutine ended

        async def runner():
            try:
                return await make_coroutine()
            finally:
                ended.append(time.monotonic())

        task = asyncio.ensure_future(runner())
        while True:  # the deadline moves as credits arrive, so it is read again whenever the wait ends
            deadline = retry.deadline_at(retry.current_deadline())
            left = deadline - time.monotonic()
            if left <= 0:
                break
            done, _pending = await asyncio.wait({task}, timeout=left)
            if task in done:
                deadline = retry.deadline_at(retry.current_deadline())
                if ended[0] < deadline:
                    return task.result()
                break  # the runner read the deadline before this wait did and ended at it: that is the cut
        cut = True
        play_one_episode.note_cut(deadline)
        task.cancel()
        try:
            return await task
        except asyncio.CancelledError:
            return None

    started = time.perf_counter()
    try:
        with retry.driver_cut():
            result = asyncio.run(limited())
        error = getattr(result, "error", None)
    except Exception as exc:  # noqa: BLE001 - a failed run is a result of the benchmark
        result, error = None, f"{type(exc).__name__}: {exc}"
    if cut:
        error = f"{WALL_LIMIT}: the episode passed {limit_s:g} s" + (f"; {error}" if error else "")
    elif getattr(result, "end", None) == "cancelled":
        raise KeyboardInterrupt
    return result, error, time.perf_counter() - started


def finish_stamp(calls, ended, cut):
    """(the wall time of a harness finish, where it comes from): the end of the attempt's last model call, as both recorders
    write it into a call record (ended_at), so a runner is charged its last model call and not the teardown of its sessions;
    the end of the runner's part without a finished model call; the deadline the cut fell on after a cut at the wall limit,
    which the caller passes as ended (EpisodeRunner.attempt), so the teardown of the sessions is not charged."""
    if cut:
        return ended, "the cut at the wall limit"
    last = max((c["ended_at"] for c in calls if c.get("ended_at") is not None), default=None)
    return (last, "the end of the last model call") if last is not None else (ended, "the end of the runner's part")


def tool_error(record):
    """The error of a record of a tool call, a dict of the ports and the controller or a record of the released runner."""
    return record.get("error") if isinstance(record, dict) else getattr(record, "error", None)


def runner_end_of(agent, result, error):
    """How the runner ended, in the words of RUNNER_ENDS: finish, own_limit, episode_over, cancelled or error, and empty for an
    agent without a runner (the reference policies)."""
    if agent in agent_kinds.POLICY_AGENTS:
        return ""
    if failures.error_type(error or "") == WALL_LIMIT:
        return "cancelled"
    if result is None:
        return "error"
    end = getattr(result, "end", None)
    if end is None:  # the released runner of the scripted check names no end: it ends when its plan is done
        return "error" if error else "finish"
    return RUNNER_ENDS.get(end, "error")


def cut_replies(calls):
    """The model calls whose reply ended at the token cap (finish reason length)."""
    return sum(1 for c in calls if c.get("finish_reason") == "length")


def failed_calls(calls, tools):
    """The model calls that ended in an error after the retries of their client, the failed token counts of a port among them,
    and the tool calls that got no answer."""
    return sum(1 for c in calls if c.get("error")) + sum(1 for t in tools if failures.kind_of(tool_error(t) or "") == failures.TOOL)


def diagnosis_refusals(log):
    """{reason: count} of the refused entries of a log whose action is a diagnosis, empty when no diagnosis was refused."""
    counts = {}
    for e in log:
        if e["event"] == "refused" and (e.get("action") or {}).get("type") == "submit_diagnosis":
            counts[e["reason"]] = counts.get(e["reason"], 0) + 1
    return json.dumps(dict(sorted(counts.items()))) if counts else ""


def harness_record(attempts, aside):
    """harness.json: the number of attempts, the attempt set aside with its error, its kinds, its health check and its folder,
    and the attempt that stands with its class, its kinds, the health check when one ran, the finish and the runner's end."""
    def part(a):
        return {"class": a["class"], "kinds": a["kinds"], "health": a["health"], "error": a["harness_error"] or a["error"],
                "finish_by": a["finish_by"], "finish": a["finish"], "runner_end": a["runner_end"], "wall_limit": a["cut"],
                "cut_end": a.get("cut_end", ""), "cut_probe": a.get("cut_probe"), "cache_flush": a.get("cache_flush")}
    return {"attempts": len(attempts), "set_aside": None if aside is None else {**part(attempts[0]), "folder": aside},
            "standing": part(attempts[-1])}


def harness_settings():
    """The block harness of run.json: the rule of an episode that every agent plays under."""
    return {"episode_wall_limit_s": EPISODE_WALL_LIMIT_S, "infrastructure_replays": INFRASTRUCTURE_REPLAYS,
            "stop_after_failures": STOP_AFTER_FAILURES, "health_timeout_s": HEALTH_TIMEOUT_S,
            "finish_stamp": "a harness finish is stamped at the end of the last model call of the attempt, at the end of the runner's "
                            "part without one, and at the time of the cut after a cut at the wall limit. On a row with a thinking "
                            "clock a no_call step with the same stamp comes first, and no finish follows when that step ends the "
                            "episode"}
