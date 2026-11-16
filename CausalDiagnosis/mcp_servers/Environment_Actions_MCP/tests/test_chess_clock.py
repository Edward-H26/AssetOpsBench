"""The chess clock of the thinking time on the action server (servers/thinking_clock.py, CausalDiagnosis/docs/environment.md):
the charge of a counted call is the monotonic time since the cursor less the credited segments, the cursor moves by the credit
of a service alone, the first start of each server and every failed model attempt are credited, a call before the handover
and a call of a process whose ready line shows a problem are refused, and the identity of the charges holds on random plays.
Every reading is handed to the server through episode.recorded_clock, so each value below is exact."""
import dataclasses
import os
import random
from pathlib import Path

import pytest

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.utils import model_failures
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import thinking_clock
from CausalDiagnosis.mcp_servers.utils import transcript
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode

pytestmark = pytest.mark.skipif(not Path(default_facts_path()).exists(), reason="hydraulic rig facts file not present; set HYD_FACTS_PATH")
H = 100.0  # the handover on the monotonic clock
THINK = 10.0
PROFILE = {"type": "get_asset_profile"}
ISOLATED = {"flags": {"isolated": 1}}


def started(tmp_path, hand_over=True):
    tmp_path.mkdir(parents=True, exist_ok=True)  # episode.start creates no folder, as the driver makes each episode folder itself
    path = tmp_path / "episode.json"
    config = dataclasses.replace(EpisodeConfig(), clock_mode="generation_time", think_seconds_per_cycle=THINK)
    with episode.recorded_clock(H - 5.0, H - 5.0):
        episode.start(path, 3, config, default_facts_path())
    if hand_over:
        episode.hand_over(path, wall=H, mono=H)
    return path


def serve(path, e, f=None, credit=0.0):
    """One counted call entering at e and ending at f with `credit` seconds of the serving thread's time; returns its charge."""
    f = e if f is None else f
    with episode.recorded_clock(e, e, stamps=[(e, e, 0.0), (f, f, credit)]):
        episode.call(path, "get_asset_profile", {}, PROFILE)
    return episode.load_state(path)["think_seconds"][-1]


def ready(path, server, s, r, thread=None, pid=None, **facts):
    """A ready line of a server process started at s and ready at r, with `thread` seconds of its main thread's time."""
    thread = r - s if thread is None else thread
    fields = {"server": server, "s_wall": s, "s_mono": s, "s_thread_cpu": 0.0, "r_wall": r, "r_mono": r, "r_thread_cpu": thread,
              **ISOLATED, **facts}
    n = transcript.append(path.parent, "ready", **fields)
    if pid is not None:  # the line names another process than the test's own
        text = (path.parent / transcript.NAME).read_text(encoding="utf-8").replace(f'"pid":{os.getpid()}', f'"pid":{pid}')
        (path.parent / transcript.NAME).write_text(text, encoding="utf-8")
    return n


def test_the_arithmetic_of_the_clock():
    assert thinking_clock.charge(0.0, 10.0, [(2.0, 3.0), (2.5, 4.0)]) == 8.0
    assert thinking_clock.charge(5.0, 4.0, []) == 0.0  # a call that entered before the cursor is charged nothing
    assert thinking_clock.advance(10.0, 9.0, 15.0, 2.0) == 12.0  # the credit counts from the cursor when the call entered before it
    assert thinking_clock.advance(10.0, 12.0, 15.0, 2.0) == 14.0
    assert thinking_clock.advance(10.0, 12.0, 13.0, 5.0) == 13.0  # never past the end of the service
    assert thinking_clock.start_segment({"s_mono": 1.0, "s_thread_cpu": 0.0, "r_mono": 3.0, "r_thread_cpu": 0.5}) == (1.0, 1.5)
    assert thinking_clock.start_segment({"s_mono": None, "s_thread_cpu": None, "r_mono": 3.0, "r_thread_cpu": 0.5}) is None


def test_a_call_is_charged_the_time_since_the_handover(tmp_path):
    path = started(tmp_path)
    assert serve(path, H + 7.0) == 7.0
    state = episode.load_state(path)
    assert state["handover_mono"] == H and state["open_service"]["e"] == H + 7.0


def test_the_cursor_moves_by_the_credit_of_the_service_alone(tmp_path):
    whole, part = started(tmp_path / "a"), started(tmp_path / "b")
    for path, credit in ((whole, 3.0), (part, 1.0)):
        serve(path, H + 7.0, H + 10.0, credit)
    assert serve(whole, H + 15.0) == 5.0  # the three seconds of service were the thread's own
    assert serve(part, H + 15.0) == 7.0  # two seconds of the service beyond its credit are charged at the next call


def test_a_stop_inside_a_handler_raises_the_next_charge_by_exactly_its_length(tmp_path):
    plain, stopped = started(tmp_path / "a"), started(tmp_path / "b")
    serve(plain, H + 7.0, H + 10.0, 3.0)
    serve(stopped, H + 7.0, H + 10.0 + 4.0, 3.0)  # four seconds in which the serving thread did no work
    # the agent answers the same way, so its next call comes four seconds later, after the later answer
    assert serve(stopped, H + 24.0) - serve(plain, H + 20.0) == 4.0


def test_a_pause_before_a_burst_raises_the_total_by_exactly_its_length(tmp_path):
    plain, paused = started(tmp_path / "a"), started(tmp_path / "b")
    for path, pause in ((plain, 0.0), (paused, 6.5)):
        for k, e in enumerate((H + 5.0, H + 6.0, H + 7.0)):
            serve(path, e + pause, e + pause + 0.5, 0.5)
    totals = [sum(episode.load_state(p)["think_seconds"]) for p in (plain, paused)]
    assert totals[1] - totals[0] == 6.5


def test_the_first_start_of_each_server_is_credited_and_a_restart_excludes_nothing(tmp_path):
    path = started(tmp_path)
    ready(path, "hydraulic_rig", H + 1.0, H + 4.0, thread=3.0)
    ready(path, "causal_diagnosis", H + 2.0, H + 6.0, thread=1.0)  # credited from its start: [H + 2, H + 3]
    assert serve(path, H + 10.0) == 10.0 - 3.0  # [H + 1, H + 4] holds [H + 2, H + 3]
    ready(path, "hydraulic_rig", H + 11.0, H + 13.0, thread=2.0)  # a restart of a server already started
    assert serve(path, H + 15.0) == 5.0
    state = episode.load_state(path)
    assert state["started_servers"] == ["hydraulic_rig", "causal_diagnosis"]
    assert state["credited_seconds"]["starts"] == 4.0 and state["uncredited_seconds"]["starts"] == 3.0


@pytest.mark.parametrize("status", ["connection", 429, 502, 503])
def test_a_failed_model_attempt_with_its_wait_changes_no_charge(tmp_path, status):
    plain, failed = started(tmp_path / "a"), started(tmp_path / "b")
    model_failures.add_failed(failed.parent, H + 2.0, H + 5.0, status)
    assert serve(failed, H + 10.0) == serve(plain, H + 10.0) - 3.0
    state = episode.load_state(failed)
    assert state["failed_model_seconds"] == 3.0 and state["failed_model_attempts"] == 1


def test_listings_change_no_charge(tmp_path):
    plain, listed = started(tmp_path / "a"), started(tmp_path / "b")
    for _ in range(50):
        transcript.append(listed.parent, "list", server="hydraulic_rig", listing_sha256="0" * 64)
    assert serve(listed, H + 9.0) == serve(plain, H + 9.0) == 9.0


def test_a_call_before_the_handover_is_refused_and_a_second_handover_too(tmp_path):
    path = started(tmp_path, hand_over=False)
    with pytest.raises(ValueError, match="never handed over"):
        serve(path, H + 1.0)
    episode.hand_over(path, wall=H, mono=H)
    with pytest.raises(ValueError, match="handed over already"):
        episode.hand_over(path)
    assert serve(path, H + 2.0) == 2.0


@pytest.mark.parametrize("facts, words", [({"flags": {"isolated": 0}}, "isolated"), ({"trace": True}, "trace"),
                                          ({"tracemalloc": True}, "allocations"), ({"customize": ["sitecustomize"]}, "sitecustomize"),
                                          ({"environment": {"PYTHONASYNCIODEBUG": "1"}}, "PYTHONASYNCIODEBUG")])
def test_a_call_of_a_process_whose_ready_line_shows_a_problem_is_refused(tmp_path, facts, words):
    path = started(tmp_path)
    ready(path, "hydraulic_rig", H + 1.0, H + 2.0, **facts)  # this process
    with pytest.raises(ValueError, match=words):
        serve(path, H + 5.0)
    other = started(tmp_path / "other")
    ready(other, "hydraulic_rig", H + 1.0, H + 2.0, pid=1, **facts)  # another process: this one is not refused
    assert serve(other, H + 5.0) == 4.0


def test_accrued_thinking_is_the_charge_of_a_call_entering_now_and_writes_nothing(tmp_path):
    path = started(tmp_path)
    serve(path, H + 4.0, H + 6.0, 2.0)
    model_failures.add_failed(path.parent, H + 8.0, H + 9.0, 503)
    before = path.read_bytes()
    seconds, cycles = episode.accrued_thinking(path, H + 21.0)
    assert path.read_bytes() == before
    assert seconds == 14.0 and cycles == 1
    assert serve(path, H + 21.0) == seconds


def test_the_charges_meet_the_identity_of_the_clock_on_random_plays(tmp_path):
    """The charges sum to |[H, u_K]| less the measure of the union of the credited segments of B and of the service segments
    [max(e_k, u_{k-1}), u_k], when the calls are served in their order of entry and every record lands in time; no charge is
    negative."""
    rng = random.Random(11)
    for play in range(12):
        path = started(tmp_path / str(play))
        t, u, credited = H, H, []
        for k in range(rng.randint(3, 9)):
            if rng.random() < 0.4:  # a server starts before this call
                s = t + rng.uniform(0.0, 3.0)
                r = s + rng.uniform(0.0, 4.0)
                thread = rng.uniform(0.0, r - s)
                ready(path, f"server{k}", s, r, thread=thread)
                credited.append((s, s + thread))
            if rng.random() < 0.4:  # a failed model attempt before this call
                a = t + rng.uniform(0.0, 3.0)
                b = a + rng.uniform(0.0, 3.0)
                model_failures.add_failed(path.parent, a, b, 503)
                credited.append((a, b))
            e = t + rng.uniform(0.5, 8.0)
            f = e + rng.uniform(0.0, 3.0)
            g = rng.uniform(0.0, f - e)
            assert serve(path, e, f, g) >= 0.0
            u_next = thinking_clock.advance(u, e, f, g)
            credited.append((max(e, u), u_next))
            u, t = u_next, f
        serve(path, t + 1.0)  # the last service moves the cursor once this call reads its line
        charges = episode.load_state(path)["think_seconds"][:-1]
        union = thinking_clock.merged(credited)
        covered = sum(min(b, u) - max(a, H) for a, b in union if b > H and a < u)
        assert sum(charges) == pytest.approx((u - H) - covered, abs=1e-9)


def test_a_call_whose_line_is_never_written_is_not_charged_again(tmp_path, monkeypatch):
    path = started(tmp_path)
    written = transcript.answered

    def lost(*args, **kwargs):
        raise RuntimeError("the process ended between the save and the line")
    monkeypatch.setattr(transcript, "answered", lost)
    with pytest.raises(RuntimeError):
        serve(path, H + 50.0)
    monkeypatch.setattr(transcript, "answered", written)
    assert episode.load_state(path)["think_seconds"] == [50.0]
    assert serve(path, H + 100.0) == 50.0  # not 100: the cursor passed the entry of the lost call with its charge


def test_a_stepped_call_line_names_the_ledger_offset_its_clock_read(tmp_path):
    path = started(tmp_path)
    model_failures.add_failed(path.parent, H + 2.0, H + 5.0, 503)
    serve(path, H + 10.0)
    model_failures.add_failed(path.parent, H + 11.0, H + 12.0, 503)  # after the last counted call
    line = [entry for entry in transcript.read(path.parent) if entry["kind"] == "call"][-1]
    ledger = (path.parent / model_failures.NAME).read_bytes()
    assert 0 < line["failures_read"] < len(ledger) and ledger[:line["failures_read"]].endswith(b"\n")


def test_a_handover_given_both_readings_takes_no_stamp(tmp_path, monkeypatch):
    path = started(tmp_path, hand_over=False)

    def no_stamp(*args, **kwargs):
        raise AssertionError("the handover read a clock although both readings were given")
    monkeypatch.setattr(episode, "stamp", no_stamp)
    assert episode.hand_over(path, wall=H, mono=H) == (H, H)


def test_accrued_thinking_on_the_token_clock_is_the_token_charge(tmp_path, monkeypatch):
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_2_real_time import TOKEN_MODE
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock.token_clock import token_seconds

    path = tmp_path / "episode.json"
    config = dataclasses.replace(EpisodeConfig(), clock_mode=TOKEN_MODE)
    with episode.recorded_clock(H - 5.0, H - 5.0):
        episode.start(path, 3, config, default_facts_path())
    episode.hand_over(path, wall=H, mono=H)
    monkeypatch.setattr(episode, "GENERATED_TOKENS", lambda folder: 1200)
    seconds, _cycles = episode.accrued_thinking(path, H + 500.0)  # the wall seconds since the handover play no part
    assert seconds == token_seconds(EpisodeConfig.from_json(episode.load_state(path)["config"]), 1200)
