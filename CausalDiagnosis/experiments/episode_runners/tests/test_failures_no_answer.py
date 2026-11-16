"""A tool call without an answer is the infrastructure's only when a server process of the
attempt shows a fault of its own."""
from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures

NO_ANSWER = "ToolTransportError: read_sensors of rig got no answer: the sessions closed before it came"


def _folder(tmp_path, *processes):
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    for server, k, code, log in processes:
        (hidden / f"{server}.{k}.stderr").write_text(log, encoding="utf-8")
        if code is not None:
            (hidden / f"{server}.{k}.returncode").write_text(f"{code}\n", encoding="utf-8")
    return tmp_path


def test_the_evidence_reads_each_process_with_its_code_and_traceback(tmp_path):
    folder = _folder(tmp_path, ("rig", 0, 0, "started\n"), ("rig", 1, 1, "Traceback (most recent call last):\n  ...\nKeyError: x\n"),
                     ("causal", 0, None, ""))
    assert failures.server_evidence(folder) == [("causal", 0, None, False), ("rig", 0, 0, False), ("rig", 1, 1, True)]


def test_a_nonzero_exit_with_a_traceback_is_the_infrastructures_for_every_client(tmp_path):
    servers = failures.server_evidence(_folder(tmp_path, ("rig", 0, 1, "Traceback (most recent call last):\nValueError: y\n")))
    for model_agent in (True, False):
        assert failures.attempt_class([NO_ANSWER], servers=servers, model_agent=model_agent)[0] == failures.INFRASTRUCTURE


def test_a_nonzero_exit_without_a_traceback_is_no_fault(tmp_path):
    servers = failures.server_evidence(_folder(tmp_path, ("rig", 0, 3, "started\n")))
    assert failures.attempt_class([NO_ANSWER], servers=servers, model_agent=True)[0] == failures.AGENT


def test_a_signal_is_a_fault_for_the_model_agents_only(tmp_path):
    servers = failures.server_evidence(_folder(tmp_path, ("rig", 0, -9, "")))
    assert failures.attempt_class([NO_ANSWER], servers=servers, model_agent=True)[0] == failures.INFRASTRUCTURE
    assert failures.attempt_class([NO_ANSWER], servers=servers, model_agent=False)[0] == failures.AGENT


def test_a_signal_of_the_blocks_own_teardown_is_no_fault(tmp_path):
    """A server still running at the end of its block and ended there by a signal (a cut while it was busy in a tool) leaves its
    call without an answer to the agent, so a cut by the agent's own time is not played again; so does a server that ends with a
    traceback after the close began. The control above: the same signal without the mark stays a fault for the model agents."""
    from CausalDiagnosis.agents.utils.mcp_sessions import TEARDOWN

    folder = _folder(tmp_path, ("causal", 0, -15, ""))
    (folder / "hidden" / "causal.0.returncode").write_text(f"-15\n{TEARDOWN}\n", encoding="utf-8")
    (folder / "hidden" / "rig.0.stderr").write_text("Traceback (most recent call last):\n", encoding="utf-8")
    (folder / "hidden" / "rig.0.returncode").write_text(f"1\n{TEARDOWN}\n", encoding="utf-8")  # an exit of its own after the close
    servers = failures.server_evidence(folder)
    assert servers == [("causal", 0, None, False), ("rig", 0, None, True)]
    assert failures.attempt_class([NO_ANSWER], servers=servers, model_agent=True)[0] == failures.AGENT


def test_healthy_processes_make_a_call_without_an_answer_the_agents(tmp_path):
    servers = failures.server_evidence(_folder(tmp_path, ("rig", 0, 0, "started\n"), ("causal", 0, 0, "")))
    assert failures.attempt_class([NO_ANSWER], servers=servers)[0] == failures.AGENT


def test_other_infrastructure_evidence_still_counts_without_a_server_fault(tmp_path):
    servers = failures.server_evidence(_folder(tmp_path, ("rig", 0, 0, "")))
    assert failures.attempt_class([NO_ANSWER, "APIConnectionError: refused"], servers=servers)[0] == failures.INFRASTRUCTURE
    assert failures.attempt_class([NO_ANSWER], harness_error="OSError: store", servers=servers)[0] == failures.INFRASTRUCTURE


def test_without_evidence_a_call_without_an_answer_stays_the_infrastructures(tmp_path):
    assert failures.attempt_class([NO_ANSWER])[0] == failures.INFRASTRUCTURE
    assert failures.attempt_class([NO_ANSWER], servers=failures.server_evidence(tmp_path))[0] == failures.INFRASTRUCTURE
