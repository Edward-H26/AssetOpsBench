"""A run killed in the middle of an episode keeps the model calls and the plans of that episode in its trace
(CausalDiagnosis.utils.episode_trace), while llm_calls.json and plans.json, which the end of the episode writes, are missing. The
driver runs in a process of its own, as on the cluster, and the stand-in endpoint kills its process group with SIGKILL, so that no
handler, finally block or end file of the driver runs. Run folders go under pytest's tmp_path, outside the repository."""
import importlib.util
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.datasets.tests.conftest import session_tables
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import ROOT
from CausalDiagnosis.utils import episode_trace

needs_a_run = pytest.mark.skipif(not Path(default_facts_path()).exists() or importlib.util.find_spec("mcp") is None,
                                 reason="a run needs the data file of the rig and the mcp package")
# the chat completion at which the driver is killed: the closing summary that the finish step of the third plan asks for, when
# two plans have run and the third is running
KILLED_AT = 5


class KillingPlanner:
    """The scripted planner of the causal agent, which kills the process group of the driver when the KILLED_AT-th chat completion
    arrives and keeps the prompts it received and the replies it sent."""

    def __init__(self, inner):
        self.inner, self.prompts, self.replies, self.driver = inner, [], [], None

    def generate(self, prompt, temperature=0.0):
        self.prompts.append(prompt)
        if len(self.prompts) == KILLED_AT:
            os.killpg(self.driver.pid, signal.SIGKILL)
            self.driver.wait()
        self.replies.append(self.inner.generate(prompt))
        return self.replies[-1]


@needs_a_run
def test_a_run_killed_in_the_middle_of_an_episode_keeps_its_model_calls_and_plans_in_the_trace(tmp_path):
    """The trace holds the calls answered before the kill with their replies, the call in flight with its prompt and no reply, the
    two plans that ran with their steps and the decision of the third, whose steps had not ended. The control: llm_calls.json and
    plans.json are missing, so without the trace the killed episode would keep none of them."""
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.tests.test_causal_diagnosis_agent import ScriptedPlanner, _planner_endpoint, act_now

    planner = KillingPlanner(ScriptedPlanner(act_now))
    server, _requests = _planner_endpoint(planner)
    env = {**os.environ, "API_KEY": "stand-in-key", "NO_PROXY": "127.0.0.1,localhost",  # pragma: allowlist secret
           "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    driver = "\n".join([*session_tables(), "import runpy", "runpy.run_module('CausalDiagnosis.experiments.episode_runners.run_episodes', "
                                                              "run_name='__main__', alter_sys=True)"])
    try:
        planner.driver = subprocess.Popen(
            [sys.executable, "-c", driver, "--config", "static", "--agent", "causal", "--model", "vendor/stand-in-1.0", "--base-url",
             f"http://127.0.0.1:{server.server_address[1]}/v1/", "--seeds", "1", "--no-grade", "--out", str(tmp_path / "run")],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        assert planner.driver.wait(timeout=900) == -signal.SIGKILL
    finally:
        server.shutdown()
        if planner.driver is not None and planner.driver.poll() is None:
            os.killpg(planner.driver.pid, signal.SIGKILL)
    (ep_dir,) = (tmp_path / "run").glob("*/seed000000")
    assert not (ep_dir / "llm_calls.json").exists() and not (ep_dir / "plans.json").exists()
    traced = episode_trace.records(ep_dir)
    calls, plans = traced["model_call"], traced["plan"]
    assert [call["prompt"] for call in calls] == planner.prompts and len(calls) == KILLED_AT
    assert [call.get("reply") for call in calls[:-1]] == planner.replies[:KILLED_AT - 1] and all("ended_at" in call for call in calls[:-1])
    assert not {"reply", "ended_at", "seconds"} & set(calls[-1]), calls[-1]  # the request in flight, kept without an answer
    assert [plan["index"] for plan in plans] == [1, 2, 3] and all(plan["steps"] for plan in plans[:2])
    assert plans[2]["plan"]["steps"] and plans[2]["steps"] == [], plans[2]  # the decision of the running plan, before its steps ended
