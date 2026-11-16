"""A run of the reproduction names its time limit as the episode deadline of CausalDiagnosis.utils.model_request_retry (run_scenarios.timed_run), which every
model request and tool wait of the paper-era ports reads, so a dead service ends the run at the cut."""
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import run_scenarios
from CausalDiagnosis.utils import model_request_retry as retry


def test_a_timed_run_names_the_episode_deadline_that_the_ports_read():
    seen = {}

    class Runner:
        async def run(self, question):
            seen["left"] = retry.deadline_left()
            return "answer"

    result, error, _ = run_scenarios.timed_run(Runner(), "What IoT sites are available?", 60)
    assert (result, error) == ("answer", None) and 0 < seen["left"] <= 60
    assert retry.deadline_left() is None
