"""wall_cut.py: the end of a cut at the wall limit from the recorded facts and, while a model request is open, one probe of the
model service through a client of its own."""
from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures
from CausalDiagnosis.experiments.episode_runners import wall_cut
from CausalDiagnosis.experiments.table_definitions.experiment_tables import HEALTH_TIMEOUT_S

from CausalDiagnosis.experiments.episode_runners.tests.fake_model_service import FakeService


def _probe(service):
    return lambda: wall_cut.cut_probe(service.base_url, "", "stand-in")


def test_a_service_that_generates_gives_wall():
    with FakeService(generates=True) as service:
        result = wall_cut.cut_end(False, False, True, _probe(service))
    assert result["end"] == failures.WALL
    assert result["probe"]["ok"] and result["probe"]["cut_probes"] == 1 and result["probe"]["status"] == 200
    assert ("POST", "/v1/chat/completions") in service.paths


def test_a_front_end_that_answers_models_and_generates_nothing_gives_service():
    with FakeService(generates=False) as service:
        assert failures.endpoint_health(service.base_url, "", HEALTH_TIMEOUT_S)["ok"]  # the health probe alone would pass it
        result = wall_cut.cut_end(False, False, True, _probe(service))
    assert result["end"] == failures.SERVICE and not result["probe"]["ok"] and result["probe"]["cut_probes"] == 0


def test_nothing_open_gives_wall_without_a_probe():
    called = []
    result = wall_cut.cut_end(False, False, False, lambda: called.append(1))
    assert result == {"end": failures.WALL, "probe": None} and not called


def test_a_tool_call_or_a_failed_span_open_gives_service_without_a_probe():
    called = []
    for facts in ((False, True, False), (True, False, True), (None, False, False)):
        assert wall_cut.cut_end(*facts, probe=lambda: called.append(1)) == {"end": failures.SERVICE, "probe": None}
    assert not called


def test_the_probe_is_bounded_by_the_health_bound():
    import inspect
    assert inspect.signature(wall_cut.cut_probe).parameters["timeout_s"].default == HEALTH_TIMEOUT_S
