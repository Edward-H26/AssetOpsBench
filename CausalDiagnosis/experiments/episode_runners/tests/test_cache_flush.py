"""cache_flush.py: the flush of the model server's cache before an episode of a wall-clock preset: 200 required, 400 refuses
the episode, the engine's counts read from /metrics, the cached tokens of the first request left for the harness."""
import pytest

from CausalDiagnosis.experiments.episode_runners import cache_flush

from CausalDiagnosis.experiments.episode_runners.tests.fake_model_service import FakeService


def test_a_flush_answered_200_returns_the_record():
    with FakeService(flush_status=200, metrics="sglang:num_running_reqs 0.0\nsglang:num_queue_reqs 2.0\n") as service:
        record = cache_flush.flush_before_episode(service.base_url, "")
    assert record["status"] == 200 and record["flush_seconds"] >= 0
    assert record["running"] == 0 and record["queued"] == 2 and record["first_request_cached_tokens"] is None
    assert any(m == "POST" and p.startswith("/flush_cache?timeout=") for m, p in service.paths)  # at the root, beside /v1


def test_a_flush_answered_400_refuses_the_episode():
    with FakeService(flush_status=400) as service:
        with pytest.raises(cache_flush.CacheNotFlushed, match="400"):
            cache_flush.flush_before_episode(service.base_url, "")


def test_no_answer_refuses_the_episode():
    with pytest.raises(cache_flush.CacheNotFlushed, match="no answer"):
        cache_flush.flush_before_episode("http://127.0.0.1:9/v1", "", timeout_s=1)


def test_a_server_without_metrics_leaves_the_counts_empty():
    with FakeService(metrics="other_metric 1\n") as service:
        assert cache_flush.engine_load(service.base_url) == {"running": None, "queued": None}
