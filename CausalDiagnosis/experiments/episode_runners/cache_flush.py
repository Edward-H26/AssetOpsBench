"""The flush of the model server's cache before an episode of a preset with the wall clock. Every episode of a wall-clock row
starts from the same cache state, an empty prefix tree and KV pool, so its charged seconds do not depend on the episode before it.
No warm-up episode is played: the first episode after a launch carries the JIT work of the engine, which the run records and the
flush does not remove.

The harness calls it after the last reply of the episode before has returned. SGLang flushes only when its scheduler is idle;
with ?timeout it waits that long for idle and answers 400 otherwise, and on any answer but 200 the episode must not start."""
import json
import re
import time
import urllib.error
import urllib.request

from CausalDiagnosis.experiments.table_definitions.experiment_tables import HEALTH_TIMEOUT_S

METRICS = {"running": "sglang:num_running_reqs", "queued": "sglang:num_queue_reqs"}


class CacheNotFlushed(RuntimeError):
    """The model server did not flush its cache, so the episode must not start."""


def _root(base_url):
    """The server root of an OpenAI base URL: /flush_cache and /metrics live beside /v1, not under it."""
    root = base_url.rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def _headers(api_key):
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def engine_load(base_url, api_key=None, timeout_s=HEALTH_TIMEOUT_S):
    """{"running", "queued"} from the Prometheus text of /metrics, each None when the server does not export it."""
    load = dict.fromkeys(METRICS)
    try:
        with urllib.request.urlopen(urllib.request.Request(_root(base_url) + "/metrics", headers=_headers(api_key)),
                                    timeout=timeout_s) as answer:
            text = answer.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - a server without metrics leaves the counts empty
        return load
    for key, name in METRICS.items():
        values = [float(m) for m in re.findall(rf"^{re.escape(name)}(?:{{[^}}]*}})?\s+([0-9.eE+-]+)\s*$", text, re.MULTILINE)]
        load[key] = sum(values) if values else None
    return load


def cache_report(base_url, api_key=None, timeout_s=HEALTH_TIMEOUT_S):
    """Whether the server reports the cached prompt tokens of a reply by SGLang's rule, enable_cache_report of its GET /server_info
    (SGLang v0.5.19, the route that engine_record.py reads), under which a reply that reused no cached token carries no count: True
    or False, and None when the route gives no such answer, as vLLM's server and a router give none. The answer holds the server's
    key and paths, so the one value alone is kept."""
    try:
        with urllib.request.urlopen(urllib.request.Request(_root(base_url) + "/server_info", headers=_headers(api_key)),
                                    timeout=timeout_s) as answer:
            value = json.loads(answer.read().decode("utf-8")).get("enable_cache_report")
    except Exception:  # noqa: BLE001 - a server without the route, or with an answer that is no JSON object, reports nothing here
        return None
    return value if isinstance(value, bool) else None


def flush_before_episode(base_url, api_key=None, timeout_s=HEALTH_TIMEOUT_S):
    """POST /flush_cache?timeout=timeout_s and require 200. Returns {flush_seconds, status, running, queued,
    first_request_cached_tokens}; the last is None here and the harness writes the cached tokens of the episode's first model
    request into it, which must be 0, and the server's cache_report beside it, which must be true. Raises CacheNotFlushed on 400
    or on any other answer."""
    url = f"{_root(base_url)}/flush_cache?timeout={timeout_s}"
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=b"", headers=_headers(api_key), method="POST"),
                                    timeout=timeout_s + HEALTH_TIMEOUT_S) as answer:
            status = answer.status
    except urllib.error.HTTPError as exc:
        raise CacheNotFlushed(f"flush_cache answered {exc.code}: {exc.read().decode('utf-8', 'replace').strip()}") from exc
    except Exception as exc:  # noqa: BLE001 - no answer is no flush
        raise CacheNotFlushed(f"flush_cache got no answer: {type(exc).__name__}: {exc}") from exc
    seconds = round(time.perf_counter() - started, 3)
    if status != 200:
        raise CacheNotFlushed(f"flush_cache answered {status}")
    return {"flush_seconds": seconds, "status": status, **engine_load(base_url, api_key, timeout_s), "first_request_cached_tokens": None}
