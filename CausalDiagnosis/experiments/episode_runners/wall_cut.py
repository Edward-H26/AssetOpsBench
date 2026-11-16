"""The end of an episode cut at the wall limit, decided from the facts recorded at the cut before anything is played. A failed
span of the retry rule open, a tool call open, or a model request open whose probe of the model service generates no token gives
"service": a service under the agent held the episode, and nothing is played. Otherwise the end is "wall", the agent's own time
ran out, and on a row with a thinking clock the caller then plays the accrued thinking once, stamped at the deadline.

The probe is one generation request of one token, sent after the cut through a client of its own (urllib), never through the
agent's route object or the recording backend, so it adds nothing to the episode's calls, tokens or ledgers. Its token is
returned as cut_probes, a count of its own. Its bound is the bound of the health probe of the harness, design.HEALTH_TIMEOUT_S,
which run_episodes passes to failures.endpoint_health."""
import json
import time
import urllib.error
import urllib.request

from CausalDiagnosis.experiments.table_definitions.experiment_tables import HEALTH_TIMEOUT_S
from CausalDiagnosis.experiments.episode_runners.attempt_error_classes import SERVICE, WALL


def cut_probe(base_url, api_key, model, timeout_s=HEALTH_TIMEOUT_S):
    """One POST of a chat completion of at most one token: {ok, status, error, seconds, cut_probes}. ok holds only when the
    service answered 2xx with a generated token, so a front end that answers its model list while the engine generates nothing
    is not ok."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "ok"}], "max_tokens": 1, "temperature": 0}).encode()
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})}
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body, headers=headers, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as answer:
            status, reply = answer.status, json.loads(answer.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "error": f"HTTPError: {exc}", "seconds": round(time.perf_counter() - started, 3),
                "cut_probes": 0}
    except Exception as exc:  # noqa: BLE001 - a probe that fails is its answer
        return {"ok": False, "status": None, "error": f"{type(exc).__name__}: {exc}", "seconds": round(time.perf_counter() - started, 3),
                "cut_probes": 0}
    tokens = generated_tokens(reply)
    return {"ok": 200 <= status < 300 and tokens > 0, "status": status, "error": None if tokens else "no token generated",
            "seconds": round(time.perf_counter() - started, 3), "cut_probes": tokens}


def generated_tokens(reply):
    """The tokens a chat completion generated: usage.completion_tokens, or one for a choice whose message holds text when the
    service sends no usage."""
    usage = (reply or {}).get("usage") or {}
    if isinstance(usage.get("completion_tokens"), int):
        return usage["completion_tokens"]
    choices = (reply or {}).get("choices") or []
    return int(any(((c.get("message") or {}).get("content") or c.get("text")) for c in choices))


def cut_end(failed_span_open, tool_call_open, model_request_open, probe=None):
    """{"end", "probe"} of a cut. A fact the harness did not record (None) counts as open. A failed span of the retry rule or a
    tool call open gives "service" with no probe. A model request open gives "wall" when probe() generated a token and "service"
    when it did not (or when no probe was given). Nothing open gives "wall"."""
    def is_open(fact):
        return fact is None or bool(fact)

    if is_open(failed_span_open) or is_open(tool_call_open):
        return {"end": SERVICE, "probe": None}
    if not is_open(model_request_open):
        return {"end": WALL, "probe": None}
    record = probe() if probe is not None else None
    return {"end": WALL if record and record.get("ok") else SERVICE, "probe": record}
