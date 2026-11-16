"""The class of a failed attempt of an episode, decided by code at run time from the type of each error under the harness
rule: whether the agent failed or the infrastructure under it, and whether the run stops.

An error is read in the form the recorders write it, "TypeName: message". Every piece of evidence of an attempt gets a kind from
its type, and the attempt is of class infrastructure when any piece is the infrastructure's, of class agent when there is any
other error, and of no class without one. A timeout is the agent's while the endpoint answers its model list, and the
infrastructure's when it does not, so one health check settles it, and it runs only when a timeout or a cut at the wall limit
needs it. A tool call without an answer is the infrastructure's when a server process of the attempt shows a fault of its own
(server_fault), and the agent's when the attempt's processes show none. The rule reads no score."""
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
import urllib.error
import urllib.request

# the kinds of a piece of evidence
CONNECTION, SERVER, SETUP, TIMEOUT, REQUEST, TOOL, HARNESS, OTHER = "connection", "server", "setup", "timeout", "request", "tool", "harness", "other"
INFRASTRUCTURE_KINDS = (CONNECTION, SERVER, SETUP, TOOL, HARNESS)
# the class of an attempt, and the error_class of a row: an episode whose two attempts were of class infrastructure is a
# harness_error, and an episode the verifier cannot read is of class verifier
INFRASTRUCTURE, AGENT, HARNESS_ERROR, VERIFIER = "infrastructure", "agent", "harness_error", "verifier"
# the ledger event of a run that stopped after harness errors in a row, which the leaderboard reads as stopped
STOP_EVENT = "the run stopped after harness errors in a row"

TYPES = {
    CONNECTION: ("APIConnectionError", "ConnectError", "ConnectTimeout", "ReadError", "RemoteProtocolError"),
    SERVER: ("InternalServerError", "RateLimitError", "ConflictError", "APIResponseValidationError"),
    SETUP: ("AuthenticationError", "PermissionDeniedError", "NotFoundError", "RouteSetupError"),
    TIMEOUT: ("APITimeoutError", "ReadTimeout"),
    REQUEST: ("BadRequestError", "UnprocessableEntityError"),
    TOOL: ("ToolTransportError",),
}
KIND_OF_TYPE = {name: kind for kind, names in TYPES.items() for name in names}
SERVER_STATUSES = (408, 409, 429)  # with every status of 500 or above
SETUP_STATUSES = (401, 403, 404)
REQUEST_STATUSES = (400, 422)
TRACEBACK = "Traceback (most recent call last):"  # what a Python process writes to its error output when it dies of an exception
SERVICE, WALL = "service", "wall"  # the ends of a cut at the wall limit


def error_type(text):
    """The type name before the first colon of an error text, "TypeName: message", or the empty string."""
    head = (text or "").split(":", 1)[0].strip()
    return head if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", head) else ""


def http_status(text):
    """The status of a status error in its message: "Error code: 404" of the OpenAI client, "'503 Service Unavailable'" of
    httpx, or None without one."""
    found = re.search(r"Error code: (\d{3})\b", text or "") or re.search(r"'(\d{3}) [A-Za-z][^']*'", text or "")
    return int(found.group(1)) if found else None


def kind_of(text, harness=False):
    """The kind of one piece of evidence: harness for an exception of a step of the harness itself, else by the type of the
    error, a status error by its status."""
    if harness:
        return HARNESS
    name = error_type(text).rsplit(".", 1)[-1]
    if name in KIND_OF_TYPE:
        return KIND_OF_TYPE[name]
    status = http_status(text) if name in ("HTTPStatusError", "APIStatusError") else None
    if status is not None:
        if status in SERVER_STATUSES or status >= 500:
            return SERVER
        if status in SETUP_STATUSES:
            return SETUP
        if status in REQUEST_STATUSES:
            return REQUEST
    return OTHER


def server_evidence(episode_folder):
    """[(server, k, return code or None, whether its error log holds a traceback)] of every server process the blocks of an
    attempt started in an episode folder, read from hidden/<server>.<k>.stderr and the return code written beside it; a code that
    the block marks as an end after its close began reads as None."""
    from CausalDiagnosis.agents.utils.mcp_sessions import TEARDOWN  # the mark the block writes, read where it is needed alone
    evidence = []
    for log in sorted((Path(episode_folder) / "hidden").glob("*.stderr")):
        server, _, k = log.stem.rpartition(".")
        code_file = log.with_suffix(".returncode")
        lines = code_file.read_text(encoding="utf-8").split() if code_file.exists() else []
        code = lines[0] if lines else ""
        # a server that was still running when the block began to close ended after what the agent saw, so its end is no evidence
        returncode = int(code) if code.lstrip("-").isdigit() and TEARDOWN not in lines[1:] else None
        evidence.append((server, int(k) if k.isdigit() else 0, returncode, TRACEBACK in log.read_text(encoding="utf-8", errors="replace")))
    return evidence


def server_fault(evidence, model_agent):
    """Whether a server process shows a fault of its own: a nonzero exit with a traceback in its own error
    log for every client, or an end by a signal for the fork's model agents, whose processes OOM kills, Slurm preemption and the
    driver's teardown also end by signal; for any other client a process ended by a signal is no fault of the server."""
    return any((code is not None and code > 0 and traceback) or (code is not None and code < 0 and model_agent)
               for _server, _k, code, traceback in evidence)


def attempt_class(errors, harness_error=None, cut=False, healthy=None, servers=None, model_agent=True, cut_end=None):
    """(class, kinds, health) of an attempt from its errors (texts in the recorders' form), the exception of a step of the
    harness if one raised, and whether the wall limit cut it. healthy is a function that checks the endpoint once; it is called
    at most once and only when a timeout or a cut needs it. servers is the attempt's server_evidence: when it names a process, a
    tool call without an answer is the infrastructure's only with a server_fault (model_agent saying whether the client is one of
    the fork's model agents), else the agent's; without it every such call is the infrastructure's. The class is infrastructure
    when any piece of evidence is the infrastructure's, agent when there is any other error, and empty without one."""
    kinds = [kind_of(text) for text in errors if text]
    if harness_error:
        kinds.append(HARNESS)
    counted = INFRASTRUCTURE_KINDS
    if servers and not server_fault(servers, model_agent):
        counted = tuple(kind for kind in INFRASTRUCTURE_KINDS if kind != TOOL)
    if any(kind in counted for kind in kinds):
        return INFRASTRUCTURE, kinds, None
    if cut and cut_end is not None:  # the end the cut was given: "service" is replayed, "wall" is the agent's own end
        return (INFRASTRUCTURE if cut_end == SERVICE else AGENT), kinds, None
    if TIMEOUT in kinds or cut:
        health = healthy() if healthy is not None else None
        return (INFRASTRUCTURE if health is not None and not health.get("ok") else AGENT), kinds, health
    return (AGENT if kinds else ""), kinds, None


def endpoint_health(base_url, api_key, timeout_s):
    """One GET of the model list of an endpoint: {ok, status, error, seconds}."""
    request = urllib.request.Request(base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as answer:
            status = answer.status
        return {"ok": 200 <= status < 300, "status": status, "error": None, "seconds": round(time.perf_counter() - started, 3)}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "error": f"HTTPError: {exc}", "seconds": round(time.perf_counter() - started, 3)}
    except Exception as exc:  # noqa: BLE001 - a health check that fails is its answer
        return {"ok": False, "status": None, "error": f"{type(exc).__name__}: {exc}", "seconds": round(time.perf_counter() - started, 3)}


class EndpointProbe:
    """The health probe of an endpoint for CausalDiagnosis.utils.model_request_retry.health_check: true while GET /models returns any HTTP status
    (endpoint_health). down holds once it answered no, which outside an episode ends the request with ServiceDown, so a caller
    whose own code keeps that ServiceDown as an outcome (a scorer's failed review, an answer's error) still learns of it."""

    def __init__(self, base_url):
        self.base_url, self.down = base_url, False

    def __call__(self):
        from CausalDiagnosis.utils.model_request_retry import READ_TIMEOUT_S  # here and not at the top: the agent package loads its runners on import

        answers = endpoint_health(self.base_url, os.environ.get("API_KEY"), READ_TIMEOUT_S)["status"] is not None
        self.down = self.down or not answers
        return answers


@contextmanager
def endpoint_probe(base_url):
    """Name the health probe of the endpoint at base_url for every model request made inside (CausalDiagnosis.utils.model_request_retry.health_check): a
    failed attempt outside an episode is sent again while the endpoint answers, and once it does not the request ends with
    ServiceDown instead of attempting again for ever. Yields the probe (EndpointProbe)."""
    from CausalDiagnosis.utils.model_request_retry import health_check

    probe = EndpointProbe(base_url)
    with health_check(probe):
        yield probe


def stop_due(rows, count):
    """Whether the last count rows of a run are all of the class harness_error, which stops the run."""
    tail = list(rows)[-count:]
    return len(tail) == count and all(row.get("error_class") == HARNESS_ERROR for row in tail)
