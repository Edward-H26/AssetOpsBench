"""The model route of the Agent-As-Tool port: raw text completions served by SGLang, as the paper's agents
generated text on watsonx.ai.

ReActXen's watsonx_llm sent the prompt string to watsonx text generation, greedy at temperature 0, at most 500 new
tokens and at least 1, with the stop sequences of the call, through a client that retried five times on 502 and 503,
and read back the first result, whose generated_text ends with the matched stop sequence (IBM's samples show the raw
outputs ending in "\\nAction" and "\\nObservation"). A route sends the same request to SGLang's /v1/completions with
no_stop_trim, so the reply keeps the matched stop string too, and keeps one record per call for llm_calls.json: the
exact prompt and reply, the token counts, the finish reason, the stop list, the agent whose call it was and the kind of
call, read from the stop list ReActXen passes. count_tokens and context_length use SGLang's /v1/tokenize and /v1/models.
A record also keeps the cached prompt tokens that the server reports, None when it reports no count, and enters the trace of the
episode as it happens (CausalDiagnosis.utils.episode_trace).
Every request carries the paper's max_tokens of 500, except on a run with the CausalDiagnosis MCP (reply_room), where every request
carries the room the served context leaves its prompt, so that no reply is capped, as no reply of the CausalDiagnosis agent is
(CausalDiagnosis.utils.model_reply_room). The route has no time limit and no count of retries of its own
(CausalDiagnosis.utils.model_request_retry): each attempt gets the time left before the row's deadline. A prompt that, with the
max_tokens of its request, does not fit the served context is refused by SGLang with 400 BadRequest; that refusal is not retried
and ends the run with the end "context". On a run with the CausalDiagnosis MCP a prompt that leaves no room ends the run with that
end before any request (room_of).

The vendored agents reach the route of the running episode through active_route(), since ReActXen binds watsonx_llm as
the default argument of every agent class at import.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import httpx

from CausalDiagnosis.utils.model_request_retry import RETRY_DELAY_S, STATUSES, current_deadline, post_with_retries, reported_cached_tokens
from CausalDiagnosis.utils.model_reply_room import ENGINE_MARGIN, ContextEnd, ReplyRoom, card_context_length
from CausalDiagnosis.utils import episode_trace

PAPER_DECODING = {"temperature": 0.0, "max_tokens": 500, "min_tokens": 1, "no_stop_trim": True}
# the max_tokens of a route with reply_room as run.json records it, a value computed for each request
REPLY_ROOM = "the served context less the tokenize count of the prompt less 2"
# the route retries as every model client of the benchmark does (CausalDiagnosis.utils.model_request_retry): no count of attempts, the registered wait of
# watsonx's ModelInference(delay_time=2), and 429 beside the paper's 502 and 503
RETRY_STATUS = STATUSES
SUPERVISOR = "supervisor"
# the kind of a call by the stop list ReActXen passes, the agent's "\nQuestion" appended where it appends it
KINDS = {("\nAction", "\nObservation", "\nQuestion"): "thought", ("\nObservation", "\nThought", "\nQuestion"): "action",
         ("\nThought", "\nAction", "\nQuestion"): "self_ask", ("\nQuestion",): "probe",
         ("Previous Trial:", "Previous Trial", "\n\n"): "reflection", ("\n(END OF RESPONSE)",): "review"}


class RunStopped(BaseException):
    """Ends a run from inside the agents' thread: the environment ended the episode or the run was cancelled. A
    BaseException, so that no ``except Exception`` of ReActXen or of the paper's tool wrappers holds it back."""


class RouteSetupError(RuntimeError):
    """The route cannot read from the endpoint what it needs before any reply of the model."""


CONTEXT_END = "context"
# The texts of SGLang v0.5.19's 400 BadRequest for a prompt past the served context (srt/managers/tokenizer_manager.py
# _validate_one_request, srt/managers/utils.py validate_input_length), which no retry can serve
CONTEXT_REFUSALS = ("context length", "maximum allowed length")


def context_refusal(exc):
    """Whether an error of a request is the server refusing a prompt that does not fit the served context."""
    response = getattr(exc, "response", None)
    return response is not None and response.status_code == 400 and any(text in response.text for text in CONTEXT_REFUSALS)


class CompletionRoute:
    """One endpoint, one served model and the records of one run.

    temperature, when given, replaces the temperature of every call (a sampling run for training data), where the paper's
    agents pass 0.0 on every call. reply_room, on a run with the CausalDiagnosis MCP, replaces the max_tokens of every call by
    the room the served context leaves its prompt (room_of)."""

    def __init__(self, base_url, api_key, model, temperature=None, client=None, deadline=None, reply_room=False):
        self.base_url, self.model, self.temperature = base_url.rstrip("/"), model, temperature
        self.reply_room = reply_room
        self.deadline = deadline  # the monotonic deadline of the row's wall limit, else the one the driver names (CausalDiagnosis.utils.model_request_retry)
        # no time limit of the client's own: every request is bounded by the row's deadline (CausalDiagnosis.utils.model_request_retry.post_with_retries)
        self._client = client or httpx.Client(timeout=None, headers={"Authorization": f"Bearer {api_key}"})
        self.calls: list[dict] = []
        self._agents = [SUPERVISOR]
        self._context_length = None
        self._stopped = threading.Event()
        self.stop_reason = None

    # ----- the agent whose calls are recorded -----
    @contextmanager
    def agent(self, name):
        self._agents.append(name)
        try:
            yield
        finally:
            self._agents.pop()

    def stop(self, reason):
        """Makes the next call raise RunStopped, from any thread."""
        self.stop_reason = self.stop_reason or reason
        self._stopped.set()

    def check(self):
        if self._stopped.is_set():
            raise RunStopped(self.stop_reason)

    def stopped(self):
        return self._stopped.is_set()

    # ----- the three routes -----
    def _request(self, method, path, body=None, model_reply=True):
        """One request of the route under the one retry policy (CausalDiagnosis.utils.model_request_retry): no count of attempts, each attempt given the time
        left before the row's deadline, so no request outlives the episode's wall limit. model_reply False marks a request
        that is no model reply, the model listing and the tokenize count, as CausalDiagnosis.utils.model_reply_room marks them, so that its answer never
        stands for the episode's first reply, whose cached tokens the driver records."""
        url = f"{self.base_url}/{path}"
        keywords = {} if body is None else {"json": body}

        def send(timeout):
            return self._client.request(method, url, **keywords) if timeout is None else self._client.request(method, url, timeout=timeout, **keywords)
        response = post_with_retries(send, deadline=self.deadline if self.deadline is not None else current_deadline(),
                                     delay_s=RETRY_DELAY_S, statuses=RETRY_STATUS, model_reply=model_reply)
        response.raise_for_status()
        return response

    def _post(self, path, body, model_reply=True):
        return self._request("POST", path, body, model_reply).json()

    def generate(self, prompt, temperature=0.0, max_tokens=500, stop=None, seed=None):
        """watsonx_llm's result for one call: generated_text, generated_token_count, input_token_count, stop_reason."""
        self.check()
        stop = list(stop or [])
        temperature = self.temperature if self.temperature is not None else temperature
        if self.reply_room:
            max_tokens = self.room_of(prompt)
        body = {"model": self.model, "prompt": prompt, "temperature": temperature, "max_tokens": max_tokens,
                "min_tokens": PAPER_DECODING["min_tokens"], "stop": stop, "no_stop_trim": PAPER_DECODING["no_stop_trim"]}
        if seed is not None:
            body["seed"] = seed
        record = {"agent": self._agents[-1], "kind": KINDS.get(tuple(stop), "other"), "prompt": prompt, "stop": stop,
                  "max_tokens": max_tokens, "temperature": temperature, "model": self.model, "started_at": time.time(), "cached_tokens": None}
        with episode_trace.recorded(episode_trace.MODEL_CALL, self.calls, record):
            started = time.perf_counter()
            try:
                answer = self._post("completions", body)
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                if context_refusal(exc):
                    # the end of the served context ends the run, as RunStopped so that no handler of ReActXen holds it back
                    self.stop(CONTEXT_END)
                    raise RunStopped(CONTEXT_END) from exc
                raise
            finally:
                record["seconds"] = round(time.perf_counter() - started, 2)
                record["ended_at"] = time.time()  # the call's span in wall time, for the drivers' split of the thinking time
            choice, usage = answer["choices"][0], answer.get("usage") or {}
            record.update(reply=choice["text"], input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                          cached_tokens=reported_cached_tokens(usage), finish_reason=choice.get("finish_reason"), matched_stop=choice.get("matched_stop"))
        from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import token_clock  # at the call, so that importing the port loads no file of the servers

        token_clock.add_generated(record["output_tokens"])
        stop_reason = ("max_tokens" if choice.get("finish_reason") == "length"
                       else "stop_sequence" if isinstance(choice.get("matched_stop"), str) else "eos_token")
        return {"generated_text": choice["text"], "generated_token_count": record["output_tokens"],
                "input_token_count": record["input_tokens"], "stop_reason": stop_reason}

    def room_of(self, prompt):
        """The max_tokens of a request that caps no reply on the completion route, which reads an absent max_tokens as 16 tokens:
        the served context less the tokenize count of the prompt less ENGINE_MARGIN, the most the engine generates, which also
        keeps the request clear of the engine's refusal of a prompt and a reply that reach the context. A prompt that leaves no
        room ends the run with the end context, as the engine's refusal of it does."""
        try:
            room = ReplyRoom(self.context_length(), self.count_tokens).max_tokens(prompt) - ENGINE_MARGIN
        except ContextEnd:
            room = 0
        if room < 1:
            self.stop(CONTEXT_END)
            raise RunStopped(CONTEXT_END)
        return room

    def count_tokens(self, text):
        self.check()
        # /v1/tokenize reports the tokenizer's model_max_length (2^63 - 1 for granite), not the served context, so the
        # context length comes from /v1/models alone
        started_at, started = time.time(), time.perf_counter()
        try:
            return self._post("tokenize", {"model": self.model, "prompt": text}, model_reply=False)["count"]
        except Exception as exc:
            # a count that failed after its retries leaves a record, as a failed generation does, so that it reaches the row
            self.calls.append({"agent": self._agents[-1], "kind": "tokenize", "prompt": text, "model": self.model, "started_at": started_at, "cached_tokens": None,
                               "error": f"{type(exc).__name__}: {exc}", "seconds": round(time.perf_counter() - started, 2), "ended_at": time.time()})
            episode_trace.ended(episode_trace.MODEL_CALL, len(self.calls) - 1, self.calls[-1])
            raise

    def context_length(self):
        """The context length the server serves the model with, from /v1/models: the card of the model, else that of its
        base model, since SGLang lists the adapter of a base:adapter name as a card of its own without a context length
        and splits the name at its first colon (serving_base._parse_model_parameter), by the rule the driver records
        (CausalDiagnosis.utils.model_reply_room.card_context_length)."""
        if self._context_length is None:
            cards = self._request("GET", "models", model_reply=False).json().get("data") or []
            length = card_context_length(cards, self.model)
            if length is None:
                raise RouteSetupError(f"{self.base_url}/models names no context length for {self.model}")
            self._context_length = length
        return self._context_length

    def close(self):
        self._client.close()


_ACTIVE: list[CompletionRoute] = []


def active_route():
    if not _ACTIVE:
        raise RuntimeError("no completion route is active: the Agent-As-Tool runner activates one for each run")
    return _ACTIVE[-1]


@contextmanager
def activated(route):
    _ACTIVE.append(route)
    try:
        yield route
    finally:
        _ACTIVE.remove(route)
