"""The chat route that the agents and the judge of the benchmark send their requests through.

AssetOpsBench's OpenAICompatBackend (src/llm/openai_compat.py) reads its endpoint from the router's environment variables
at every call, caps every reply at 2048 tokens, lets the SDK client retry a failed request and reports no reason for the end
of a reply. EndpointChatBackend keeps that request shape and adds what a run of the benchmark needs:
- an endpoint bound to the backend, so that an agent and a judge in one process never share an address;
- request fields beyond the OpenAI ones (extra_body), such as SGLang's chat_template_kwargs, sent only when given;
- the request field response_format of a call that binds its reply to a JSON schema, sent only with that call;
- no reply cap of the harness: the engine gives the reply the rest of the context, and a reply that ends by length is checked
  against that room (CausalDiagnosis.utils.model_reply_room), read from the endpoint's model listing and tokenize route;
- the engine's refusal of a prompt past the context raised as ContextEnd, the end of the context and never a failed attempt;
- a client that sends every request once, while the benchmark's one retry policy (CausalDiagnosis.utils.model_request_retry) attempts a failed request
  again with no count of attempts, the registered wait, the row's deadline inside an episode and the read timeout outside;
- the reasoning text as the reply when a server with a reasoning parser returns no content, and the server's reason for the
  end of a reply;
- the settings of the route for the record of a run.
The file sits beside the drivers, so that episode_judge.py reads it without importing the agent package.
"""

import os
from dataclasses import dataclass

from llm import LLMBackend  # noqa: F401  re-exported: the core reaches upstream's base class only through this module
from llm.base import LLMResult
from llm.openai_compat import OpenAICompatBackend
from CausalDiagnosis.utils.model_request_retry import READ_TIMEOUT_S, RETRY_DELAY_S, STATUSES, current_deadline, post_with_retries
from CausalDiagnosis.utils.model_reply_room import ContextEnd, NoReplyRoom, ReplyRoom, endpoint_room, is_context_refusal
from llm.routers import resolve_router_creds

__all__ = ["NO_THINKING", "ChatResult", "EndpointChatBackend", "endpoint_backend"]

# The request field that asks a model served with a reasoning parser to answer without thinking. A judge or a reflection model
# whose reply thinks comes back as thinking without the JSON it is asked for, so both ask for no thinking.
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}

ROUTE = "/v1/chat/completions"
# The client sends every request once: CausalDiagnosis.utils.model_request_retry attempts a failed request again and writes each failed attempt to the
# episode's ledger.
CLIENT_MAX_RETRIES = 0
CLIENT_TIMEOUT_S = READ_TIMEOUT_S
CLIENT_CONNECT_TIMEOUT_S = 5.0
# the policy as the record of a run states it
CLIENT_RETRY_STATUS = "no answer, " + ", ".join(str(s) for s in STATUSES)
CLIENT_RETRY_DELAY_S = RETRY_DELAY_S
# the template arguments that change how the chat route renders a prompt, so the tokenize count must send them too
_RENDERING_FIELDS = ("chat_template_kwargs", "tools")


@dataclass(frozen=True)
class ChatResult(LLMResult):
    """LLMResult with the server's reason for stopping, when it reports one ("stop", "length"), and the thinking that a server
    with a reasoning parser moved out of the text, when it reports some."""

    finish_reason: str | None = None
    reasoning: str | None = None


class EndpointChatBackend(OpenAICompatBackend):
    """AssetOpsBench's OpenAI compatible backend with an endpoint bound to it, the room of a reply and the record a run needs.

    Args:
        model_id: prefixed model string, e.g. ``"tokenrouter/granite"``.
        base_url, api_key: the endpoint, given together or not at all; without them the router's environment variables are
            read at every call, as the released backend reads them.
        extra_body: request fields beyond the OpenAI ones, sent only when given.
        max_tokens: a caller's own cap; by default none is sent and the engine gives the reply the rest of the context.
        deadline: the monotonic deadline of the row's wall limit; without one, the deadline the driver names (CausalDiagnosis.utils.model_request_retry).
        room: the context and the prompt count, built from the endpoint when first needed; a test may pass one.
    """

    def __init__(self, model_id: str, base_url: str | None = None, api_key: str | None = None,
                 extra_body: dict | None = None, max_tokens: int | None = None, deadline: float | None = None,
                 room: ReplyRoom | None = None) -> None:
        super().__init__(model_id)
        if bool(base_url) != bool(api_key):
            raise ValueError("base_url and api_key are given together or not at all")
        self._base_url, self._api_key = base_url, api_key
        self.extra_body = extra_body
        self.max_tokens, self._room, self.longest_reply = max_tokens, room, 0
        self.deadline = deadline

    takes_response_format = True  # a call can bind its reply to a JSON schema (generate_with_usage)

    def _endpoint(self) -> tuple[str, str]:
        if self._base_url and self._api_key:
            return self._base_url, self._api_key
        creds = resolve_router_creds(self._model_id)  # strict: clear error if unset
        return creds.base_url, creds.api_key

    def generate_with_usage(self, prompt: str, temperature: float = 0.0, response_format: dict | None = None) -> ChatResult:
        """One chat request; response_format, where given, is the request field that binds the reply to a JSON schema."""
        import httpx
        from openai import OpenAI

        base_url, api_key = self._endpoint()
        client = OpenAI(base_url=base_url, api_key=api_key, max_retries=CLIENT_MAX_RETRIES,
                        timeout=httpx.Timeout(CLIENT_TIMEOUT_S, connect=CLIENT_CONNECT_TIMEOUT_S))
        extra = {"extra_body": self.extra_body} if self.extra_body else {}
        cap = {} if self.max_tokens is None else {"max_tokens": self.max_tokens}
        bound = {} if response_format is None else {"response_format": response_format}

        def send(timeout):
            return client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                **cap,
                **bound,
                **extra,
                # None is a read without a bound inside an episode on a row without a wall limit (CausalDiagnosis.utils.model_request_retry.attempt_timeout),
                # with the client's own connect timeout
                timeout=httpx.Timeout(None, connect=CLIENT_CONNECT_TIMEOUT_S) if timeout is None else timeout,
            )
        try:
            response = post_with_retries(send, deadline=self.deadline if self.deadline is not None else current_deadline())
        except Exception as exc:
            if is_context_refusal(exc):
                raise ContextEnd(str(exc)) from exc
            raise
        usage = getattr(response, "usage", None)
        choice = response.choices[0]
        # A server with a reasoning parser returns the thinking in reasoning_content and may leave content empty when
        # the token budget ends inside the thinking, so the reasoning is the reply when nothing else came back.
        message = choice.message
        result = ChatResult(
            text=message.content or getattr(message, "reasoning_content", None) or "",
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            finish_reason=getattr(choice, "finish_reason", None),
            reasoning=getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None),
        )
        # The longest reply counts only replies that ended on their own (finish_reason stop). A reply that ends by length was cut at
        # its room (the context less the prompt); it is a turn without a usable plan and adds nothing to the term.
        if result.finish_reason == "stop":
            self.longest_reply = max(self.longest_reply, result.output_tokens)
        if result.finish_reason == "length" and self.max_tokens is None:
            self.room().check_length_finish(result.input_tokens, result.output_tokens)
        return result

    def room(self) -> ReplyRoom:
        """The room of this backend's endpoint: max_model_len of its model listing and the tokenize count of a chat prompt with
        the same template arguments, built once."""
        if self._room is None:
            try:
                base_url, api_key = self._endpoint()
            except Exception as exc:  # noqa: BLE001 - no endpoint named
                raise NoReplyRoom(f"{self._model_id} has no endpoint to read its context from") from exc
            extra = {k: v for k, v in (self.extra_body or {}).items() if k in _RENDERING_FIELDS}
            self._room = endpoint_room(base_url, api_key, self._model_name, chat=True, extra=extra)
        return self._room

    def request_fits(self, prompt: str) -> bool:
        """Whether the next request leaves room for a reply as long as the longest that ended on its own so far, counted by the
        endpoint's tokenize route as the chat route renders the prompt; the board of the CausalDiagnosis agent reads it."""
        return self.room().request_fits(prompt, self.longest_reply)

    def request_under_trigger(self, prompt: str) -> bool:
        """Whether the prompt's tokens alone stay at or below the trigger of compaction of the CausalDiagnosis agent, floor(context * 9 / 10),
        counted by the same tokenize route and context as request_fits (CausalDiagnosis.utils.model_reply_room)."""
        return self.room().request_under_trigger(prompt)

    def request_settings(self, temperature: float = 0.0) -> dict:
        """What this route sends with every request and how its client carries it, for the record of a run: the block
        decoding and the block transport. A setting that the route does not send is None."""
        return {"decoding": {"route": ROUTE, "max_tokens": self.max_tokens, "min_tokens": None, "temperature": temperature, "stop": None,
                             "no_stop_trim": None, "extra_body": self.extra_body},
                "transport": {"retries": None, "retry_status": CLIENT_RETRY_STATUS, "retry_delay_s": CLIENT_RETRY_DELAY_S,
                              "timeout_s": CLIENT_TIMEOUT_S, "connect_timeout_s": CLIENT_CONNECT_TIMEOUT_S}}


def endpoint_backend(model, base_url, extra_body=None):
    """The backend of a served model at an endpoint bound to it, with the key read from API_KEY, as the judge and the reflection
    model of GEPA call their endpoints; a missing key stops the program."""
    key = os.environ.get("API_KEY")
    if not key:
        raise SystemExit("API_KEY is not set")
    # the endpoint is bound to the backend, so the agent's and the judge's backends never share one address
    return EndpointChatBackend(f"tokenrouter/{model}", base_url=base_url, api_key=key, extra_body=extra_body)
